from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.llms import HuggingFacePipeline 
from transformers import TextStreamer, TextIteratorStreamer
from transformers import pipeline
from langchain.text_splitter import  CharacterTextSplitter, RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from torch import cuda
from langchain_community.vectorstores import FAISS
import json
import re
from collections import deque 
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from flask import Flask, render_template, request, Response, stream_with_context
from flask_cors import CORS  # Import CORS

import gc
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import lorem

def segment_long_text(long_text: str, max_length: int = 1000, sliding_thresh: int = 500):
    """
    This function use dynamic sliding window over a list of passages with each passage is split by new-line character.
    The output of this function is a list of paragraphs with overlapping text. The step size (number of passages shifted)
    of the sliding window so that the total number of characters of shifted passages lower than `sliding threshold`.
    varies depending on 
    - long_text: a long text to be split into passages with length shorter than `max_length`
    - max_length: maximum length in character for each paragraph
    - output: a list of paragraphs derives from the input long text
    """
    paragraphs = []

    passages = re.split(r'\n+', long_text)
    lengths = [len(passage) for passage in passages]


    # print (passages)
    # print (lengths)
    # print (sum(lengths))

    queue = deque([])
    total_length = 0
    shift = 0
    for idx, length in enumerate(lengths):
        # print ("idx: ", idx)
        if total_length > 0:
            if total_length + length < max_length:
                queue.append(length)
                total_length += length
            elif total_length + length == max_length:
                    queue.append(length)
                    total_length = max_length
                    joint_passages = ' '.join(passages[idx_2] for idx_2 in range(idx - len(queue) + 1, idx + 1))
                    paragraphs.append(joint_passages)
                    # print (">> total_length + length = max_length")
                    # print ("total_length: ", total_length)
                    # print ("queue: ", queue)

                    # print (idx, [ind for ind in range(idx - len(queue) + 1, idx + 1)])

                    shift = 0
                    while len(queue) > 0 and shift + queue[0] < sliding_thresh:
                        shift += queue[0]
                        total_length -= queue.popleft()
            elif total_length + length > max_length:
                joint_passages = ' '.join(passages[idx_2] for idx_2 in range(idx - len(queue), idx))
                paragraphs.append(joint_passages)
                # print (">> total_length + length > max_length")
                # print ("total_length: ", total_length)
                # print ("queue: ", queue)

                # print (idx, [ind for ind in range(idx - len(queue), idx)])
              
                if length >= max_length:
                    paragraphs.append(passages[idx])
                    # print (">> total_length + length > max_length & length >= max_length")
                    # print ("total_length: ", total_length)
                    # print ("queue: ", queue)
                    # print (idx, [idx])
                    queue.clear()
                    total_length = 0
                else:
                    queue.append(length)
                    total_length += length
                    
                    shift = 0
                    while len(queue) > 0 and shift + queue[0] < sliding_thresh:
                        shift += queue[0]
                        total_length -= queue.popleft()                            
                    
                    # shift as long as total_length < max_length
                    while total_length >= max_length:
                        total_length -= queue.popleft()
        else:
            if length < max_length:
                queue.append(length)
                total_length += length
            else:
                paragraphs.append(passages[idx])
                # print (">> total_length = 0 & length < max_length")
                # print ("total_length: ", total_length)
                # print ("queue: ", queue)
                # print (idx, [idx])
    if total_length > 0:
        joint_passages = ' '.join(passages[idx_2] for idx_2 in range(len(lengths) - len(queue), len(lengths)))
        paragraphs.append(joint_passages)

    return paragraphs

callbacks = [StreamingStdOutCallbackHandler()]

def load_model():
    model_path = "Viet-Mistral/Vistral-7B-Chat"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                        trust_remote_code=True,
                                                        device_map="auto",
                                                        stream = True
                                                        #attn_implementation="flash_attention_2"
                                                )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    streamer = TextStreamer(tokenizer,skip_prompt=True,skip_special_tokens=True)
    # streamer = TextIteratorStreamer(tokenizer,skip_prompt=True,skip_special_tokens=True)

    generate_text = pipeline(
        model=model,
        tokenizer=tokenizer,
        task='text-generation',
        max_new_tokens = 4000,
        temperature = 0.01,
        do_sample = True,
        # stream = True
        streamer = streamer
    )
    llm = HuggingFacePipeline(pipeline=generate_text) 
    return llm

# vector_db_path = "/home/ptpm/chatbot/vectorstores/db_faiss"
# embed_model_id = "/home/ptpm/chatbot/sup-SimCSE-VietNamese-phobert-base"

# device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
def create_db_from_files(vector_db_path,embed_model_id,device):
    with open("/home/ptpm/chatbot/question_answer_pairs.json",'r') as f:
        data = json.load(f)
    chunks = []
    for key,value in data.items():
        tmp = segment_long_text(value)
        chunks += tmp 
    embed_model = HuggingFaceEmbeddings(
        model_name=embed_model_id,
        model_kwargs={'device': device},
        #encode_kwargs={'device': device, 'batch_size': 32}
    )
    db = FAISS.from_texts(chunks, embed_model)
    db.save_local(vector_db_path)
    return db, chunks

def creat_prompt(template):
    prompt = PromptTemplate(template = template, input_variables=["context", "question"])
    return prompt

def create_qa_chain(prompt, llm,retriever):
    llm_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type= "stuff",
        retriever = retriever,
        return_source_documents = False,
        chain_type_kwargs= {'prompt': prompt}

    )
    return llm_chain

def read_vectors_db(embed_model_id,vector_db_path):
    # Embeding
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    embed_model = HuggingFaceEmbeddings(
        model_name=embed_model_id,
        model_kwargs={'device': device},
        #encode_kwargs={'device': device, 'batch_size': 32}
    )
    db = FAISS.load_local(vector_db_path, embed_model,allow_dangerous_deserialization=True)
    return db

def init_chatbot():
    vector_db_path = "/home/ptpm/chatbot/vectorstores/db_faiss"
    embed_model_id = "/home/ptpm/chatbot/sup-SimCSE-VietNamese-phobert-base"

    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    db, chunks = create_db_from_files(vector_db_path,embed_model_id,device)
    db = read_vectors_db(embed_model_id,vector_db_path)
    bm25_retriever = BM25Retriever.from_texts(chunks, k=2)
    faiss_retriever = db.as_retriever(search_kwargs = {"k":4})
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], weights=[0.7, 0.3])

    llm = load_model()

    template = """<s>[INST] 
    Cho đoạn văn:
    {context}
    Sử dụng đoạn văn trên để trả lời câu hỏi một cách chính xác và đầy đủ nhất. Nếu câu trả lời không có trong thông tin được cung cấp, hãy trả lời rằng "MITIA không tìm thấy câ trả lời trong dữ liệu của ban!", đừng cố tạo ra câu trả lời.
    {question}
    Câu trả lời: [/INST]"""

    prompt = creat_prompt(template)
    llm_chain  =create_qa_chain(prompt, llm,retriever=ensemble_retriever)
    return llm_chain

# init chatbot chain
#llm_chain = init_chatbot()

# ------------------------------------------------------------------------------
#               INIT API
# ------------------------------------------------------------------------------
app = FastAPI()
class Query(BaseModel):
    query: str
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/chatbot')
async def answer(query: Query):
    if query:
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        
        response = llm_chain.invoke({"query": query.query})
        # answer = jsonify({"query": user_question, "answer": response['result'].strip()})
        answer = response['result'].strip()
        return answer
    
def my_generator():
    for _ in range(50):
        yield lorem.sentence()
        import time
        time.sleep(0.1)

@app.post('/stream_chatbot')
async def stream_answer(query: Query):
    return "OK"
    # if query:
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #     gc.collect()
    #     #res_stream = llm_chain.astream({"query": query.query})

    #     #return StreamingResponse(res_stream, media_type='text/event-stream')
    #     return StreamingResponse(my_generator(), media_type='text/event-stream')
    
if __name__ == "__main__":
    uvicorn.run("chatbot_kdovan:app", host="0.0.0.0", port=8002)
