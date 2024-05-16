import re
import os
from collections import deque 
import json
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from torch import cuda 
from langchain_community.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

def read_corpus(corpus_path: str = "data/question_answer_pairs.json"):
    with open(corpus_path,'r') as f:
        data = json.load(f)
    chunks = []
    questions = []
    for key,value in data.items():
        # tmp = segment_long_text(value)
        # chunks += tmp

        # collect questions
        questions.append(key)

        if len(value.split(" ")) <= 512:
            tmp = key + " " + value
            chunks.append(value)
        else:
            tmp = segment_long_text(value)
            chunks += tmp 
    
    return chunks, questions
        
def create_database(chunks, questions):

    if not (os.path.exists("dbs/chroma_db") and os.path.exists("dbs/chroma_db_q")):
        # init embedding model
        embed_model_id = "models/sup-SimCSE-VietNamese-phobert-base"
        device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
        embed_model = HuggingFaceEmbeddings(
            model_name=embed_model_id,
            model_kwargs={'device': device},
            #encode_kwargs={'device': device, 'batch_size': 32}
            )
        
        # embeding content
        Chroma.from_texts(
            texts = chunks,
            embedding = embed_model,
            persist_directory="dbs/chroma_db"
            )

        # embeding questions
        Chroma.from_texts(
        texts = questions,
        embedding = embed_model,
        persist_directory="dbs/chroma_db_q"
        )

def load_database(content_emd_path: str = "dbs/chroma_db", 
                  question_emd_path: str = "dbs/chroma_db_q"):

    embed_model_id = "models/sup-SimCSE-VietNamese-phobert-base"
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    embed_model = HuggingFaceEmbeddings(
        model_name=embed_model_id,
        model_kwargs={'device': device},
        #encode_kwargs={'device': device, 'batch_size': 32}
        )
    content_emb_db = Chroma(persist_directory=content_emd_path, embedding_function=embed_model)
    question_emb_db = Chroma(persist_directory=question_emd_path, embedding_function=embed_model)
    
    return content_emb_db, question_emb_db

def get_qa_prompt():
    template = """<s>[INST] 
        Cho đoạn văn:
        {context}
        Sử dụng đoạn văn trên để trả lời câu hỏi dưới đây một cách chính xác và đầy đủ nhất. Nếu câu trả lời không có trong thông tin được cung cấp, hãy trả lời rằng "[GPT86] không tìm thấy câu trả lời trong dữ liệu chuyển đổi số đã có!", đừng cố tạo ra câu trả lời. Lưu ý, chỉ trả lời bằng tiếng Việt và không dùng ký tự đặc biệt.
        Câu hỏi:
        {question}
        Câu trả lời: [/INST]"""

    return ChatPromptTemplate.from_template(template)

def get_chatchit_prompt():    
    template = """<s>[INST]
        Bạn là một trợ lý Tiếng Việt nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể, đồng thời giữ an toàn. Câu trả lời của bạn không nên chứa bất kỳ nội dung gây hại, phân biệt chủng tộc, phân biệt giới tính, độc hại, nguy hiểm hoặc bất hợp pháp nào. Hãy đảm bảo rằng các câu trả lời của bạn không có thiên kiến xã hội và mang tính tích cực. Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác. Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trẳ lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch.
        - Người dùng: Xin chào!
        - Chatbot trả lời: Chào đồng chí! Rất vui được nói chuyện với đồng chí!
        - Người dùng: chán quá đi mất
        - Chatbot trả lời: Ai cũng có lúc chán nản, mệt mỏi. Chuyện này như cơm bữa ấy mà :) Tôi có thể giúp đỡ gì đồng chí không?
        - Người dùng: Tạm biệt!
        - Chatbot trả lời: Tạm biệt! Hẹn gặp lại đồng chí!
        - Người dùng hỏi: {question}
        - Chatbot trả lời: [/INST]"""

    return ChatPromptTemplate.from_template(template)

def init_chatbot():
    chunks, questions = read_corpus()
   
    # create embeding database - once
    create_database(chunks=chunks, questions=questions)

    # load embedding dbs
    content_emb_db,question_emb_db = load_database()

    content_retriever = content_emb_db.as_retriever(search_kwargs={"k": 2})
    question_retriever = question_emb_db.as_retriever(search_kwargs={"k": 5})

    bm25_content_retriever = BM25Retriever.from_texts(chunks, k=2)
    bm25_question_retriever = BM25Retriever.from_texts(questions, k=5)

    ensemble_content_retriever = EnsembleRetriever(
                        retrievers = [bm25_content_retriever, content_retriever], 
                        weights=[0.6, 0.4])
    
    ensemble_question_retriever = EnsembleRetriever(
                        retrievers = [bm25_question_retriever, question_retriever], 
                        weights=[0.6, 0.4])                        

    ollama = Ollama(base_url='http://192.168.1.99:11434',model="ontocord/vistral",temperature=0.01)
    
    # init QA chain
    qa_prompt = get_qa_prompt()
    qa_chain = (
        {"context": ensemble_content_retriever , "question": RunnablePassthrough()}
        | qa_prompt
        | ollama
        | StrOutputParser()
    )

    # init chatchit chain
    chatchit_prompt = get_chatchit_prompt()
    chatchit_chain = (
        {"question": RunnablePassthrough()}
        | chatchit_prompt
        | ollama
        | StrOutputParser()
    )

    return qa_chain, chatchit_chain, ensemble_question_retriever

def get_relevant_questions(ensemble_question_retriever: EnsembleRetriever, question: str, top_k: int = 5):

    relevant_questions = ensemble_question_retriever.invoke(normalize_question(question))

    results = {"question": question, "relevant_questions": []}
    unique_questions = []
    lower_question = question.lower()
    for doc in relevant_questions:
        lower_rel_question = doc.page_content.lower()
        if lower_rel_question != lower_question:
            if lower_rel_question not in unique_questions:
                results["relevant_questions"].append(doc.page_content)
                unique_questions.append(lower_rel_question)
    
    return results

def normalize_question(question: str):
    return question.lower().rstrip("? ")

def segment_long_text(long_text: str, max_length: int = 512, sliding_thresh: int = 256):
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

if __name__ == "__main__":
    query = "Công nghệ thông tin là gì?"
    
    chunks, questions = read_corpus()
   
    # create embeding database - once
    create_database(chunks=chunks, questions=questions)

    # load embedding dbs
    content_emb_db,question_emb_db = load_database()

    content_retriever = content_emb_db.as_retriever(search_kwargs={"k": 2})

    bm25_content_retriever = BM25Retriever.from_texts(chunks, k=2)

    ensemble_content_retriever = EnsembleRetriever(
                        retrievers = [bm25_content_retriever, content_retriever], 
                        weights=[0.6, 0.4])
    
    print (f"-----------------------------------------------------------------------")
    print (f">> semantic content retriver: {content_retriever.invoke(query)}")
    print (f">> bm25 content retriver: {bm25_content_retriever.invoke(query)}")
    print (f"ensemble content retriver: {ensemble_content_retriever.invoke(query)}")
