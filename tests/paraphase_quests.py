from openai import OpenAI
import json
import csv
import random
import requests
from tqdm import tqdm

CHATBOT_API_ENDPOINT = 'http://localhost:5001/chatbot'

def bot_answer(question: str) -> str:
    res = requests.post(CHATBOT_API_ENDPOINT, json={'user_question': question})
    
    return res.text if res.ok else ""

# load questions to paraphase
dict_qa = json.load(open("../data/question_answer_pairs.json", encoding='utf-8'))
chosen_indexes = random.choices(list(range(0,len(dict_qa.keys()))), k= 30)

questions = [q for idx,q in enumerate(list(dict_qa.keys())) if idx in chosen_indexes]
contexts = [c for idx,c in enumerate(list(dict_qa.values())) if idx in chosen_indexes]

client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4o",
    messages= [
        {
            "role": "system", 
            "content": "Bạn là trợ lý AI có kiến thức sâu rộng về ngôn ngữ tiếng Việt."
        },
        {
            "role": "user", 
            "content": f"""
                Với mỗi một câu hỏi được yêu cầu, hãy viết lại thành 3 câu hỏi tương tự nhưng phải đảm bảo ý nghĩa của câu hỏi là giống với câu hỏi gốc.
                Danh sách các câu hỏi cần viết lại như sau:
                {questions}
                
                Hãy trả về kết quả dạng dữ liệu JSON, trong đó, có `key` là câu hỏi gốc, và `value` là danh sách 3 câu hỏi được viết lại với nghĩa tương đồng.
                LƯU Ý: KHÔNG ĐƯỢC SỬA ĐỔI CÂU HỎI GỐC TRONG DANH SÁCH CÁC CÂU HỎI ĐẦU VÀO!
            """
        }
    ]
)

api_json_content = completion.choices[0].message.content
start_idx = api_json_content.index('```json') + 7
end_idx = api_json_content.rindex('```')
pure_json_content = api_json_content[start_idx:end_idx]

json_data = json.loads(pure_json_content)

print (json_data)

with open("../data/paraphased_questions_top30.csv", 'w', encoding='utf-8') as f:
    csv_writer = csv.DictWriter(
        f, 
        fieldnames=['question', 'type', 'paraphased_q_score', 'context', 'answer', 'answer_score', 'paraphased_q_answer_score']
    )
    csv_writer.writeheader()
    rows = []
    used_context_indexes = []
    for idx,(q,p) in tqdm(enumerate(json_data.items())):
        # if LLM changed origin question (key of dict not matched) -> match with index
        if q not in dict_qa.keys():
            print (f"#row{idx}: Used context index instead of question as key!")
            used_context_indexes.append(idx)
            context = contexts[idx]
        else:
            context = dict_qa[q]
            
        rows.append({
            'question': q, 
            'type': 'root', 
            'paraphased_q_score': "", 
            'context': context, 
            'answer': bot_answer(q).strip(), 
            'answer_score': "", 
            'paraphased_q_answer_score': ""
        })
        
        for p_i in p:
            rows.append({
                'question': p_i, 
                'type': 'paraphased', 
                'paraphased_q_score': "", 
                'context': "", 
                'answer': bot_answer(p_i).strip(), 
                'answer_score': "", 
                'paraphased_q_answer_score': ""
            })
    csv_writer.writerows(rows)

print (f"used_context_indexes: \n> total: {len(used_context_indexes)}, \n> details: {used_context_indexes}")