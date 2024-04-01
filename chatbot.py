import gc
import torch

from flask import Flask, request, jsonify, Response
from flask_cors import CORS

from utils import *

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app)

chain, question_retriever = init_chatbot()
# answer = chain.stream("số hóa là gì?")

def answer_question(user_question):
    
    answer = chain.stream(user_question)
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    return answer

@app.route('/chatbot', methods=['GET', 'POST'])
def ask_question():
    user_question = request.get_json()["user_question"]
    if user_question:
        # answer question (generator object)
        answer = answer_question(user_question)
   
        return Response(answer)

@app.route('/suggest', methods=['GET', 'POST'])
def suggest_questions():
    user_question = request.get_data().decode('utf-8')
    
    if user_question:
        # suggest relevant questions
        relevant_questions = get_relevant_questions(question_retriever, user_question)
    
        return jsonify(relevant_questions)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)
