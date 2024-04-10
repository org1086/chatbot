import gc
import torch

from flask import Flask, request, jsonify, Response
from flask_cors import CORS

from utils import *

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app)

qa_chain, chatchit_chain, question_retriever = init_chatbot()

def answer_question(user_question):
    
    answer = qa_chain.stream(normalize_question(user_question))
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    return answer

def answer_chatchit(user_question):
    
    answer = chatchit_chain.stream(user_question)
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

@app.route('/chatchit', methods=['GET', 'POST'])
def ask_chatchit():
    user_question = request.get_json()["user_question"]
    if user_question:
        # answer question (generator object)
        answer = answer_chatchit(user_question)
   
        return Response(answer)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=False)
