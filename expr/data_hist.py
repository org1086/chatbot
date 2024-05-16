import json
from matplotlib import pyplot as plt

# load data 
qa_data = json.load(open("data/question_answer_pairs.json"))

questions = list(map(lambda d: len(d), qa_data.keys()))
answers = list(map(lambda d: len(d), qa_data.values()))

# question
plt.hist(questions, bins=15)
plt.xlabel('Questions')
plt.ylabel('Frequency')
plt.title('Question Length Distribution')
plt.savefig("expr/hists/Question-Length-Distribution.png")

# answer
plt.hist(answers, bins=15)
plt.xlabel('Answers')
plt.ylabel('Frequency')
plt.title('Answer Length Distribution')
plt.savefig("expr/hists/Answer-Length-Distribution.png")