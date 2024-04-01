import json
from os.path import join

src_json_fpath = "data/data.json"
dst_json_fpath = "data/question_answer_pairs.json"

with open(dst_json_fpath, 'w', encoding='utf-8') as fout:
    src_data = json.load(open(src_json_fpath, 'r', encoding='utf-8'))
    qa_pairs_dict = {}
    for d_item in src_data:
        qa_pairs_dict[d_item["question"][0]] = d_item["answer_text"]

    json.dump(qa_pairs_dict, fout, ensure_ascii=False, indent=4)
    print (f"> Total {len(qa_pairs_dict)} question-answer pairs!")