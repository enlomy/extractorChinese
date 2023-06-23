from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
import os
import torch
import json

load_dotenv('.env')
# Load Limit Score constant from env
limit_score = float(os.environ['LIMIT_SCORE'])
input_cn_file_path = os.environ['OUT_CN_FILE_PATH']
input_en_file_path = os.environ['OUT_EN_FILE_PATH']
result_file_path = os.environ['RESULT_JSON_FILE_PATH']

# save model in current directory
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu', cache_folder='./')
# save model in models folder (you need to create the folder on your own beforehand)
# model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu', cache_folder='./models/')
print("Read Input Files ...")
# Read queries from file and split by line breaks
with open(input_en_file_path, "r", encoding='utf-8') as query_file:
    queries = query_file.read().split("\n\n")

# Read corpus from file and split by line breaks
with open(input_cn_file_path, "r", encoding='utf-8') as corpus_file:
    corpus = corpus_file.read().split("\n\n")

# print(corpus)


# Corpus with example sentences
# corpus = [
#     'I am a boy',
#     'What are you doing?',
#     'Can you help me?',
#     'A man is riding a horse.',
#     'A woman is playing violin.',
#     'A monkey is chasing after a goat',
#     'The quick brown fox jumps over the lazy dog'
# ]

# Query sentences:
# queries = ['你能幫助我嗎？', '我是男孩子', '一個女人正在拉小提琴']
print("Chinese model encoding by embedding ...\n")
# corpus_embedding = model.encode(corpus, convert_to_tensor=True)
top_k = 1
results = []
len_query = len(queries)
len_corpus = len(corpus)
pitch = int(0.01 * len_corpus)
print("Loop of queries ...")
for idx, query in enumerate(queries):
    corpus_index = int( idx / len_query * len_corpus )

    high_limit = len_corpus if len_corpus < corpus_index + pitch else corpus_index + pitch
    low_limit = 0 if 0 > corpus_index - pitch else corpus_index - pitch
    print("high_limit : ", high_limit)
    print("low_limit : ", low_limit)
    corpus_embedding = model.encode(corpus[low_limit:high_limit], convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)

    cos_scores = util.cos_sim(query_embedding, corpus_embedding)[0]
    top_score, top_idx = torch.topk(cos_scores, k=top_k)

    # If score is less than LIMIT_SCORE, ignore the training data
    # if round(top_score.item(), 3) < limit_score:
    #     continue

    result = {
        "query": query,
        "score": round(top_score.item(), 3),
        "document": corpus[top_idx.item()]
    }
    print(result)
    print("\n\n")
    results.append(result)

print("Export to files ...")
print("Result count : ")
print(len(results))

with open(result_file_path, 'w', encoding="utf-8") as file:
    json.dump(results, file, ensure_ascii=False, indent=4)
