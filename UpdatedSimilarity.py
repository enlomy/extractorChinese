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

def similarity_score(query, document):
    # Calculate cosine similarity between query and document
    query_embedding = model.encode(query, convert_to_tensor=True)
    document_embedding = model.encode(document, convert_to_tensor=True)
    cosine_similarity = util.pytorch_cos_sim(query_embedding, document_embedding)
    return cosine_similarity
def top_similarity_score(query, document_list):
    # Calculate cosine similarity between query and document
    query_embedding = model.encode(query, convert_to_tensor=True)
    cosine_similarity_list = []
    for document in document_list:
        document_embedding = model.encode(document, convert_to_tensor=True)
        cosine_similarity = util.pytorch_cos_sim(query_embedding, document_embedding)
        cosine_similarity_list.append(cosine_similarity)
    # Get top 1 similarity score document
    top_similarity_document = document_list[cosine_similarity_list.index(max(cosine_similarity_list))]
    return top_similarity_document

def similarity_score_list(query_list, document_list):
    # Batch queries and documents for similarity score calculation
    query_embedding_list = model.encode(query_list, convert_to_tensor=True)
    document_embedding_list = model.encode(document_list, convert_to_tensor=True)
    cosine_similarity_list = util.pytorch_cos_sim(query_embedding_list, document_embedding_list)
    return cosine_similarity_list

def get_limit_score(cosine_similarity_list):
    # Get limit score for each query
    limit_score_list = []
    for cosine_similarity in cosine_similarity_list:
        if cosine_similarity > limit_score:
            limit_score_list.append(1)
        else:
            limit_score_list.append(0)
    return limit_score_list
 
# print("Chinese model encoding by embedding ...\n")
# corpus_embedding = model.encode(corpus, convert_to_tensor=True)
# top_k = max(5, len(corpus))
# # top_k = 1
# results = []
# print("Loop of queries ...")
# for query in queries:
#     query_embedding = model.encode(query, convert_to_tensor=True)

#     cos_scores = util.cos_sim(query_embedding, corpus_embedding)[0]
#     top_score, top_idx = torch.topk(cos_scores, k=top_k)

#     # If score is less than LIMIT_SCORE, ignore the training data
#     # if round(top_score.item(), 3) < limit_score:
#     #     continue

#     result = {
#         "query": query,
#         "score": round(top_score.item(), 3),
#         "document": corpus[top_idx.item()]
#     }
#     print(result)
#     print("\n\n")
#     results.append(result)
#     # for score, idx in zip(top_results[0], top_results[1]):
#     #     print(f'{round(score.item(), 3)} | {corpus[idx]}')
# # print(results)

# print("Export to files ...")
# print("Result count : ")
# print(len(results))

# with open(result_file_path, 'w', encoding="utf-8") as file:
#     json.dump(results, file, ensure_ascii=False, indent=4)
print(similarity_score("english","英语"))
print("Done!")