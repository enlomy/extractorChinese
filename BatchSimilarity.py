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
    # Max similarity score 
    max_similarity_score = max(cosine_similarity_list)
    # Get top 1 similarity score document
    top_similarity_document = document_list[cosine_similarity_list.index(max_similarity_score)]
    result = {
            "query": query,
            "document": top_similarity_document,
            "score": max_similarity_score
        }
    return result

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
def get_matching_documents(queries, document_list):
    # Get Matching documents section by high probability
    matching_list = []
    results = []
    query_count = len(queries)
    print(query_count)
    document_count = len(document_list)
    document_pitch = 10
    for index,query in enumerate(queries):
        # Select documents section by high probability
        if index / query_count * document_count - document_pitch < 0 :
            low_limit = 0  
            high_limit = int(index / query_count * document_count + document_pitch)
        elif index / query_count * document_count - document_pitch > document_count :
            low_limit = int(index / query_count * document_count - document_pitch)
            high_limit = document_count  
        else :
            low_limit =  int(index / query_count * document_count - document_pitch)
            high_limit = int(index / query_count * document_count + document_pitch)
        print(low_limit, high_limit)
        calc_document_list = document_list[low_limit:high_limit]
        
        result = top_similarity_score(query, calc_document_list)
 
        matching_list.append(result)
    return matching_list
with open(result_file_path, 'w', encoding="utf-8") as file:
    json.dump(get_matching_documents(queries,corpus), file, ensure_ascii=False, indent=4)
print("Done!")