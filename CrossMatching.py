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
def get_matching_documents(queries, corpus):
    matching_documents = []
    querylist_length = len(queries)
    documentlist_length = len(corpus)
    
    for query in queries:
        query_embedding = model.encode(query, convert_to_tensor=True)
        for document in corpus:
            document_embedding = model.encode(document, convert_to_tensor=True)
            cosine_similarity = util.pytorch_cos_sim(query_embedding, document_embedding)
            if cosine_similarity > limit_score:
                result = {
                    "query": query,
                    "document": document,
                    "score": cosine_similarity
                }
                matching_documents.append(result)
    return matching_documents

print("Calculate Similarity Scores ...")

matching_documents = get_matching_documents(queries, corpus)

print("Save Results ...")
with open(result_file_path, 'w', encoding="utf-8") as file:
    json.dump(matching_documents, file, ensure_ascii=False, indent=4)
print("Done!")