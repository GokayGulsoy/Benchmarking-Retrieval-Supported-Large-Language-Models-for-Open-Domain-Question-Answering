import pandas as pd
import json
from retrieval_techniques.lexical_retrieval_based_conf import generate_answers_for_queries
from sentence_transformers import SentenceTransformer, util

MODEL_NAME = "all-MiniLM-L6-v2"
# creating sentence transformer model
model = SentenceTransformer(MODEL_NAME) 
# function to get 3 passages most relevant to query
def get_most_relevant_three_passages_with_sentence_transformer(row, isPRF=False):
    passages = row.loc["passages"]
    passages = json.loads(passages)
    
    # extracting passage texts
    passage_texts = [passage["passage_text"] for passage in passages]
    
    query = ""
    if isPRF:
        query = row.loc["prf_query"]
    else:
        query = row.loc["query"]
    
    # generating query and passage embeddings
    query_embedding = model.encode(query, convert_to_tensor=True, show_progress_bar=True)
    passage_embeddings = model.encode(passage_texts, convert_to_tensor=True, show_progress_bar=True) 
    
    # calculating cosine similarity scores
    cosine_sim_scores = util.cos_sim(query_embedding, passage_embeddings)[0]
    scored_passages = []
    for index, score in enumerate(cosine_sim_scores):
        scored_passages.append({
            "passage": passage_texts[index],
            "score": score.item()
        })
    
    # sorting passages descending by score end getting highest 3 scored pssages
    scored_passages.sort(key= lambda x: x["score"], reverse=True)
    most_relevant_three_passages = [passage["passage"] for passage in scored_passages[:3]]
    
    return json.dumps(most_relevant_three_passages)

# function to run semantic retrieval configuration
def run_retrieval(model_name):
    dataset_df = pd.read_csv("data\\ms_marco_qna_dataset.csv")
    print(dataset_df.head())
    # retrieving most relevant 3 passages for each query and saving to dataframe
    dataset_df["most_relevant_three_passages"] = dataset_df.apply(get_most_relevant_three_passages_with_sentence_transformer, axis=1)
    # generating answer for each query using most relevant 3 passages
    dataset_df["generated_answer"] = dataset_df.apply(lambda row: generate_answers_for_queries(row, model_name), axis=1) 
    # save the most relevant three passages along with generated answers 
    dataset_df.to_csv("outputs\\ms_marco_qna_with_generated_answers_semantic.csv", sep=",", header=True, index=False)
