import pandas as pd
import json
from utils.model_factory import get_llm_model
from retrieval_techniques.lexical_retrieval_based_conf import generate_answers_for_queries
from sentence_transformers import SentenceTransformer, util
from langchain.messages import SystemMessage, HumanMessage
from collections import defaultdict

MODEL_NAME = "all-MiniLM-L6-v2"
RRF_K = 60 # standard constant for RRF (Reciprocal Rank Fusion)
# creating sentence transformer model
sentence_transformer_model = SentenceTransformer(MODEL_NAME)
"""
    function that performs reciprocal rank fusion
    
    ranking_lists: A list of lists. Each inner list contains (doc_index, score) tuples
                   sorted by score for each query
    Returns: A dict of {index: fused_score}
"""
def apply_reciprocal_rank_fusion(ranking_lists):
    fused_scores = defaultdict(float)

    for ranking in ranking_lists:
        for rank, (doc_index, _) in enumerate(ranking):
            # RRF Formula: 1 / (k + rank)
            # Rank used here is 0-based here, so we add 1 to match standard formula
            # logic becasue standard RRF usually assumes 1-based rank 
            fused_scores[doc_index] += 1 / (RRF_K + (rank + 1))
    
    return fused_scores

# function to get most relevant three passages using RAG Fusion
def get_most_relevant_three_passages_with_rrf(row, model_name):
    passages = row.loc["passages"]
    passages = json.loads(passages)
    original_question = row.loc["query"]
    
    fusion_prompt = SystemMessage(f"""You are a helpful assistant that generates 
                                  multiple search queries based on a single input query.
                                  """)
    
    user_prompt = HumanMessage(f"""search queries related to the following question and queries are from MS MARCO question answering dataset.\n\n Focus on different aspects, synonyms, or related sub-topics.\n\n Original Question: {original_question} \n\nOuput ONLY the 4 queries, one per line. Do not number them.""")
    
    model = get_llm_model(model_name, temperature=0.3)
    message = [fusion_prompt, user_prompt]
    response = model.invoke(message)
    generated_queries = response.content
   
    print("--------------GENERATED QUERIES--------------") 
    print(generated_queries)
    print("--------------GENERATED QUERIES--------------")
    generated_queries = [query.strip() for query in generated_queries.split("\n")]
    
    # extracting passage texts
    passage_texts = [passage["passage_text"] for passage in passages]
    
    all_queries = [original_question] + generated_queries
    # generating query and passage embeddings
    query_embeddings = sentence_transformer_model.encode(all_queries, convert_to_tensor=True, show_progress_bar=True)
    passage_embeddings = sentence_transformer_model.encode(passage_texts, convert_to_tensor=True, show_progress_bar=True) 
    
    all_rankings = []
    for i in range(len(all_queries)):
        # calculating cosine similarity for query[i] vs all passages
        # query_embeddings[i] is 1D, passage_embedding is 2D
        scores = util.cos_sim(query_embeddings[i], passage_embeddings)[0]

        # creating a list of (index, score) tuples
        scored_indices = []
        for idx, score in enumerate(scores):
            scored_indices.append((idx, score.item()))

        # sorting scores for specific query in descending order
        scored_indices.sort(key=lambda x: x[1], reverse=True)
        all_rankings.append(scored_indices)
            
    # applying reciprocal rank fusion
    fused_scores_dict = apply_reciprocal_rank_fusion(all_rankings)
    # converting dictionary to list of tuples and sorting by fusion score
    final_rankings = sorted(fused_scores_dict.items(), key=lambda x: x[1], reverse=True)
    
    # selecting most relevant three passages
    most_relevant_three_passage_indexes = [idx for idx, score in final_rankings[:3]]
    most_relevant_three_passages = [passage_texts[idx] for idx in most_relevant_three_passage_indexes]
    
    return json.dumps(most_relevant_three_passages)
    
# function to run the RAG fusion-based retrieval configuration
def run_retrieval(model_name):        
    dataset_df = pd.read_csv("data\\ms_marco_qna_dataset.csv")
    print(dataset_df.head())
    # retrieving most relevant 3 passages for each query using Reciprocal RAG fusion and saving to dataframe
    dataset_df["most_relevant_three_passages"] = dataset_df.apply(lambda row: get_most_relevant_three_passages_with_rrf(row, model_name), axis=1)
    # generating answer for each query using most relevant 3 passages
    dataset_df["generated_answer"] = dataset_df.apply(lambda row: generate_answers_for_queries(row, model_name), axis=1) 
    # save the most relevant three passages along with generated answers
    dataset_df.to_csv("outputs\\ms_marco_qna_with_generated_answers_rrf.csv", sep=",", header=True, index=False)
    