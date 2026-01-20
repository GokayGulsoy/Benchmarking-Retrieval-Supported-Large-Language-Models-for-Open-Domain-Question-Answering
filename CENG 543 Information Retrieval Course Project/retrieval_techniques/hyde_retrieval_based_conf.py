import pandas as pd
import json
from utils.model_factory import get_llm_model
from retrieval_techniques.lexical_retrieval_based_conf import generate_answers_for_queries
from langchain.messages import SystemMessage, HumanMessage
from sentence_transformers import SentenceTransformer, util

MODEL_NAME = "all-MiniLM-L6-v2"
# creating sentence transformer model
sentence_transformer_model = SentenceTransformer(MODEL_NAME)
# function to get 3 passages most relevant to query
# with HyDE approach (hypothetical answer generation)
def get_most_relevant_three_passages_with_hyde(row, model_name):
    passages = row.loc["passages"]
    passages = json.loads(passages)
    question = row.loc["query"]
    
    try:
        model = get_llm_model(model_name, temperature=0.3)
        
        # defining the prompt to genrate and hypothetical answer
        # for a given query using the LLM's general knowledge (zero-shot)
        system_msg = SystemMessage(f"""Please write a short, 
                concise passage to answer the question.
                Do not include any explanations, just the answer itself.
                Also beware that questions are from MS MARCO question  
                answering dataset which was initially released at 2016.
                Especially pay attention to year information because some
                information may change over time. 
                You must answer based on dataset's release date to for correctness.
                """)

        human_msg = HumanMessage(f"Question: {question}\n\nAnswer:")

        messages = [system_msg, human_msg]

        # invoking model with messages
        response = model.invoke(messages)
        hyde_answer = response.content.strip()
        print(f"HyDE answer is: answer='{hyde_answer}'")
    
    except Exception as e:
        print(f"Error generating answer: {e}")
        return ""
    
    # extracting passage texts
    passage_texts = [passage["passage_text"] for passage in passages]
    
    # generating hyde answer and passage embeddings
    hyde_embedding = sentence_transformer_model.encode(hyde_answer, convert_to_tensor=True, show_progress_bar=True)
    passage_embeddings = sentence_transformer_model.encode(passage_texts, convert_to_tensor=True, show_progress_bar=True) 
    
    # calculating cosine similarity scores
    cosine_sim_scores = util.cos_sim(hyde_embedding, passage_embeddings)[0]
    scored_passages = []
    for index, score in enumerate(cosine_sim_scores):
        scored_passages.append({
            "passage": passage_texts[index],
            "score": score.item()
        })
    
    # sorting passages descending by score
    scored_passages.sort(key= lambda x: x["score"], reverse=True)
    most_relevant_three_passages = [passage["passage"] for passage in scored_passages[:3]]
    
    return json.dumps(most_relevant_three_passages)

# function run HyDE (hypothetical embedding generation-based retrieval) configuration
def run_retrieval(model_name):
    dataset_df = pd.read_csv("data\\ms_marco_qna_dataset.csv")
    print(dataset_df.head())
    # retrieving most relevant 3 passages for each query and saving to dataframe
    dataset_df["most_relevant_three_passages"] = dataset_df.apply(lambda row: get_most_relevant_three_passages_with_hyde(row, model_name), axis=1)
    # generating answer for each query using most relevant 3 passages
    dataset_df["generated_answer"] = dataset_df.apply(lambda row: generate_answers_for_queries(row, model_name), axis=1) 
    # save the most relevant three passages along with generated answers
    dataset_df.to_csv("outputs\\ms_marco_qna_with_generated_answers_hyde.csv", sep=",", header=True, index=False)
    