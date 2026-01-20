import pandas as pd
import json
from utils.model_factory import get_llm_model
from retrieval_techniques.lexical_retrieval_based_conf import generate_answers_for_queries
from retrieval_techniques.semantic_retrieval_based_conf import get_most_relevant_three_passages_with_sentence_transformer
from langchain.messages import SystemMessage, HumanMessage

MODEL_NAME = "all-MiniLM-L6-v2"
# function to answer the question using pseudo-relevance feedback approach
def generate_question_with_prf(row, model_name):
    passages = row.loc["passages"]
    passages = json.loads(passages)
    question = row.loc["query"]    
    
    most_relevant_three_passages = row.loc["most_relevant_three_passages"]
    most_relevant_three_passages = json.loads(most_relevant_three_passages)
     # creating context from relevant passages
    context = ""
    for passage in most_relevant_three_passages:
        context = context + passage + "\n--------\n"
    
    try:
        model = get_llm_model(model_name, temperature=0.3)
        
        system_msg = SystemMessage("You are an expert search optimizer. Analyze the context and produce a concise, improved search query that will improve retrieval accuracy.")
        human_msg = HumanMessage(f"User question: {question}\n\nInitial Search Results (Context):\n{context}\n\nTask: Construct a NEW, concise, and more effective search query that incorporates missing terms to improve retrieval accuracy. Do NOT explain your resoning just give New query.\n\nNew Query:")

        messages = [system_msg, human_msg]

        # invoking model with messages
        response = model.invoke(messages)
        generated_query = response.content.strip()
        print(f"NEW query is: query='{generated_query}'")
    
    except Exception as e:
        print(f"Error generating answer: {e}")
        return ""
    
    
    return json.dumps(generated_query)


# function to run pseudo-relevance feedback-based retrieval configuration
def run_retrieval(model_name):
    dataset_df = pd.read_csv("data\\ms_marco_qna_dataset.csv")
    print(dataset_df.head())
    # getting most relevant three passages from second configuration
    dataset_df_with_top_three_passages = pd.read_csv("outputs\\ms_marco_qna_with_generated_answers_semantic.csv")
    dataset_df["most_relevant_three_passages"] = dataset_df_with_top_three_passages["most_relevant_three_passages"]
    # generating new query with pseudo-relevance feedback approach
    dataset_df["prf_query"] = dataset_df.apply(lambda row: generate_question_with_prf(row, model_name), axis=1)
    # getting most relevant three passages for the generated query
    dataset_df["most_relevant_three_passages"] = dataset_df.apply(lambda row: get_most_relevant_three_passages_with_sentence_transformer(row, isPRF=True), axis=1)
    # generating answer for each query using most relevant 3 passages and generated query
    dataset_df["generated_answer"] = dataset_df.apply(lambda row: generate_answers_for_queries(row, model_name, isPRF=True), axis=1) 
    # save the most relevant three passages along with generated answers
    dataset_df.to_csv("outputs\\ms_marco_qna_with_generated_answers_prf.csv", sep=",", header=True, index=False)
