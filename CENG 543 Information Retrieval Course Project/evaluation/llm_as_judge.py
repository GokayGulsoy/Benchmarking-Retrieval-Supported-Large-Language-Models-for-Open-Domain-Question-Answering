import pandas as pd
import json
import os 
import argparse
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

JUDGE_MODEL = "gpt-4o"       
# creating class to define output structure of judge LLM
class JudgeLLMEvaluationScore(BaseModel):
    correctness_score: int = Field(description="Score 1-5: Does the generated answer convey the same meaning as the ground truth answer?")
    faithfulness_score: int = Field(description="Score 1-5: Is the generated answer supported by the Retrieved Context?")
    context_quality_score: int = Field(description="Score 1-5: Does the Retrieved Context contain the same key information as the ground Truth Passages?")
    explanation: str = Field(description="Brief justification for the scores")


# function to create LLM that acts as a judge
def create_judge_llm():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not set. Skipping LLM answer generation.")
        return ""
    
     # initializing judge LLM agent
    judge_llm = ChatOpenAI(model=JUDGE_MODEL, api_key=api_key ,temperature=0.3)
    # defining structured output format
    parser = JsonOutputParser(pydantic_object=JudgeLLMEvaluationScore)
            
    # defining judge LLM's prompt
    judge_prompt ="""
    You are an expert evaluator for Retrieval-Augmented Generation (RAG) system.

    Your goal is to grade the system's performance based on multiple perspectives.

    --INPUT DATA--
    1. User Query: {query}
    2. Ground Truth Answer (Human Written): {ground_truth_answer}
    3. Ground Truth Context (Passages used by Human): {ground_truth_context}

    4. Generated Answer (AI written): {generated_answer}
    5. Retrieved Context (Passages found by AI): 
    {retrieved_context} 

    --EVALUATION CRITERIA--
    1. Correctness (1-5): Compare the Generated Answer to the Ground Truth Answer AND Ground Truth Context. Does the AI answer the question correctly? (Allow for phrasing differences).
    2. Faithfulness (1-5): Is the Generated Answer derived *only* from the Retrieved Context? (Penalize hallucinations not found in #5).
    3. Compare #5 (Retrieved Context) vs #3 (Ground Truth Context). Did the AI retrieve the necessary information to answer the question?
    """

    # adding formating instructions to judge prompt 
    # parser.get_format_instructions() may contain JSON-like braces which
    # ChatPromptTemplate will interpret as template variables. Escape braces
    # so the instructions are treated as literal text in the prompt.
    format_instructions = parser.get_format_instructions()
    print(format_instructions)
    format_instructions_escaped = format_instructions.replace("{", "{{").replace("}", "}}")
    judge_prompt = judge_prompt + "\n\n" + format_instructions_escaped
    judge_prompt = ChatPromptTemplate.from_template(judge_prompt)
    # creating judge chain
    judge_chain = judge_prompt | judge_llm | parser

    return judge_chain
    
# assigning scores for each generated answer with judge LLM
def run_judge_llm_to_generate_scores(row, judge):
    passages = row.loc["passages"]
    passages = json.loads(passages)
    
    # extracting passages where is_selected key is 1
    selected_passages = [p["passage_text"] for p in passages if int(p.get("is_selected")) == 1]    
    # getting most relevant 3 passages retrieved by configuration
    retrieved_passages = json.loads(row.loc["most_relevant_three_passages"])
    
    # creating contexts for selected and retrieved passages
    ground_truth_context = "\n- ".join(selected_passages)
    retrieved_context = "\n- ".join(retrieved_passages)
    
    # running judge to generate scores
    score_data = judge.invoke({
        "query": row.loc["query"],
        "ground_truth_answer": row.loc["answers"],
        "ground_truth_context": ground_truth_context,
        "generated_answer": row.loc["generated_answer"],
        "retrieved_context": retrieved_context 
    })
    
    # assigned scores are in dictionary format turning it into json dumped version 
    print(score_data)
    
    return json.dumps(score_data) 
    
# function to assing scores to each generated answer via LLM judge    
def assign_judge_llm_scores(retrieval_technique):        
    # creating LLM judge 
    judge = create_judge_llm()        
    
    # assigning score to each generated answer with baseline conf
    if retrieval_technique == "lexical":
        dataset_df = pd.read_csv("outputs\\ms_marco_qna_with_generated_answers_metrics_lexical.csv")
        dataset_df["llm_judge_scores"] = dataset_df.apply(lambda row: run_judge_llm_to_generate_scores(row, judge), axis=1)
        dataset_df.to_csv("outputs\\ms_marco_qna_with_generated_answers_metrics_judge_scores_lexical.csv", sep=",", header=True, index=False)
        
    # assigning score to each generated answer with second conf
    elif retrieval_technique == "semantic":
        dataset_df = pd.read_csv("outputs\\ms_marco_qna_with_generated_answers_metrics_semantic.csv")
        dataset_df["llm_judge_scores"] = dataset_df.apply(lambda row: run_judge_llm_to_generate_scores(row, judge), axis=1)
        dataset_df.to_csv("outputs\\ms_marco_qna_with_generated_answers_metrics_judge_scores_semantic.csv", sep=",", header=True, index=False)
        
    # assigning score to each generated answer with third conf
    elif retrieval_technique == "hyde":
        dataset_df = pd.read_csv("outputs\\ms_marco_qna_with_generated_answers_metrics_hyde.csv")     
        dataset_df["llm_judge_scores"] = dataset_df.apply(lambda row: run_judge_llm_to_generate_scores(row, judge), axis=1)
        dataset_df.to_csv("outputs\\ms_marco_qna_with_generated_answers_metrics_judge_scores_hyde.csv", sep=",", header=True, index=False)

    # assigning score to each generated answer with fourth conf
    elif retrieval_technique == "prf":
        dataset_df = pd.read_csv("outputs\\ms_marco_qna_with_generated_answers_metrics_prf.csv")     
        dataset_df["llm_judge_scores"] = dataset_df.apply(lambda row: run_judge_llm_to_generate_scores(row, judge), axis=1)
        dataset_df.to_csv("outputs\\ms_marco_qna_with_generated_answers_metrics_judge_scores_prf.csv", sep=",", header=True, index=False)
        
     # assigning score to each generated answer with fifth conf
    elif retrieval_technique == "rrf":
        dataset_df = pd.read_csv("outputs\\ms_marco_qna_with_generated_answers_metrics_rrf.csv")     
        dataset_df["llm_judge_scores"] = dataset_df.apply(lambda row: run_judge_llm_to_generate_scores(row, judge), axis=1)
        dataset_df.to_csv("outputs\\ms_marco_qna_with_generated_answers_metrics_judge_scores_rrf.csv", sep=",", header=True, index=False)
    else:
        print("RETRIEVAL TECHNIQUE IS NOT VALID: use python evaluation.llm_as_judge.py --retrieval-technique --help to see available tehcniques")

    
if __name__ == "__main__":
    cmd_parser = argparse.ArgumentParser(prog="LLM as Judge Metric Calculator" ,description="LLM-based metric calculator for correctness, faithfulness, context quality scores")
    # adding command line argument to CLI parser
    cmd_parser.add_argument("--retrieval-technique", type=str, default=None, required=True, help="name of the retrieval technique, provide one of the: lexical | semantic | hyde | prf | rrf")
    cmd_args = cmd_parser.parse_args()

    # run function to assign scores to each answer with judge LLM
    assign_judge_llm_scores(cmd_args.retrieval_technique)