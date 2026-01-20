import evaluate
import bert_score
import pandas as pd
import json
import argparse
import ast

# loading BLEU and ROUGE metrics
print(f"Loading BLEU and ROUGE metrics...this may take a moment.")
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")    
bert_scorer = bert_score.BERTScorer(lang="en", model_type="roberta-large")
# calculating BLEU, ROUGE, Bert scores for each
# (ground_truth_answers, generated_answer) 
def calculate_bleu_rouge_bertscore_metrics(row):
    # getting reference answers
    answers = row.loc["answers"]
    references = ast.literal_eval(answers)
    print(type(references))
    references = [references]
    # getting generated answer (prediction)
    generated_answer = row.loc["generated_answer"]
    generated_answer = [generated_answer]
    
    # BLEU expects plain texts as inputs (not tokenized texts)
    bleu_results = bleu_metric.compute(predictions=generated_answer, references=references)
    print(f"BLEU Score: {bleu_results["bleu"]:.2f}")
    
    # ROUGE expects plain texts as inputs (not tokenized texts)    
    rouge_results = rouge_metric.compute(predictions=generated_answer, references=references)
    print(f"ROUGE-1 F1 Score: {rouge_results["rouge1"]:2f}")
    print(f"ROUGE-L F1 Score: {rouge_results["rougeL"]:2f}")

    # calculating Bert score
    precision, recall, f1 = bert_scorer.score(generated_answer, references)
    print(f"BERT Precision: {precision.item():.4f}")
    print(f"BERT Recall: {recall.item():.4f}")
    print(f"BERT F1: {f1.item():.4f}")
    
    metrics = [bleu_results["bleu"], rouge_results["rouge1"], rouge_results["rougeL"], precision.item(), recall.item(), f1.item()]

    return json.dumps(metrics)

# function to calculate BLEU, ROUGE, and BertScore metrics
# for all the five configurations
def compute_metrics_for_retrieval_conf(retrieval_technique):
    # calculating BLEU, ROUGE, and Bert scores for base configuration  
    if retrieval_technique == "lexical":
        dataset_df = pd.read_csv("outputs\\ms_marco_qna_with_generated_answers_lexical.csv")
        dataset_df["bleu_rouge1_rougeL_bert_scores"] = dataset_df.apply(calculate_bleu_rouge_bertscore_metrics, axis=1)
        dataset_df.to_csv("outputs\\ms_marco_qna_with_generated_answers_metrics_lexical.csv", sep=",", header=True, index=False) 
     
    # calculating BLEU, ROUGE, and Bert scores for second configuration
    elif retrieval_technique == "semantic":
        dataset_df = pd.read_csv("outputs\\ms_marco_qna_with_generated_answers_semantic.csv")
        dataset_df["bleu_rouge1_rougeL_bert_scores"] = dataset_df.apply(calculate_bleu_rouge_bertscore_metrics, axis=1)
        dataset_df.to_csv("outputs\\ms_marco_qna_with_generated_answers_metrics_semantic.csv", sep=",", header=True, index=False) 

    # calculating BLEU, ROUGE, and Bert scores for third configuration
    elif retrieval_technique == "hyde":
        dataset_df = pd.read_csv("outputs\\ms_marco_qna_with_generated_answers_hyde.csv")
        dataset_df["bleu_rouge1_rougeL_bert_scores"] = dataset_df.apply(calculate_bleu_rouge_bertscore_metrics, axis=1)
        dataset_df.to_csv("outputs\\ms_marco_qna_with_generated_answers_metrics_hyde.csv", sep=",", header=True, index=False) 

    # calculating BLEU, ROUGE, and Bert scores for fourth configuration
    elif retrieval_technique == "prf":
        dataset_df = pd.read_csv("outputs\\ms_marco_qna_with_generated_answers_prf.csv")
        dataset_df["bleu_rouge1_rougeL_bert_scores"] = dataset_df.apply(calculate_bleu_rouge_bertscore_metrics, axis=1)
        dataset_df.to_csv("outputs\\ms_marco_qna_with_generated_answers_metrics_prf.csv", sep=",", header=True, index=False)
        
    # calculating BLEU, ROUGE, and Bert scores for fifth configuration
    elif retrieval_technique == "rrf":
        dataset_df = pd.read_csv("outputs\\ms_marco_qna_with_generated_answers_rrf.csv")
        dataset_df["bleu_rouge1_rougeL_bert_scores"] = dataset_df.apply(calculate_bleu_rouge_bertscore_metrics, axis=1)
        dataset_df.to_csv("outputs\\ms_marco_qna_with_generated_answers_metrics_rrf.csv", sep=",", header=True, index=False) 
    else: 
        print("RETRIEVAL TECHNIQUE IS NOT VALID: use python evaluation.evaluate_confs.py --retrieval-technique --help to see available tehcniques")

if __name__ == "__main__":
    cmd_parser = argparse.ArgumentParser(prog="Metric Calculator" ,description="Metric Calculator for BLEU, ROUGE, and Bert Scores")
    # adding command line argument to CLI parser
    cmd_parser.add_argument("--retrieval-technique", type=str, default=None, required=True, help="name of the retrieval technique, provide one of the: lexical | semantic | hyde | prf | rrf")
    cmd_args = cmd_parser.parse_args()

    # run to calculate BLEU, ROUGE, and Bert scores for each configuration
    compute_metrics_for_retrieval_conf(cmd_args.retrieval_technique)    
    