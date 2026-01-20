import pandas as pd
import argparse
import json

# function that calculates the average of 
# metric related columns
def calculate_avg_metrics(dataset_df):
    total_bleu = 0
    total_rouge1 = 0
    total_rougel = 0
    total_bert_precision = 0
    total_bert_recall = 0
    total_bert_f1 = 0
    total_correctness_score = 0
    total_faithfulness_score = 0
    total_context_quality_score = 0
    for index, row in dataset_df.iterrows():
        # list that contains BLEU, ROUGE-1, ROUGE-L, BertScore precision, recall, f1
        score_list = json.loads(row["bleu_rouge1_rougeL_bert_scores"])
        total_bleu += score_list[0]
        total_rouge1 += score_list[1]
        total_rougel += score_list[2]
        total_bert_precision += score_list[3]
        total_bert_recall += score_list[4] 
        total_bert_f1 += score_list[5]
        # dictionary that contains correctness, faithfulness, and context quality scores
        judge_llm_score_dict = json.loads(row["llm_judge_scores"])
        total_correctness_score += judge_llm_score_dict["correctness_score"]
        total_faithfulness_score += judge_llm_score_dict["faithfulness_score"]
        total_context_quality_score += judge_llm_score_dict["context_quality_score"]
    
    # computing average metrics
    num_of_rows = len(dataset_df)
    avg_bleu = total_bleu / num_of_rows
    avg_rouge1 = total_rouge1 / num_of_rows
    avg_rougel = total_rougel / num_of_rows
    avg_bert_precision = total_bert_precision / num_of_rows 
    avg_bert_recall = total_bert_recall / num_of_rows
    avg_bert_f1 = total_bert_f1 / num_of_rows
    avg_correctness_score = total_correctness_score / num_of_rows
    avg_faithfulness_score = total_faithfulness_score / num_of_rows
    avg_context_quality_score = total_context_quality_score / num_of_rows
    
    return [avg_bleu, avg_rouge1, avg_rougel, avg_bert_precision, avg_bert_recall, avg_bert_f1, avg_correctness_score, avg_faithfulness_score, avg_context_quality_score]
    
  
# function to display results of metrics for specific configuration
def print_averaged_metrics(result_metric_list, conf_name):
    # extracting configuration name for given configuration
    last_under_score_index = conf_name.rfind("_")
    file_type_seperator_index = conf_name.rfind(".")
    conf_name = conf_name[last_under_score_index+1:file_type_seperator_index]
    
    print(f"AVERAGE METRICS FOR {conf_name} CONFIGURATION\n" + "="*62)
    print(f"Average BLEU Score: \t{result_metric_list[0]}")
    print(f"Average ROUGE-1 Score: \t{result_metric_list[1]}")
    print(f"Average ROUGE-L Score: \t{result_metric_list[2]}")
    print(f"Average BERT Precision Score: \t{result_metric_list[3]}")
    print(f"Average BERT Recall Score: \t{result_metric_list[4]}")
    print(f"Average BERT F1 Score: \t{result_metric_list[5]}")
    print(f"Average LLM-Judge Correctness Score Score: \t{result_metric_list[6]}")
    print(f"Average LLM-Judge Faithfulness Score Score Score: \t{result_metric_list[7]}")
    print(f"Average LLM-Judge Context-Quality Score: \t{result_metric_list[8]}")
    print("="*62)
    
    
# function to display average metrics for each configuration
def display_avg_metrics(retrieval_technique):
    if retrieval_technique == "lexical":
        # calculating average metrics for lexical conf
        dataset_df = pd.read_csv("outputs\\ms_marco_qna_with_generated_answers_metrics_judge_scores_lexical.csv")
        result_baseline = calculate_avg_metrics(dataset_df)
        # displaying averaged metrics for lexical configuration
        conf_name = "ms_marco_qna_with_generated_answers_metrics_judge_scores_lexical.csv"
        print_averaged_metrics(result_baseline,conf_name)
    elif retrieval_technique == "semantic":
        # calculating average metrics for semantic conf
        dataset_df = pd.read_csv("outputs\\ms_marco_qna_with_generated_answers_metrics_judge_scores_semantic.csv")
        result_second = calculate_avg_metrics(dataset_df)
        # displaying averaged metrics for semantic configuration
        conf_name = "ms_marco_qna_with_generated_answers_metrics_judge_scores_semantic.csv"
        print_averaged_metrics(result_second,conf_name)
    elif retrieval_technique == "hyde":
        # calculating average metrics for hyde conf
        dataset_df = pd.read_csv("outputs\\ms_marco_qna_with_generated_answers_metrics_judge_scores_hyde.csv")     
        result_third = calculate_avg_metrics(dataset_df)
        # displaying averaged metrics for hyde configuration
        conf_name = "ms_marco_qna_with_generated_answers_metrics_judge_scores_hyde.csv"
        print_averaged_metrics(result_third,conf_name)
    elif retrieval_technique == "prf":     
        # calculating average metrics for pseuodo-relevance feedback conf
        dataset_df = pd.read_csv("outputs\\ms_marco_qna_with_generated_answers_metrics_judge_scores_prf.csv")     
        result_fourth = calculate_avg_metrics(dataset_df) 
        # displaying averaged metrics for pseuodo-relevance feedback configuration
        conf_name = "ms_marco_qna_with_generated_answers_metrics_judge_scores_prf.csv"
        print_averaged_metrics(result_fourth,conf_name)
    elif retrieval_technique == "rrf":
        # calculating average metrics for reciprocal rag-fuison conf
        dataset_df = pd.read_csv("outputs\\ms_marco_qna_with_generated_answers_metrics_judge_scores_rrf.csv")     
        result_fifth = calculate_avg_metrics(dataset_df)   
        # displaying averaged metrics for reciprocal rag-fusion configuration
        conf_name = "ms_marco_qna_with_generated_answers_metrics_judge_scores_rrf.csv"
        print_averaged_metrics(result_fifth,conf_name)
    else:
        print("RETRIEVAL TECHNIQUE IS NOT VALID: use python evaluation.calculate_avg_metrics.py --retrieval-technique --help to see available tehcniques")       

if __name__ == "__main__":
    cmd_parser = argparse.ArgumentParser(prog="Average Metrics Calculator" ,description="Average Metric Calculator for BLEU, ROUGE, BERT, and LLM-as Judge Scores")
    # adding command line argument to CLI parser
    cmd_parser.add_argument("--retrieval-technique", type=str, default=None, required=True, help="name of the retrieval technique, provide one of the: lexical | semantic | hyde | prf | rrf")
    cmd_args = cmd_parser.parse_args()
    
    # display averaged metrics given  retrieval configuration     
    display_avg_metrics(cmd_args.retrieval_technique)    