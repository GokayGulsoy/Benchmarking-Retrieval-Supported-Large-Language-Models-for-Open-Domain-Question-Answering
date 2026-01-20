import argparse
import sys

from retrieval_techniques import (
        lexical_retrieval_based_conf,
        semantic_retrieval_based_conf,
        hyde_retrieval_based_conf,
        pseudo_relevance_feedback_based_retrieval_conf,
        rag_fusion_based_retrieval_conf
)

# function to setup the command line parser
def setup_arg_parser():
    parser = argparse.ArgumentParser(description="Run specific IR retrieval techniques")

    # command line argument to select retrieval technique
    parser.add_argument(
        "--technique",
        type=str,
        required=True,
        choices=["lexical", "semantic", "hyde", "prf", "rrf"],
        help="The retrieval technique to use"
    )   

    # command line argument to select the LLM model to use with retrieval
    parser.add_argument(
        "--llm-model-id",
        type=str,
        default="gpt-3.5-turbo",
        help="LLM model to use with retrieval techniques supports: gpt | claude | gemini models"
    )

    return parser 

# function to run retrieval configuration based on provided 
# command line argument
def main():
    parser = setup_arg_parser()
    cmd_args = parser.parse_args()
    
    print(f"--- Starting Retrieval Process: {cmd_args.technique} | {cmd_args.llm_model_id}")
    try:
        if cmd_args.technique == "lexical":
            lexical_retrieval_based_conf.run_retrieval(cmd_args.llm_model_id)

        elif cmd_args.technique == "semantic":
            semantic_retrieval_based_conf.run_retrieval(cmd_args.llm_model_id)

        elif cmd_args.technique == "hyde":
            hyde_retrieval_based_conf.run_retrieval(cmd_args.llm_model_id)
        
        elif cmd_args.technique == "prf":
            pseudo_relevance_feedback_based_retrieval_conf.run_retrieval(cmd_args.llm_model_id)
        
        elif cmd_args.technique == "rrf":
            rag_fusion_based_retrieval_conf.run_retrieval(cmd_args.llm_model_id)

        else:
            raise ValueError(f"Invalid value for --technique flag provided use --help to see usage details")
        
    except Exception as e:
        print(f"Error during execution of retrieval {e}")
        sys.exit(1)    
        
# run retrieval pipeline according to given
# retrieval technique as a command line argument
if __name__ == "__main__":
    main()        