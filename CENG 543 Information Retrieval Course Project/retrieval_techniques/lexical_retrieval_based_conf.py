import pandas as pd
import json
import re
import nltk
from utils.model_factory import get_llm_model
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langchain.messages import SystemMessage, HumanMessage
from rank_bm25 import BM25Okapi

# downloading stop words and punctuation symbols from nltk library
nltk.download("stopwords")
nltk.download("punkt")

# function to preprocess passage texts
def preprocess_passage_texts(row):
    stop_words = set(stopwords.words("english"))
        
    passages = row.loc["passages"]
    passages = json.loads(passages)
    
    for passage in passages:
        passage_text = passage["passage_text"]
        # converting passage text into lowercase
        lowered_passage_text = passage_text.lower()
        # removing punctuation marks
        lowered_passage_text = re.sub(r'[^\w\s]', '', lowered_passage_text)
        # removing stopwords from passage text
        tokens = word_tokenize(lowered_passage_text)
        filtered_tokens = [word for word in tokens if word not in stop_words]
        # add filtered tokens as a new entry in passage dictionary   
        passage["tokens"] = filtered_tokens

    row["passages"] = json.dumps(passages)
    return row


# function to get 3 passages most relevant to query
def get_most_relevant_three_passages_with_bm25(row):
    corpus = []
    tokenized_corpus = []
    passages = row.loc["passages"]
    passages = json.loads(passages)
    
    for passage in passages:
        # retrieve list of tokens for passage
        tokenized_passage = passage.get("tokens", [])
        # add tokens for given passage to corpus
        tokenized_corpus.append(tokenized_passage)
        # add passage text to corpus
        passage_text = passage["passage_text"]
        corpus.append(passage_text)
            
    # creating ranker bm25 ranker object
    bm25 = BM25Okapi(tokenized_corpus)  
    # preprocessing query
    query = row.loc["query"]
    lowered_query = query.lower()
    # removing punctuation marks
    lowered_query = re.sub(r'[^\w\s]', '', lowered_query)
    # removing stopwords from passage text
    tokens = word_tokenize(lowered_query)
    stop_words = set(stopwords.words("english"))
    tokenized_query = [word for word in tokens if word not in stop_words] 
    # retrieving most relevant 3 passages
    most_relevant_three_passages = bm25.get_top_n(tokenized_query, corpus, n=3)
    
    return json.dumps(most_relevant_three_passages)

# function to generate an answer for each query 
# based on the most relevant three (or in few cases 2) passages
def generate_answers_for_queries(row, model_name, isPRF=False):
    most_relevant_three_passages = row.loc["most_relevant_three_passages"] 
    most_relevant_three_passages = json.loads(most_relevant_three_passages)
    
    question = ""
    if isPRF:
        question = row.loc["prf_query"]
    else:
        question = row.loc["query"]
    
    # creating context from relevant passages
    context = ""
    for passage in most_relevant_three_passages:
        context = context + passage + "\n--------\n"

    try:
        # creating agent
        model = get_llm_model(model_name, temperature=0.3)
        
        # defining system and user prompts with context embedded
        system_message = SystemMessage("""You are a question answering expert for the MS MARCO dataset.
        Generate answers based SOLELY on the context provided below.
        Be concise and direct. Do not ask clarifying questions.
        Answer in a single sentence or brief format.""")
        
        # including context in the user message so that the model can use it
        user_prompt = f"""
        Context:
        {context}

        Question: {question}

        Answer based only on the context above:"""
        
        human_message = HumanMessage(user_prompt)
        messages = [system_message, human_message]
        
        # invoking model with messages
        response = model.invoke(messages)
        generated_answer = response.content.strip()
        print(f"Generated answer is: answer='{generated_answer}'")

        return generated_answer
    
    except Exception as e:
        print(f"Error generating answer: {e}")
        return ""

# function to run lexical retrieval configuration
def run_retrieval(model_name):
    dataset_df = pd.read_csv("data\\ms_marco_qna_dataset.csv")
    print(dataset_df.head())
    # applying preprocessing function to convert passage texts to tokens
    dataset_df = dataset_df.apply(preprocess_passage_texts, axis=1)
    # retrieving most relevant 3 passages for each query and saving to dataframe
    dataset_df["most_relevant_three_passages"] = dataset_df.apply(get_most_relevant_three_passages_with_bm25, axis=1)
    # generating answer for each query using most relevant 3 passages
    dataset_df["generated_answer"] = dataset_df.apply(lambda row: generate_answers_for_queries(row, model_name), axis=1) 
    # save the most relevant three passages along with generated answers 
    dataset_df.to_csv("outputs\\ms_marco_qna_with_generated_answers_lexical.csv", sep=",", header=True, index=False)
    