import pandas as pd
import json 

def parse_json_dataset_to_csv():
    # creating a dataframe to represent the dataset
    dataset_df = pd.DataFrame(columns=["passages", "query_id", "answers", "query_type", "query"])
    # openning JSON file
    with open("data\\dev_v1.1.json", encoding="utf_8") as dataset_json_file:
        for line_num, line in enumerate(dataset_json_file):
            # take 400 queries as a subset            
            if line_num == 410:
                break
                
            line = line.strip()    
            query_data = json.loads(line)
            
            # queries that does not have any answers have the answer field as: "answer": []
            # excluding queries that does not have any answer
            if query_data["answers"]:
                passages = query_data["passages"]
                query_id = query_data["query_id"]
                answers = query_data["answers"]
                query_type = query_data["query_type"]
                query = query_data["query"]
                
                new_row = {"passages": passages, "query_id": query_id, "answers": answers, "query_type": query_type, "query": query}
                new_df = pd.DataFrame([new_row]) 
                dataset_df = pd.concat([dataset_df, new_df])
    
    print(dataset_df.head())
    dataset_df["passages"] = dataset_df["passages"].apply(json.dumps)
    dataset_df.to_csv("data\\ms_marco_qna_dataset.csv", sep=",", header=True, index=False, encoding="utf-8")
    
    return dataset_df   


if __name__ == "__main__":
    dataset_df = parse_json_dataset_to_csv()
    # displaying example record partially
    first_record = dataset_df.iloc[:1]
    print(f"query id of first record is: {first_record["query_id"].iat[0]}")
    print(f"query type of first record is: {first_record["query_type"].iat[0]}")
    print(f"query id of first record is: {first_record["query"].iat[0]}")
    print(f"query id of first record is: {json.loads(first_record["passages"].iat[0])[0]["passage_text"]}")