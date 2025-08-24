# structure.py - Modified for Colab
import json
from datasets import Dataset

def prepare_dataset():
    try:
        # Load mcq.json
        with open("neerpearbiology_mcqs.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} MCQs")
        
        # Convert to HuggingFace dataset format
        train_data = []
        for item in data:
            # Ensure all required fields exist
            if "subject" not in item:
                item["subject"] = "biology"
            if "options" not in item:
                item["options"] = []
            if "question" not in item:
                item["question"] = ""
            if "answer" not in item:
                item["answer"] = ""   
            
            # Convert options list to string if it's a list
            if isinstance(item["options"], list):
                options_str = " | ".join(item["options"])
            else:
                options_str = str(item["options"])
            
            formatted = (
                f"Question: {item['question']}\n"
                f"Options: {options_str}\n"
                f"Answer: {item['answer']}\n"
                f"Subject: {item['subject']}"
            )
            train_data.append({"text": formatted})
        
        # Create Hugging Face Dataset
        dataset = Dataset.from_list(train_data)
        print("Dataset created successfully")
        print("Sample:", dataset[0])
        
        # Split dataset
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        print(f"Train size: {len(dataset['train'])}, Test size: {len(dataset['test'])}")
        
        # Save to JSON
        dataset["train"].to_json("biomcq_train.json")
        dataset["test"].to_json("biomcq_test.json")
        
        print("Dataset saved as biomcq_train.json and biomcq_test.json")
        return dataset
        
    except Exception as e:
        print(f"Error in prepare_dataset: {e}")
        return None

# Run the function
dataset = prepare_dataset()