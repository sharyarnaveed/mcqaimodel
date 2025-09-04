# structure_ft.py - Prepare textbook content for fine-tuning with input_text + target_text
import json
import os
import re
from datasets import Dataset

def clean_content(content):
    """Clean content by removing page markers and other unwanted text"""
    if not content:
        return ""
    
    # Remove page markers like "===== Page 24 ====="
    content = re.sub(r'=+ Page \d+ =+', '', content)
    
    # Remove any remaining sequences of = characters
    content = re.sub(r'=+', '', content)
    
    # Clean up extra whitespace
    content = re.sub(r'\s+', ' ', content).strip()
    
    return content

def format_for_finetune(item):
    """Format a content item into input_text/target_text structure"""
    chapter = item.get('chapter', 'Unknown Chapter')
    chunk_id = item.get('chunk_id', '')
    content = item.get('content', '')
    subject = item.get('subject', 'biology')
    
    input_text = f"Subject: {subject}\nChapter: {chapter}\nChunk ID: {chunk_id}\nContent: {content}"
    
    # target_text is empty for now (to be filled with MCQs later)
    target_text = "Q1. ...\nA. ...\nB. ...\nC. ...\nD. ...\nAnswer: ..."
    
    return {"input_text": input_text, "target_text": target_text}

def combine_json_files():
    try:
        # List of JSON file paths
        json_files = [
            "/kaggle/working/bioenergetics.json",
            "/kaggle/working/biomolecular.json",
            "/kaggle/working/biomolecular2.json",
            "/kaggle/working/cell structure.json",
            "/kaggle/working/circulation.json",
            "/kaggle/working/digestion.json",
            "/kaggle/working/enzymes.json",
            "/kaggle/working/enzymes2.json",
            "/kaggle/working/immunity.json",
            "/kaggle/working/viruses.json",
            "/kaggle/working/viruses2.json"
        ]
        
        all_data = []
        formatted_data = []
        
        for file_path in json_files:
            if os.path.exists(file_path):
                print(f"Processing {file_path}...")
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    for item in data:
                        if "subject" not in item:
                            item["subject"] = "biology"
                        if "chapter" not in item:
                            item["chapter"] = "Unknown Chapter"
                        if "chunk_id" not in item:
                            item["chunk_id"] = ""
                        if "content" in item:
                            item["content"] = clean_content(item["content"])
                        else:
                            item["content"] = ""
                        
                        formatted = format_for_finetune(item)
                        formatted_data.append(formatted)
                        all_data.append(item)
                    
                    print(f"Added {len(data)} items from {file_path}")
                    
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
            else:
                print(f"File not found: {file_path}")
        
        print(f"Total items combined: {len(all_data)}")
        if not all_data:
            print("No data found. Check your input files.")
            return None
        
        # Create Hugging Face Dataset from formatted data
        dataset = Dataset.from_list(formatted_data)
        print("Formatted dataset created successfully")
        print("Sample (input_text + target_text):")
        print("-" * 50)
        print(json.dumps(dataset[0], indent=2)[:400] + "...")
        print("-" * 50)
        
        # Split dataset for training/testing
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        print(f"Train size: {len(dataset['train'])}, Test size: {len(dataset['test'])}")
        
        # Save fine-tuning datasets
        dataset["train"].to_json("biology_ft_train.json")
        dataset["test"].to_json("biology_ft_test.json")
        
        # Save full raw data too
        with open("biology_content_full.json", "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        
        print("Datasets saved as:")
        print("- biology_ft_train.json (fine-tuning format for training)")
        print("- biology_ft_test.json (fine-tuning format for testing)")
        print("- biology_content_full.json (raw merged content)")
        
        return dataset
    
    except Exception as e:
        print(f"Error in combine_json_files: {e}")
        import traceback
        traceback.print_exc()
        return None

# Run the function
dataset = combine_json_files()
