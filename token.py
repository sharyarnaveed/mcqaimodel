# token.py - Complete fixed version
from transformers import AutoTokenizer
from datasets import load_dataset  # Import added here

def tokenize_dataset():
    try:
        # Load model and tokenizer
        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Tokenizer loaded ✅")
        
        # Load dataset
        data_files = {"train": "biomcq_train.json", "test": "biomcq_test.json"}
        dataset = load_dataset("json", data_files=data_files)
        print("Dataset loaded ✅")
        
        def format_for_generation(example):
            # Extract subject or use default
            subject = "biology"  # default
            if 'text' in example:
                text_parts = example['text'].split('Subject: ')
                if len(text_parts) > 1:
                    subject = text_parts[-1].strip()
            
            prompt = f"Generate a multiple choice question about {subject}:"
            completion = example['text']
            return {"input_text": prompt, "target_text": completion}
        
        # Format dataset
        dataset = dataset.map(format_for_generation)
        print("Dataset formatted for generation ✅")
        print("Sample input:", dataset["train"][0]["input_text"])
        print("Sample target:", dataset["train"][0]["target_text"][:100] + "...")
        
        # Tokenization parameters
        max_input_length = 128
        max_output_length = 256
        
        def tokenize_function(batch):
            # Tokenize inputs
            inputs = tokenizer(
                batch["input_text"],
                max_length=max_input_length,
                truncation=True,
                padding="max_length"
            )
            
            # Tokenize targets with special handling
            with tokenizer.as_target_tokenizer():
                targets = tokenizer(
                    batch["target_text"],
                    max_length=max_output_length,
                    truncation=True,
                    padding="max_length"
                )
            
            # Replace padding tokens with -100 in labels
            labels = targets["input_ids"]
            labels = [[(label if label != tokenizer.pad_token_id else -100) for label in labels_seq] 
                     for labels_seq in labels]
            
            inputs["labels"] = labels
            return inputs
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        print("Dataset tokenized ✅")
        
        # Verify tokenization
        print("Sample tokenized input:", tokenized_dataset["train"][0]["input_ids"][:10])
        print("Sample labels (non-padded):", [x for x in tokenized_dataset["train"][0]["labels"] if x != -100][:10])
        
        # Save tokenized dataset
        tokenized_dataset.save_to_disk("tokenized_biomcq_dataset")
        print("Tokenized dataset saved ✅")
        
        return tokenized_dataset
        
    except Exception as e:
        print(f"Error in tokenize_dataset: {e}")
        import traceback
        traceback.print_exc()
        return None

# Run tokenization
tokenized_dataset = tokenize_dataset()