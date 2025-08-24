from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the fine-tuned model
model_path = "./mcq_model_fixed"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

def generate_mcq(prompt, max_len=128):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=max_len,
        do_sample=True,        # Enable sampling
        top_p=0.9,             # Nucleus sampling
        temperature=0.8,       # Add randomness
        num_return_sequences=1
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test the model
prompt = "Generate 10  biology MCQ."
print(generate_mcq(prompt))
