from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model
checkpoint = "result_stage2/checkpoint-1500"  # Change to your checkpoint
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Model loaded! Type 'quit' to exit.\n")

while True:
    user_input = input("User Ques: ")
    if user_input.lower() == 'quit':
        break

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_input}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the assistant's response
    print(f"Assistant: {response.split('assistant')[-1].strip()}\n")
