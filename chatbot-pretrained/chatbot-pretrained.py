import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW
from datasets import load_dataset

def load_model_and_tokenizer(model_name="gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

def get_dialog_dataset(max_samples=1000):
    dataset = load_dataset("OpenAssistant/oasst1", split="train")
    pairs = []
    count = 0
    for item in dataset:
        if item["role"] == "user" and item.get("reply") and item["reply"]:
            for reply in item["reply"]:
                if reply["role"] == "assistant":
                    pairs.append(f"You: {item['text']}\nBot: {reply['text']}")
                    count += 1
                    if count >= max_samples:
                        return pairs
    return pairs

def finetune(model, tokenizer, device, data, epochs=1, lr=5e-5, max_length=64):
    model.train()
    optim = AdamW(model.parameters(), lr=lr)
    for _ in range(epochs):
        for text in data:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            input_ids = inputs["input_ids"].to(device)
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optim.step()
            optim.zero_grad()
    model.eval()

def generate_response(history, model, tokenizer, device, max_length=100):
    prompt = "\n".join(history) + "\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_length=inputs.input_ids.shape[1]+max_length, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].split("\n",1)[0].strip()

def main():
    print("Simple GPT-2 Chatbot. Type 'exit' to quit.")
    model, tokenizer, device = load_model_and_tokenizer()
    print("Finetuning...")
    data = get_dialog_dataset(1000)
    finetune(model, tokenizer, device, data)
    history = []
    while True:
        user = input("You: ")
        if user.strip().lower() == "exit": break
        history.append(f"You: {user}")
        bot = generate_response(history, model, tokenizer, device)
        print(f"Bot: {bot}")
        history.append(f"Bot: {bot}")

if __name__ == "__main__":
    main()
