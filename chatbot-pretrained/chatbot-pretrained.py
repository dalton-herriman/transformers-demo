import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim.adamw import AdamW
from datasets import load_dataset

# --- Custom Dataset for User-Assistant pairs ---
class DialogDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=64):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        text = self.pairs[idx]
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        return enc.input_ids.squeeze(0), enc.attention_mask.squeeze(0)

# --- Load GPT-2 ---
def load_model_and_tokenizer(model_name="gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

# --- Extract oasst1 user-assistant pairs ---
def get_dialog_dataset(max_samples=2000):
    dataset = load_dataset("OpenAssistant/oasst1", split="train")
    messages = {msg["message_id"]: msg for msg in dataset}  # map id -> message

    pairs = []
    count = 0
    for msg in dataset:
        # Only take assistant messages that reply to a prompter (user)
        if msg["role"] == "assistant" and msg["parent_id"]:
            parent = messages.get(msg["parent_id"])
            if parent and parent["role"] == "prompter":
                pairs.append(f"You: {parent['text']}\nBot: {msg['text']}")
                count += 1
                if count >= max_samples:
                    break
    return pairs

# --- Finetuning ---
def finetune(model, tokenizer, device, data, epochs=1, batch_size=8, lr=5e-5, max_length=64, grad_accum=1):
    dataset = DialogDataset(data, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    model.config.loss_type = "ForCausalLMLoss"  # avoid warning
    optim = AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    for epoch in range(epochs):
        total_loss = 0
        for step, (input_ids, attention_mask) in enumerate(loader):
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            with torch.amp.autocast(device_type, enabled=scaler is not None):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss / grad_accum
            total_loss += loss.item()

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % grad_accum == 0:
                if scaler:
                    scaler.step(optim)
                    scaler.update()
                else:
                    optim.step()
                optim.zero_grad()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    model.eval()


# --- Response Generation ---
def generate_response(history, model, tokenizer, device, max_length=100):
    prompt = "\n".join(history) + "\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_length=inputs.input_ids.shape[1]+max_length,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].split("\n",1)[0].strip()

# --- Main ---
def main():
    print("Simple GPT-2 Chatbot. Type 'exit' to quit.")
    model, tokenizer, device = load_model_and_tokenizer()
    print("Loading dataset...")
    data = get_dialog_dataset(2000)
    data = get_dialog_dataset(50)
    print("Loaded pairs:", len(data))
    print("Example:", data[0] if data else "No data")   
    print("Finetuning...")
    finetune(model, tokenizer, device, data, epochs=2, batch_size=8, grad_accum=2)
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
