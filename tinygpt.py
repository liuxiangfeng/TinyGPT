import torch
import tiktoken
from model import TinyGPTModel
from utils import text_to_token_ids, token_ids_to_text

class TinyGPT:
    @staticmethod
    def build(model_conf, model_path: str, seed: int = 123):
        tokenizer = tiktoken.get_encoding("gpt2")
        
        # Load model
        model = TinyGPTModel(model_conf)
        model.load_state_dict(torch.load(model_path))
        model.to("cpu")
        model.eval()

        return TinyGPT(model, tokenizer)

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(self, input_prompt, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
        encoded = self.tokenizer.encode(input_prompt)
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)
        idx = text_to_token_ids(input_prompt, self.tokenizer)

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]

            with torch.no_grad():
                logits = self.model(idx_cond)
            logits = logits[:, -1, :]

            if top_k is not None: # Keep only top_k values
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

            if temperature > 0.0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)
                idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
            else: 
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

            if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
                break

            idx = torch.cat((idx, idx_next), dim=1)  # append sampled index to the running sequence (batch_size, num_tokens+1)

        decoded_text = token_ids_to_text(idx, self.tokenizer)
        return decoded_text


if __name__ == "__main__":
    model_path = "models/tinygpt-simple.pth"
    MODEL_CONFIG = {
        "vocab_size": 50257,    # Vocabulary size
        "context_length": 256,  # Shortened context length (orig: 1024)
        "emb_dim": 768,         # Embedding dimension
        "n_heads": 12,          # Number of attention heads
        "n_layers": 12,         # Number of layers
        "drop_rate": 0.1,       # Dropout rate
        "qkv_bias": False       # Query-key-value bias
    }
    tinygpt = TinyGPT.build(MODEL_CONFIG, model_path)

    prompt = "Hello, I am"
    out = tinygpt.generate(
        prompt,
        max_new_tokens=10,
        context_size=MODEL_CONFIG["context_length"],
        temperature=0.1
    )
    print(out)