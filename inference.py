import torch
from model.model import ModelConfig, MedSLM
from tokenizers import Tokenizer

# load tokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE

tokenizer = Tokenizer(BPE.from_file(
    "tokenizer/vocab.json",
    "tokenizer/merges.txt"
))

# load model
device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = ModelConfig()
model = MedSLM(cfg).to(device)

ckpt = torch.load("checkpoints/medslm_final.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

print(f"Model loaded — {model.count_parameters():,} parameters")

# inference function
@torch.no_grad()
def generate(prompt: str, max_new_tokens: int = 100, temperature: float = 0.8):
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long).to(device)

    for _ in range(max_new_tokens):
        if input_ids.shape[1] >= cfg.context_length:
            break
        logits = model(input_ids)
        next_token_logits = logits[0, -1, :] / temperature
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    output_ids = input_ids[0].tolist()
    text = tokenizer.decode(output_ids)
    return text.replace("Ġ", " ").replace("Ċ", "\n")

# test prompts
prompts = [
    "The patient presented with symptoms of",
    "Diabetes mellitus is characterized by",
    "The study investigated the effects of",
]

for prompt in prompts:
    print(f"\nPrompt: {prompt}")
    print(f"Output: {generate(prompt)}")
    print("-" * 60)