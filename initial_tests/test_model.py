import os, time, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"
MODEL_DIR = "./models/t5_jargon_v1"

print(f"Loading model from {MODEL_DIR}...")
t0 = time.perf_counter()
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR, dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model.eval()
print(f"Loaded in {time.perf_counter()-t0:.2f}s")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Using device: {device}")

# Warmup GPU
with torch.inference_mode():
    w = tokenizer("warmup", return_tensors="pt")
    w = {k: v.to(device) for k, v in w.items()}
    model.generate(**w, max_new_tokens=8, num_beams=1)

prefix = "Translate to corporate jargon: "
test_examples = [
    "I'm going to get coffee.",
    "Let's talk about this in a meeting.",
    "The project is delayed.",
    "We need to reduce costs.",
]

prompts = [prefix + s for s in test_examples]
batch = tokenizer(prompts, return_tensors="pt", truncation=True, padding=True)
batch = {k: v.to(device) for k, v in batch.items()}

with torch.inference_mode():
    outputs = model.generate(
        **batch,
        max_new_tokens=128,
        min_length=10,
        num_beams=4,
        do_sample=False,
        early_stopping=True,
        no_repeat_ngram_size=2,
    )

decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

print("\n" + "="*50)
for inp, out in zip(test_examples, decoded):
    print(f"\nInput : '{inp}'")
    print(f"Jargon: '{out}'")
print("\n" + "="*50)
