import sys

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = sys.argv[1]

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype="auto",
    device_map="cuda:0",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

messages = [{
    "role": "user",
    "content": "What is your name?"
}]
conversation = tokenizer.apply_chat_template(
    messages,
    tokenize=False
)
inputs = tokenizer(conversation, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=50)[0]
response = tokenizer.decode(outputs[inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print(repr(response))
