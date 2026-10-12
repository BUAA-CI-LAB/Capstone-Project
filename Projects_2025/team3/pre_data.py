from datasets import load_dataset

ds = load_dataset("Iess/chinese_modern_poetry")
print(ds.keys())

print(ds['train'][0])

train = []
for data in ds['train']:
    texts = data['prompt']
    answer = data['response']
    train.append(
        {
            "instruction": texts,
            "input":"",
            "output":answer
        }
    )




import json

output_path = "../data/chinese_modern_poetry.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(train[:len(train)-100], f, ensure_ascii=False, indent=2)

print(f"Saved to {output_path}")


output_path = "../data/chinese_modern_poetry-test.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(train[len(train)-100:], f, ensure_ascii=False, indent=2)

print(f"Saved to {output_path}")