import os 

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = os.environ.get("MODEL_DIR", "qwen")
MODEL_DIR = MODEL_DIR or "Qwen"

MODEL_PATH = f"{MODEL_DIR}/Qwen3-0.6B"
SAVE_PATH = f"{MODEL_DIR}/Qwen3-0.6B-FP8-Dynamic"

print(f"Loading model from {MODEL_PATH}, will save to {SAVE_PATH}")

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])

oneshot(
    model=MODEL_PATH,
    recipe=recipe,
    output_dir=SAVE_PATH,
)
