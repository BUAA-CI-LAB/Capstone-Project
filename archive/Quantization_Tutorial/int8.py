import os 

from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor import oneshot

MODEL_DIR = os.environ.get("MODEL_DIR", "qwen")
MODEL_DIR = MODEL_DIR or "Qwen"

MODEL_PATH = f"{MODEL_DIR}/Qwen3-0.6B"
SAVE_PATH = f"{MODEL_DIR}/Qwen3-0.6B-W8A8"

recipe = [
    SmoothQuantModifier(smoothing_strength=0.5),
    QuantizationModifier(
        scheme="W8A8", 
        targets="Linear", 
        ignore=["lm_head"]
    ),
]

oneshot(
    model=MODEL_PATH,
    recipe=recipe,
    dataset="open_platypus",
    output_dir=SAVE_PATH,
    max_seq_length=512,
    num_calibration_samples=128,
)
