"""
Qwen3 Text Model Loader.
Centralized loading for Qwen3 language models (4B/8B/14B).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SUPPORTED_MODELS = {
    # Dense models (ordered by size)
    "qwen3-4b": "Qwen/Qwen3-4B",       # Budget-friendly, fast
    "qwen3-8b": "Qwen/Qwen3-8B",       # High quality, recommended
    "qwen3-14b": "Qwen/Qwen3-14B",     # Best quality
}


def load_qwen3_model(
    model_name: str = "qwen3-8b",
    quantization: str = "bfloat16",
    device: str = "cuda",
    cache_dir: Optional[str] = None
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load Qwen3 text model and tokenizer.

    Args:
        model_name: Model size to use (qwen3-4b, qwen3-8b, qwen3-14b)
        quantization: Quantization method ("4bit" for AWQ-INT4, "bfloat16" for full precision, default: "bfloat16")
        device: Device to run on ("cuda" or "cpu")
        cache_dir: Optional directory for model cache

    Returns:
        Tuple of (model, tokenizer)
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Choose from: {list(SUPPORTED_MODELS.keys())}"
        )

    model_id = SUPPORTED_MODELS[model_name]
    logger.info(f"Loading {model_name} ({quantization})...")

    # Configure model loading
    load_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir

    # Configure quantization (only 4bit or bfloat16)
    if quantization == "4bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        logger.info("  Using 4-bit quantization")
    elif quantization == "bfloat16":
        load_kwargs["dtype"] = torch.bfloat16
        logger.info("  Using BFloat16 full precision")
    else:
        raise ValueError(f"Unsupported quantization: {quantization}. Choose '4bit' or 'bfloat16'")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        cache_dir=cache_dir
    )

    # Set padding for batch generation
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()

    logger.info(f"✓ Qwen3 model loaded: {model_name}")

    # Print memory usage if CUDA
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"  GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    return model, tokenizer


def main():
    """Test Qwen3 model loader."""
    print("=" * 60)
    print("Qwen3 Model Loader Test")
    print("=" * 60)

    # Test loading smallest model
    print("\nLoading Qwen3-4B with 4-bit quantization...")
    model, tokenizer = load_qwen3_model("qwen3-4b", quantization="4bit")

    # Test inference
    print("\nTesting inference...")
    text = "Hello, how are you?"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input: {text}")
    print(f"Output: {response}")

    print("\n" + "=" * 60)
    print("✓ Qwen3 loader test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
