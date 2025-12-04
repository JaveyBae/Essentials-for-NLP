"""
SigLIP2 Model Loader.
Centralized loading for SigLIP2 vision-language models.
Supports both HuggingFace models and PEFT LoRA adapters.
"""

import os
import json
import torch
from transformers import AutoModel, AutoProcessor
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SUPPORTED_MODELS = {
    # Base models (0.4B params - balanced speed/quality)
    "siglip2-base-patch16-224": "google/siglip2-base-patch16-224",          # Fastest, highly rated (77 likes)
    "siglip2-base-patch16-256": "google/siglip2-base-patch16-256",
    "siglip2-base-patch16-384": "google/siglip2-base-patch16-384",
    "siglip2-base-patch16-512": "google/siglip2-base-patch16-512",          # Best base model with high res

    # Large models (0.9B params - better quality)
    "siglip2-large-patch16-256": "google/siglip2-large-patch16-256",
    "siglip2-large-patch16-384": "google/siglip2-large-patch16-384",
    "siglip2-large-patch16-512": "google/siglip2-large-patch16-512",

    # So400m models (1B params - high quality, most popular series)
    "siglip2-so400m-patch14-224": "google/siglip2-so400m-patch14-224",
    "siglip2-so400m-patch14-384": "google/siglip2-so400m-patch14-384",      # MOST POPULAR (476k downloads, 62 likes)
    "siglip2-so400m-patch16-256": "google/siglip2-so400m-patch16-256",
    "siglip2-so400m-patch16-384": "google/siglip2-so400m-patch16-384",
    "siglip2-so400m-patch16-512": "google/siglip2-so400m-patch16-512",      # High resolution variant (144k downloads)

    # Giant optimized models (2B params - highest quality)
    "siglip2-giant-opt-patch16-256": "google/siglip2-giant-opt-patch16-256",
    "siglip2-giant-opt-patch16-384": "google/siglip2-giant-opt-patch16-384",  # Best giant model (158k downloads)
}


def load_siglip2_model(
    model_name: str = "siglip2-base-patch16-224",
    quantization: Optional[str] = None,
    device: str = "cuda",
    cache_dir: Optional[str] = None,
    use_flash_attention: bool = True,
    local_model_path: Optional[str] = None,
) -> Tuple[AutoModel, AutoProcessor]:
    """
    Load SigLIP2 model and processor.

    Args:
        model_name: Model variant to use (see SUPPORTED_MODELS)
        quantization: Must be None (quantization not supported for SigLIP2)
        device: Device to run on ("cuda" or "cpu")
        cache_dir: Optional directory for model cache
        use_flash_attention: Use Flash Attention 2 for memory efficiency (default: True)
        local_model_path: Path to fine-tuned model directory (overrides model_name)

    Returns:
        Tuple of (model, processor)
    """
    # Check if loading from local fine-tuned model
    is_peft_adapter = False
    base_model_id = None
    model_id = None

    if local_model_path is not None:
        if os.path.isdir(local_model_path):
            # Check if this is a PEFT adapter (has adapter_config.json)
            adapter_config_path = os.path.join(local_model_path, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                is_peft_adapter = True
                with open(adapter_config_path) as f:
                    adapter_config = json.load(f)
                base_model_id = adapter_config.get("base_model_name_or_path")
                model_id = base_model_id  # Use base model for config checks
                logger.info(f"Loading PEFT LoRA adapter from: {local_model_path}")
                logger.info(f"  Base model: {base_model_id}")
            else:
                # Merged model - load directly
                logger.info(f"Loading fine-tuned SigLIP2 from: {local_model_path}")
                model_id = local_model_path
        else:
            raise ValueError(f"Local model path not found: {local_model_path}")
    elif model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Choose from: {list(SUPPORTED_MODELS.keys())}"
        )
    else:
        model_id = SUPPORTED_MODELS[model_name]

    if quantization is not None:
        raise ValueError(
            f"Quantization not supported for SigLIP2 models. "
            f"BitsAndBytes causes dtype mismatches in attention layers. "
            f"Use F32 (base/large models) or float16 (so400m/giant-opt models) only."
        )

    # Configure model loading
    load_kwargs = {
        "device_map": "auto",
    }

    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir

    # Configure attention implementation
    # Base and Large models have issues with Flash Attention/SDPA - use eager attention (F32)
    # So400m and Giant-opt models work fine with optimized attention (float16)
    # For fine-tuned models, infer from the model_id path
    # Skip for PEFT adapters - we configure attention when loading base model
    if is_peft_adapter:
        pass  # Will configure when loading base model
    elif "base" in str(model_id) or "large" in str(model_id):
        load_kwargs["attn_implementation"] = "eager"
        # Use F32 (default) for base/large models as per model card specs
        logger.info(f"Loading SigLIP2: {model_name} (F32 with eager attention)")
    elif use_flash_attention and torch.cuda.is_available():
        try:
            import flash_attn
            load_kwargs["attn_implementation"] = "flash_attention_2"
            load_kwargs["dtype"] = torch.float16  # Use float16 with Flash Attention per official examples
            logger.info(f"Loading SigLIP2: {model_name} (float16 with Flash Attention 2)")
        except ImportError:
            logger.warning("  Flash Attention 2 not available, using SDPA with float16")
            load_kwargs["attn_implementation"] = "sdpa"
            load_kwargs["dtype"] = torch.float16  # Use float16 with SDPA per official examples
            logger.info(f"Loading SigLIP2: {model_name} (float16 with SDPA)")
    else:
        load_kwargs["attn_implementation"] = "sdpa"
        load_kwargs["dtype"] = torch.float16  # Use float16 with SDPA per official examples
        logger.info(f"Loading SigLIP2: {model_name} (float16 with SDPA)")

    # Load processor and model
    if is_peft_adapter:
        # Load base model first, then apply PEFT adapter
        from peft import PeftModel

        processor = AutoProcessor.from_pretrained(
            base_model_id,
            cache_dir=cache_dir
        )

        # For PEFT, determine load kwargs based on base model type
        if "base" in base_model_id or "large" in base_model_id:
            load_kwargs["attn_implementation"] = "eager"
            logger.info(f"Loading base model: {base_model_id} (F32 with eager attention)")
        else:
            load_kwargs["attn_implementation"] = "sdpa"
            load_kwargs["dtype"] = torch.float16
            logger.info(f"Loading base model: {base_model_id} (float16 with SDPA)")

        base_model = AutoModel.from_pretrained(base_model_id, **load_kwargs)
        model = PeftModel.from_pretrained(base_model, local_model_path)
        logger.info(f"✓ PEFT adapter loaded from: {local_model_path}")
    else:
        # Standard loading (HuggingFace model or merged model)
        processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=cache_dir
        )
        model = AutoModel.from_pretrained(model_id, **load_kwargs)

    model.eval()

    model_desc = local_model_path if local_model_path else model_name
    logger.info(f"✓ SigLIP2 model loaded: {model_desc}")

    # Print memory usage if CUDA
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"  GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    return model, processor


def main():
    """Test SigLIP2 model loader."""
    import requests
    from PIL import Image

    print("=" * 60)
    print("SigLIP2 Model Loader Test")
    print("=" * 60)

    # Test loading base model
    print("\nLoading SigLIP2 base model...")
    model, processor = load_siglip2_model("siglip2-base-patch16-224")

    # Test inference
    print("\nTesting zero-shot image classification...")

    # Load test image
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
    image = Image.open(requests.get(url, stream=True).raw)

    # Prepare inputs
    texts = [
        "this is a photo of a cat.",
        "this is a photo of a dog.",
        "this is a photo of a bird."
    ]

    inputs = processor(
        text=texts,
        images=image,
        padding="max_length",
        max_length=64,
        return_tensors="pt"
    ).to(model.device)

    # Forward pass
    with torch.inference_mode():
        outputs = model(**inputs)

    # Compute probabilities
    probs = torch.sigmoid(outputs.logits_per_image)

    print(f"\nImage classification results:")
    for text, prob in zip(texts, probs[0]):
        print(f"  {prob:.1%} - {text}")

    print("\n" + "=" * 60)
    print("✓ SigLIP2 loader test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
