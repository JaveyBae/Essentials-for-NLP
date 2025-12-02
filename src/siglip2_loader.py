"""
SigLIP2 Model Loader.
Centralized loading for SigLIP2 vision-language models.
"""

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
    use_flash_attention: bool = True
) -> Tuple[AutoModel, AutoProcessor]:
    """
    Load SigLIP2 model and processor.

    Args:
        model_name: Model variant to use (see SUPPORTED_MODELS)
        quantization: Must be None (quantization not supported for SigLIP2)
        device: Device to run on ("cuda" or "cpu")
        cache_dir: Optional directory for model cache
        use_flash_attention: Use Flash Attention 2 for memory efficiency (default: True)

    Returns:
        Tuple of (model, processor)
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Choose from: {list(SUPPORTED_MODELS.keys())}"
        )

    if quantization is not None:
        raise ValueError(
            f"Quantization not supported for SigLIP2 models. "
            f"BitsAndBytes causes dtype mismatches in attention layers. "
            f"Use F32 (base/large models) or float16 (so400m/giant-opt models) only."
        )

    model_id = SUPPORTED_MODELS[model_name]

    # Configure model loading
    load_kwargs = {
        "device_map": "auto",
    }

    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir

    # Configure attention implementation
    # Base and Large models have issues with Flash Attention/SDPA - use eager attention (F32)
    # So400m and Giant-opt models work fine with optimized attention (float16)
    if "base" in model_name or "large" in model_name:
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

    # Load processor (tokenizer + image processor)
    processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir=cache_dir
    )

    # Load model
    model = AutoModel.from_pretrained(model_id, **load_kwargs)
    model.eval()

    logger.info(f"✓ SigLIP2 model loaded: {model_name}")

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
