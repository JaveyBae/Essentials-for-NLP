"""
VLM Model Loader for Qwen3-VL Vision-Language Models.

This loader is specifically for loading Qwen3-VL models for visual understanding tasks.
Supports only bfloat16 (full precision) and AWQ-INT4 (4-bit quantization).

For text-only models (definition generation), use models/definition_generator.py instead.

Optimizations:
- Model caching to avoid reloading
- Context manager support for automatic cleanup
- Better GPU memory management
"""

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from typing import Optional, Tuple
import logging
import gc
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model cache
_MODEL_CACHE = {}


class Qwen3VLModelLoader:
    """Loads and manages Qwen3-VL vision-language models for VWSD tasks."""

    SUPPORTED_MODELS = {
        # Qwen3-VL models (SOTA as of 2025)
        "qwen3-vl-2b": "Qwen/Qwen3-VL-2B-Instruct",
        "qwen3-vl-4b": "Qwen/Qwen3-VL-4B-Instruct",
        "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
        "qwen3-vl-32b": "Qwen/Qwen3-VL-32B-Instruct",
    }

    def __init__(self, model_name: str, quantization: Optional[str] = None, device: str = "cuda", use_cache: bool = True):
        """
        Initialize VLM model loader.

        Args:
            model_name: Short name of the model (e.g., 'qwen3-vl-8b')
            quantization: Quantization method ('4bit' for AWQ-INT4, or None for bfloat16)
            device: Device to load model on ('cuda' or 'cpu')
            use_cache: Whether to use cached models (default: True)
        """
        self.model_name = model_name
        self.quantization = quantization
        self.device = device
        self.use_cache = use_cache

        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Model '{model_name}' not supported. "
                f"Choose from: {list(self.SUPPORTED_MODELS.keys())}"
            )

        if quantization is not None and quantization != "4bit":
            raise ValueError(
                f"Unsupported quantization: {quantization}. "
                f"Only '4bit' (AWQ-INT4) or None (bfloat16) are supported."
            )

        self.model_id = self.SUPPORTED_MODELS[model_name]
        self.model = None
        self.processor = None
        self._cache_key = f"{model_name}_{quantization}"

    def load(self) -> Tuple[AutoModelForImageTextToText, AutoProcessor]:
        """
        Load the model and processor (with caching support).

        Returns:
            Tuple of (model, processor)
        """
        # Check cache first
        if self.use_cache and self._cache_key in _MODEL_CACHE:
            logger.info(f"Using cached model: {self.model_id}")
            cached = _MODEL_CACHE[self._cache_key]
            self.model = cached['model']
            self.processor = cached['processor']
            return self.model, self.processor

        logger.info(f"Loading model: {self.model_id}")
        logger.info(f"Quantization: {self.quantization or 'None (full precision)'}")
        logger.info(f"Device: {self.device}")

        # Prepare loading arguments
        load_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
        }

        # Configure quantization: only AWQ-INT4 (4bit) or bfloat16 (full precision)
        if self.quantization == "4bit":
            logger.info("Using AWQ-INT4 quantization (4-bit via bitsandbytes)")
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            load_kwargs["quantization_config"] = quantization_config
        else:
            # Full precision with bfloat16 (only supported dtype for VLM)
            logger.info("Using bfloat16 precision (full precision)")
            load_kwargs["dtype"] = torch.bfloat16

        # Enable Flash Attention 2 for better performance (especially with multi-image)
        try:
            load_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Flash Attention 2 enabled for optimized inference")
        except Exception as e:
            logger.warning(f"Could not enable Flash Attention 2: {e}. Falling back to default attention.")

        # Load model (using AutoModelForImageTextToText for Qwen3-VL)
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                **load_kwargs
            )
            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

        # Load processor
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_id
            )
            logger.info("Processor loaded successfully")

            # Set padding side to left for batch processing (Qwen3-VL requirement)
            self.processor.tokenizer.padding_side = 'left'

        except Exception as e:
            logger.error(f"Error loading processor: {e}")
            raise

        # Print memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")

        # Cache the model and processor
        if self.use_cache:
            _MODEL_CACHE[self._cache_key] = {
                'model': self.model,
                'processor': self.processor
            }
            logger.info(f"Model cached with key: {self._cache_key}")

        return self.model, self.processor

    def get_model(self):
        """Get the loaded model."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self.model

    def get_processor(self):
        """Get the loaded processor."""
        if self.processor is None:
            raise RuntimeError("Processor not loaded. Call load() first.")
        return self.processor

    def cleanup(self):
        """Clean up model from memory and cache."""
        if self._cache_key in _MODEL_CACHE:
            del _MODEL_CACHE[self._cache_key]
            logger.info(f"Removed model from cache: {self._cache_key}")

        if self.model is not None:
            del self.model
            self.model = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        # Force garbage collection and clear CUDA cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")

    def __enter__(self):
        """Context manager entry."""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.cleanup()
        return False

    @staticmethod
    def clear_all_cache():
        """Clear all cached models."""
        global _MODEL_CACHE
        _MODEL_CACHE.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("All model caches cleared")


def load_model(model_name: str = "qwen3-vl-8b",
               quantization: Optional[str] = None,
               device: str = "cuda",
               use_cache: bool = True) -> Tuple[AutoModelForImageTextToText, AutoProcessor]:
    """
    Convenience function to load a Qwen3-VL vision-language model.

    Args:
        model_name: Model name ('qwen3-vl-2b', 'qwen3-vl-4b', 'qwen3-vl-8b', 'qwen3-vl-32b')
        quantization: Quantization method ('4bit' for AWQ-INT4, or None for bfloat16)
        device: Device to load on ('cuda' or 'cpu')
        use_cache: Whether to use cached models (default: True)

    Returns:
        Tuple of (model, processor)

    Example:
        >>> # Full precision (bfloat16)
        >>> model, processor = load_model("qwen3-vl-8b")
        >>>
        >>> # 4-bit quantization (AWQ-INT4)
        >>> model, processor = load_model("qwen3-vl-8b", quantization="4bit")
        >>>
        >>> # Use with context manager for automatic cleanup
        >>> with Qwen3VLModelLoader("qwen3-vl-8b") as loader:
        ...     model = loader.get_model()
        ...     # Use model...
        >>> # Model automatically cleaned up
    """
    loader = Qwen3VLModelLoader(model_name, quantization, device, use_cache)
    return loader.load()


def main():
    """Test model loading."""
    print("Testing Qwen3-VL model loader...")
    print("\nSupported models:")
    for name, model_id in Qwen3VLModelLoader.SUPPORTED_MODELS.items():
        print(f"  {name}: {model_id}")

    # Test loading (uncomment when environment is ready)
    # print("\nLoading Qwen3-VL-8B...")
    # model, processor = load_model("qwen3-vl-8b")
    # print("Model loaded successfully!")


if __name__ == "__main__":
    main()
