"""
Qwen3 Definition Generator for Word Sense Disambiguation.
Generates contextual sense definitions using Qwen3 language models.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List, Dict, Optional
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Qwen3DefinitionGenerator:
    """Generate word sense definitions using Qwen3 text models."""

    SUPPORTED_MODELS = {
        # Dense models (ordered by size)
        "qwen3-4b": "Qwen/Qwen3-4B",       # Budget-friendly, fast
        "qwen3-8b": "Qwen/Qwen3-8B",       # High quality, recommended
        "qwen3-14b": "Qwen/Qwen3-14B",     # Best quality
    }

    DEFINITION_PROMPT_TEMPLATE = """Word: {target_word}
Context: {full_phrase}
Definition (5-10 words):"""

    def __init__(
        self,
        model_name: str = "qwen3-8b",
        quantization: Optional[str] = "4bit",
        device: str = "cuda",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize Qwen3 definition generator.

        Args:
            model_name: Model size to use (see SUPPORTED_MODELS)
            quantization: Quantization method ("4bit", "bf16", or None for fp16)
            device: Device to run on ("cuda" or "cpu")
            cache_dir: Optional directory for model cache
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Choose from: {list(self.SUPPORTED_MODELS.keys())}"
            )

        self.model_name = model_name
        self.device = device
        self.quantization = quantization

        model_id = self.SUPPORTED_MODELS[model_name]
        logger.info(f"Loading {model_name} ({quantization or 'fp16'})...")

        # Configure model loading
        load_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
        }

        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir

        # Configure quantization
        if quantization == "4bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif quantization == "bf16" or quantization is None:
            load_kwargs["dtype"] = torch.bfloat16
        else:
            raise ValueError(f"Unsupported quantization: {quantization}. Choose from: '4bit', 'bf16', or None")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            cache_dir=cache_dir
        )

        # Set padding for batch generation
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        self.model.eval()

    def generate_definition(
        self,
        target_word: str,
        full_phrase: str,
        max_tokens: int = 60,
        temperature: float = 0.0
    ) -> str:
        """
        Generate a single definition for a target word in full phrase.

        Args:
            target_word: The ambiguous word
            full_phrase: Sentence/phrase containing the target word
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature (0 = deterministic, default)

        Returns:
            Generated definition string (cleaned)
        """
        prompt = self.DEFINITION_PROMPT_TEMPLATE.format(
            target_word=target_word,
            full_phrase=full_phrase
        )

        # Format as chat message with few-shot examples to show desired format
        system_prompt = """You are a lexical semantic expert. Provide concise contextual definitions for ambiguous words, similar to WordNet and BabelNet sense definitions.

For each word in context, provide ONLY a short definition (5-10 words) that captures the specific sense used. No explanations, no meta-commentary.

Examples:
Word: bank
Full Phrase: river bank
Definition: Edge or slope beside a body of water

Word: bass
Full Phrase: bass guitar
Definition: Low-pitched stringed musical instrument

Word: goal
Full Phrase: football goal
Definition: Scored point or target area in sports"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # Disable thinking mode for concise definitions
        )

        # Tokenize
        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)

        # Generate (use greedy decoding when temperature=0)
        with torch.inference_mode():
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            # Use greedy decoding for temperature=0, sampling otherwise
            if temperature == 0.0:
                gen_kwargs["do_sample"] = False
            else:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = 0.9

            outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decode only the generated part (return raw response directly)
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip()

        return generated_text

    def generate_batch(
        self,
        instances: List[Dict[str, str]],
        batch_size: int = 32,
        max_tokens: int = 60,
        temperature: float = 0.0,
        show_progress: bool = True
    ) -> List[str]:
        """
        Generate definitions for multiple instances in batches.

        Args:
            instances: List of dicts with 'target_word' and 'full_phrase' keys
            batch_size: Number of instances to process at once
            max_tokens: Maximum tokens per definition
            temperature: Generation temperature (0 = deterministic, default)
            show_progress: Show progress bar

        Returns:
            List of generated definitions (cleaned, same order as input)
        """
        definitions = []
        total_batches = (len(instances) + batch_size - 1) // batch_size

        iterator = range(0, len(instances), batch_size)
        if show_progress:
            iterator = tqdm(
                iterator,
                total=total_batches,
                desc=f"Generating definitions ({self.model_name})"
            )

        for i in iterator:
            batch = instances[i:i + batch_size]

            # Create prompts for batch
            prompts = [
                self.DEFINITION_PROMPT_TEMPLATE.format(
                    target_word=inst['target_word'],
                    full_phrase=inst['full_phrase']
                )
                for inst in batch
            ]

            # Format as chat messages with few-shot system prompt
            system_prompt = """You are a lexical semantic expert. Provide concise contextual definitions for ambiguous words, similar to WordNet and BabelNet sense definitions.

For each word in context, provide ONLY a short definition (5-10 words) that captures the specific sense used. No explanations, no meta-commentary.

Examples:
Word: bank
Full Phrase: river bank
Definition: Edge or slope beside a body of water

Word: bass
Full Phrase: bass guitar
Definition: Low-pitched stringed musical instrument

Word: goal
Full Phrase: football goal
Definition: Scored point or target area in sports"""

            messages_list = [
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": p}
                ]
                for p in prompts
            ]
            texts = [
                self.tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False  # Disable thinking mode for concise definitions
                )
                for msgs in messages_list
            ]

            # Tokenize batch
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)

            # Generate batch (use greedy decoding when temperature=0)
            with torch.inference_mode():
                gen_kwargs = {
                    "max_new_tokens": max_tokens,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }

                # Use greedy decoding for temperature=0, sampling otherwise
                if temperature == 0.0:
                    gen_kwargs["do_sample"] = False
                else:
                    gen_kwargs["do_sample"] = True
                    gen_kwargs["temperature"] = temperature
                    gen_kwargs["top_p"] = 0.9

                outputs = self.model.generate(**inputs, **gen_kwargs)

            # Decode batch results (return raw responses directly)
            for j, output in enumerate(outputs):
                input_length = inputs['input_ids'][j].shape[0]
                generated_text = self.tokenizer.decode(
                    output[input_length:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                ).strip()

                definitions.append(generated_text)

        return definitions

    def cleanup(self):
        """Free GPU memory by deleting model and clearing cache."""
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.cleanup()


def main():
    """Test the definition generator."""
    print("=" * 80)
    print("Qwen3 Definition Generator Test")
    print("=" * 80)

    # Test instances (matching actual VWSD data format: short 2-3 word phrases)
    test_instances = [
        {
            "target_word": "goal",
            "full_phrase": "football goal"
        },
        {
            "target_word": "bank",
            "full_phrase": "river bank"
        },
        {
            "target_word": "bank",
            "full_phrase": "money bank"
        },
        {
            "target_word": "bass",
            "full_phrase": "bass guitar"
        },
        {
            "target_word": "seat",
            "full_phrase": "eating seat"
        }
    ]

    # Test with smaller model for demo
    print("\nTesting with Qwen3-4B (4-bit)...")
    print("-" * 80)

    with Qwen3DefinitionGenerator("qwen3-4b", quantization="4bit") as generator:
        # Single generation test
        print("\n[Single Generation Test]")
        definition = generator.generate_definition(
            target_word="goal",
            full_phrase="football goal"
        )
        print(f"Target: 'goal' in 'football goal'")
        print(f"Definition: {definition}")

        # Batch generation test
        print("\n[Batch Generation Test]")
        definitions = generator.generate_batch(test_instances, batch_size=3)

        for inst, defn in zip(test_instances, definitions):
            print(f"\nWord: '{inst['target_word']}'")
            print(f"Full Phrase: {inst['full_phrase']}")
            print(f"Definition: {defn}")

    print("\n" + "=" * 80)
    print("Test complete! Model automatically cleaned up via context manager.")
    print("=" * 80)


if __name__ == "__main__":
    main()
