"""
Two-Stage Cascade Reranker for Visual Word Sense Disambiguation.

Pipeline:
1. Stage 1: SigLIP2 ranks all 10 candidate images (fast, embedding-based)
2. Stage 2: Qwen3-VL reranks top-K candidates (accurate, generative)
3. Final ranking: [VLM-reranked top-K] + [SigLIP2-ranked rest]

This approach reduces VLM compute by 70-90% while maintaining accuracy.
"""

import torch
from PIL import Image
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CascadeResult:
    """Result from cascade reranking."""
    instance_id: int
    ranked_images: List[str]  # Final ranked list (VLM top-K + SigLIP2 rest)
    scores: List[float]  # Final scores
    target_word: str
    full_phrase: str
    gold_image: str = None
    # Stage-specific info for analysis
    stage1_ranking: List[str] = None  # SigLIP2 initial ranking
    stage1_scores: List[float] = None
    stage2_ranking: List[str] = None  # VLM reranking of top-K
    stage2_scores: List[float] = None


class CascadeReranker:
    """
    Two-stage cascade reranker combining SigLIP2 speed with VLM accuracy.

    Usage:
        reranker = CascadeReranker(
            siglip2_model="siglip2-so400m-patch14-384",
            vlm_model="qwen3-vl-8b",
            vlm_method="matching_cot",
            top_k=3
        )
        results = reranker.process_instances(instances_data, data_loader)
    """

    def __init__(
        self,
        siglip2_model: str = "siglip2-so400m-patch14-384",
        vlm_model: str = "qwen3-vl-8b",
        vlm_method: str = "matching_cot",
        vlm_quantization: str = "bfloat16",
        top_k: int = 3,
        device: str = "cuda",
        definitions: Optional[Dict] = None
    ):
        """
        Initialize cascade reranker.

        Args:
            siglip2_model: SigLIP2 model variant for Stage 1
            vlm_model: Qwen3-VL model for Stage 2
            vlm_method: VLM inference method ('matching', 'matching_cot', 'description')
            vlm_quantization: VLM quantization ('bfloat16' or '4bit')
            top_k: Number of candidates to rerank with VLM (default: 3)
            device: Device to run inference on
            definitions: Optional definitions dict for 'description' method
        """
        self.siglip2_model_name = siglip2_model
        self.vlm_model_name = vlm_model
        self.vlm_method = vlm_method
        self.vlm_quantization = vlm_quantization
        self.top_k = top_k
        self.device = device
        self.definitions = definitions or {}

        # Models loaded lazily
        self._siglip2_model = None
        self._siglip2_processor = None
        self._siglip2_engine = None

        self._vlm_model = None
        self._vlm_processor = None
        self._vlm_engine = None

        logger.info(f"CascadeReranker initialized:")
        logger.info(f"  Stage 1: SigLIP2 ({siglip2_model})")
        logger.info(f"  Stage 2: Qwen3-VL ({vlm_model}, method={vlm_method}, top_k={top_k})")

    def _load_siglip2(self):
        """Lazy-load SigLIP2 model."""
        if self._siglip2_engine is None:
            from src.siglip2_loader import load_siglip2_model
            from src.siglip2_inference import SigLIP2Inference

            logger.info(f"Loading SigLIP2 model: {self.siglip2_model_name}...")
            self._siglip2_model, self._siglip2_processor = load_siglip2_model(
                model_name=self.siglip2_model_name,
                quantization=None,
                device=self.device
            )
            self._siglip2_engine = SigLIP2Inference(
                self._siglip2_model,
                self._siglip2_processor,
                device=self.device
            )
            logger.info("SigLIP2 model loaded")

    def _unload_siglip2(self):
        """Unload SigLIP2 to free GPU memory before loading VLM."""
        if self._siglip2_model is not None:
            del self._siglip2_model
            del self._siglip2_processor
            del self._siglip2_engine
            self._siglip2_model = None
            self._siglip2_processor = None
            self._siglip2_engine = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("SigLIP2 model unloaded to free GPU memory")

    def _load_vlm(self):
        """Lazy-load VLM model."""
        if self._vlm_engine is None:
            from src.qwen_vlm_loader import load_model
            from src.qwen_vlm_inference import QwenVLInference

            logger.info(f"Loading VLM model: {self.vlm_model_name}...")
            self._vlm_model, self._vlm_processor = load_model(
                model_name=self.vlm_model_name,
                quantization=self.vlm_quantization,
                device=self.device
            )
            self._vlm_engine = QwenVLInference(
                self._vlm_model,
                self._vlm_processor,
                device=self.device,
                definitions=self.definitions
            )
            logger.info("VLM model loaded")

    def _unload_vlm(self):
        """Unload VLM to free GPU memory."""
        if self._vlm_model is not None:
            del self._vlm_model
            del self._vlm_processor
            del self._vlm_engine
            self._vlm_model = None
            self._vlm_processor = None
            self._vlm_engine = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("VLM model unloaded")

    def stage1_rank(
        self,
        instances_data: List[Dict]
    ) -> List[Tuple[List[str], List[float]]]:
        """
        Stage 1: Rank all candidates using SigLIP2.

        Args:
            instances_data: List of dicts with 'images', 'image_filenames',
                           'target_word', 'full_phrase'

        Returns:
            List of (ranked_filenames, scores) tuples
        """
        self._load_siglip2()

        results = self._siglip2_engine.rank_images(
            instances_data=instances_data,
            definitions=None,
            use_definitions=False
        )

        return results

    def stage2_rerank(
        self,
        instances_data: List[Dict],
        stage1_results: List[Tuple[List[str], List[float]]]
    ) -> List[Tuple[List[str], List[float]]]:
        """
        Stage 2: Rerank top-K candidates using VLM.

        Args:
            instances_data: Original instance data (with all images)
            stage1_results: Results from Stage 1 (rankings and scores)

        Returns:
            List of (reranked_top_k_filenames, scores) tuples
        """
        self._load_vlm()

        reranked_results = []

        for instance_data, (stage1_ranking, stage1_scores) in zip(instances_data, stage1_results):
            # Get top-K candidates from Stage 1
            top_k_filenames = stage1_ranking[:self.top_k]

            # Build mapping from filename to image
            filename_to_image = {
                fn: img for fn, img in zip(
                    instance_data['image_filenames'],
                    instance_data['images']
                )
            }

            # Prepare subset data for VLM (only top-K images)
            top_k_images = [filename_to_image[fn] for fn in top_k_filenames]

            subset_data = [{
                'images': top_k_images,
                'image_filenames': top_k_filenames,
                'target_word': instance_data['target_word'],
                'full_phrase': instance_data['full_phrase']
            }]

            # Run VLM inference on top-K only
            vlm_results = self._vlm_engine.rank_images(
                instances_data=subset_data,
                method=self.vlm_method
            )

            reranked_filenames, reranked_scores = vlm_results[0]
            reranked_results.append((reranked_filenames, reranked_scores))

        return reranked_results

    def combine_rankings(
        self,
        stage1_results: List[Tuple[List[str], List[float]]],
        stage2_results: List[Tuple[List[str], List[float]]]
    ) -> List[Tuple[List[str], List[float]]]:
        """
        Combine Stage 1 and Stage 2 rankings.

        Final ranking: [VLM-reranked top-K] + [SigLIP2-ranked rest]

        Args:
            stage1_results: Full rankings from SigLIP2
            stage2_results: Reranked top-K from VLM

        Returns:
            List of final (ranked_filenames, scores) tuples
        """
        combined_results = []

        for (stage1_ranking, stage1_scores), (stage2_ranking, stage2_scores) in zip(
            stage1_results, stage2_results
        ):
            # VLM reranked top-K comes first
            final_ranking = list(stage2_ranking)
            final_scores = list(stage2_scores)

            # Add remaining items from Stage 1 (not in top-K)
            top_k_set = set(stage2_ranking)
            for filename, score in zip(stage1_ranking, stage1_scores):
                if filename not in top_k_set:
                    final_ranking.append(filename)
                    # Adjust score to be lower than VLM scores
                    # This ensures SigLIP2 items always rank after VLM items
                    adjusted_score = score * 0.1  # Scale down SigLIP2 scores
                    final_scores.append(adjusted_score)

            combined_results.append((final_ranking, final_scores))

        return combined_results

    def process_instances(
        self,
        instances: List,  # List of VWSDInstance
        data_loader,  # VWSDDataLoader
        show_progress: bool = True
    ) -> List[CascadeResult]:
        """
        Process multiple instances through the cascade pipeline.

        Args:
            instances: List of VWSDInstance objects
            data_loader: VWSDDataLoader for loading images
            show_progress: Whether to show progress bars

        Returns:
            List of CascadeResult objects
        """
        from tqdm import tqdm

        total_start = time.time()

        # Prepare all instance data
        logger.info(f"Preparing {len(instances)} instances...")
        all_instances_data = []
        for instance in instances:
            images = data_loader.load_instance_images(instance)
            all_instances_data.append({
                'images': images,
                'image_filenames': instance.candidate_images,
                'target_word': instance.target_word,
                'full_phrase': instance.full_phrase
            })

        # Stage 1: SigLIP2 ranking
        logger.info("=" * 60)
        logger.info("[Stage 1/2] Running SigLIP2 initial ranking...")
        stage1_start = time.time()

        stage1_results = []
        iterator = tqdm(all_instances_data, desc="Stage 1: SigLIP2") if show_progress else all_instances_data
        for instance_data in iterator:
            result = self.stage1_rank([instance_data])
            stage1_results.append(result[0])

        stage1_time = time.time() - stage1_start
        logger.info(f"Stage 1 complete: {stage1_time:.2f}s ({stage1_time/len(instances):.3f}s/instance)")

        # Unload SigLIP2, load VLM
        self._unload_siglip2()

        # Stage 2: VLM reranking
        logger.info("=" * 60)
        logger.info(f"[Stage 2/2] Running VLM reranking (top-{self.top_k}, method={self.vlm_method})...")
        stage2_start = time.time()

        stage2_results = []
        iterator = enumerate(zip(all_instances_data, stage1_results))
        if show_progress:
            iterator = tqdm(list(iterator), desc=f"Stage 2: VLM (top-{self.top_k})")

        for idx, (instance_data, stage1_result) in iterator:
            result = self.stage2_rerank([instance_data], [stage1_result])
            stage2_results.append(result[0])

        stage2_time = time.time() - stage2_start
        logger.info(f"Stage 2 complete: {stage2_time:.2f}s ({stage2_time/len(instances):.3f}s/instance)")

        # Combine rankings
        logger.info("Combining rankings...")
        combined_results = self.combine_rankings(stage1_results, stage2_results)

        # Build final results
        results = []
        for idx, instance in enumerate(instances):
            final_ranking, final_scores = combined_results[idx]
            stage1_ranking, stage1_scores = stage1_results[idx]
            stage2_ranking, stage2_scores = stage2_results[idx]

            result = CascadeResult(
                instance_id=instance.instance_id,
                ranked_images=final_ranking,
                scores=final_scores,
                target_word=instance.target_word,
                full_phrase=instance.full_phrase,
                gold_image=instance.gold_image,
                stage1_ranking=stage1_ranking,
                stage1_scores=stage1_scores,
                stage2_ranking=stage2_ranking,
                stage2_scores=stage2_scores
            )
            results.append(result)

        # Summary
        total_time = time.time() - total_start
        logger.info("=" * 60)
        logger.info(f"Cascade pipeline complete:")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  Stage 1 (SigLIP2): {stage1_time:.2f}s ({100*stage1_time/total_time:.1f}%)")
        logger.info(f"  Stage 2 (VLM): {stage2_time:.2f}s ({100*stage2_time/total_time:.1f}%)")
        logger.info(f"  Compute saved: {100*(1 - self.top_k/10):.0f}% (VLM processed {self.top_k}/10 images)")
        logger.info("=" * 60)

        return results

    def cleanup(self):
        """Release all GPU resources."""
        self._unload_siglip2()
        self._unload_vlm()


def main():
    """Test cascade reranker."""
    print("=" * 60)
    print("Cascade Reranker for VWSD")
    print("=" * 60)
    print("\nPipeline:")
    print("  Stage 1: SigLIP2 ranks all 10 candidates (fast)")
    print("  Stage 2: Qwen3-VL reranks top-K (accurate)")
    print("  Final: [VLM top-K] + [SigLIP2 rest]")
    print("\nBenefits:")
    print("  - 70-90% VLM compute reduction")
    print("  - Combines SigLIP2 speed with VLM reasoning")
    print("\nUsage:")
    print("  python main.py --cascade --topk 3 --reranker-method matching_cot")
    print("=" * 60)


if __name__ == "__main__":
    main()
