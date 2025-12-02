"""
VWSD Ranking Metric Evaluation
Adapted from: https://github.com/asahi417/visual-wsd-baseline

Optimizations:
- Parallel evaluation across languages
- Numpy vectorization for metrics
- Progress tracking with tqdm
"""

import argparse
import json
import os
from statistics import mean
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
from ranx import Qrels, Run, evaluate
from tqdm import tqdm


def get_metric(prediction: List[List[str]], gold_ranking: List[str]) -> tuple:
    """
    Get MRR and HIT Rate (optimized with numpy).

    Args:
        prediction: List of ranked predictions (list of image filenames)
        gold_ranking: List of gold image filenames

    Returns:
        Tuple of (mrr_score, hit_score)

    Example:
        >>> p = [['a', 'b', 'c'], ['a', 'b', 'c'], ['a', 'b', 'c']]
        >>> r = ['a', 'a', 'b']
        >>> mrr, hit = get_metric(p, r)
        >>> print(f"MRR: {mrr:.3f}, Hit@1: {hit:.3f}")
    """
    # Validate predictions
    assert all(len(set(p)) == len(p) for p in prediction), "Predictions contain duplicates"
    assert all(r in p for p, r in zip(prediction, gold_ranking)), "Gold not in predictions"

    # Vectorized computation using numpy for better performance
    ranks = np.array([p.index(r) for p, r in zip(prediction, gold_ranking)])

    # Calculate MRR (Mean Reciprocal Rank)
    mrr = float(np.mean(1.0 / (ranks + 1)))

    # Calculate Hit@1 (Hit Rate)
    hit = float(np.mean(ranks == 0))

    return mrr, hit


def load_predictions(prediction_file: str) -> List[List[str]]:
    """
    Load predictions from file.

    File format: Each line is tab-separated ranked list of image filenames
    Example: image1.jpg\timage2.jpg\timage3.jpg...

    Args:
        prediction_file: Path to prediction file

    Returns:
        List of predictions (each prediction is a list of image filenames)
    """
    with open(prediction_file, 'r', encoding='utf-8') as f:
        predictions = [line.strip().split('\t') for line in f if line.strip()]
    return predictions


def load_gold_labels(gold_file: str) -> List[str]:
    """
    Load gold labels from file.

    File format: One image filename per line

    Args:
        gold_file: Path to gold label file

    Returns:
        List of gold image filenames
    """
    with open(gold_file, 'r', encoding='utf-8') as f:
        gold = [line.strip() for line in f if line.strip()]
    return gold


def compute_metrics(predictions: List[List[str]],
                   gold_labels: List[str],
                   metrics: List[str] = None) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Args:
        predictions: List of ranked predictions
        gold_labels: List of gold labels
        metrics: List of metrics to compute (for ranx library)

    Returns:
        Dictionary of metric scores
    """
    if metrics is None:
        metrics = ["hit_rate@1", "map@5", "mrr@5", "ndcg@5", "map@10", "mrr@10", "ndcg@10"]

    # Compute official metrics (MRR and Hit@1)
    mrr_official, hit_official = get_metric(predictions, gold_labels)

    # Compute ranx metrics
    # Create qrels (gold labels) in ranx format
    qrels_dict = {str(n): {gold: 1} for n, gold in enumerate(gold_labels)}

    # Create run (predictions) in ranx format
    # Score = 1/(rank+1) for each candidate
    run_dict = {
        str(n): {candidate: 1 / (rank + 1) for rank, candidate in enumerate(pred)}
        for n, pred in enumerate(predictions)
    }

    # Evaluate with ranx
    ranx_metrics = evaluate(Qrels(qrels_dict), Run(run_dict), metrics=metrics)

    # Combine results
    results = {
        'mrr_official': mrr_official,
        'hit_official': hit_official,
    }
    results.update(ranx_metrics)

    return results


def evaluate_language(prediction_file: str,
                     gold_file: str,
                     language: str,
                     metrics: List[str] = None) -> Dict[str, float]:
    """
    Evaluate predictions for a single language.

    Args:
        prediction_file: Path to prediction file
        gold_file: Path to gold label file
        language: Language code ('en', 'fa', 'it')
        metrics: List of metrics to compute

    Returns:
        Dictionary of metric scores with language suffix
    """
    # Load data
    predictions = load_predictions(prediction_file)
    gold_labels_full = load_gold_labels(gold_file)

    # Handle partial evaluation (when testing with --max-instances)
    if len(predictions) < len(gold_labels_full):
        print(f"  Note: Partial evaluation - {len(predictions)} predictions out of {len(gold_labels_full)} instances")
        gold_labels = gold_labels_full[:len(predictions)]
    elif len(predictions) > len(gold_labels_full):
        raise ValueError(
            f"Number of predictions ({len(predictions)}) "
            f"> number of gold labels ({len(gold_labels_full)})"
        )
    else:
        gold_labels = gold_labels_full

    # Compute metrics
    results = compute_metrics(predictions, gold_labels, metrics)

    # Add language suffix
    results_with_lang = {f"{k}/{language}": v for k, v in results.items()}

    return results_with_lang


def evaluate_language_wrapper(args):
    """Wrapper for parallel evaluation."""
    lang, pred_file, gold_file, metrics = args
    try:
        results = evaluate_language(pred_file, gold_file, lang, metrics)
        return lang, results, None
    except Exception as e:
        return lang, None, str(e)


def main():
    parser = argparse.ArgumentParser(description="Compute VWSD ranking metrics")
    parser.add_argument('-p', '--prediction-dir', help='Directory containing prediction files',
                       type=str, required=True)
    parser.add_argument('-d', '--data-dir', help='Directory containing gold label files',
                       type=str, default='data/test_data')
    parser.add_argument('-l', '--languages', help='Languages to evaluate',
                       type=str, nargs='+', default=['en', 'fa', 'it'])
    parser.add_argument('-m', '--metrics', help='Metrics to report',
                       type=str, nargs='+',
                       default=["hit_rate@1", "map@5", "mrr@5", "ndcg@5",
                               "map@10", "mrr@10", "ndcg@10"])
    parser.add_argument('-o', '--output-file', help='Output file (JSONL format)',
                       default='results/rank_metrics.jsonl', type=str)
    parser.add_argument('--model-name', help='Model name for results',
                       type=str, default=None)
    parser.add_argument('--workers', help='Number of parallel workers',
                       type=int, default=3)

    opt = parser.parse_args()

    # Results dictionary
    metric_dict = {'model': opt.model_name or opt.prediction_dir}

    # Prepare evaluation tasks
    eval_tasks = []
    for lang in opt.languages:
        pred_file = os.path.join(opt.prediction_dir, f"prediction.{lang}.txt")
        gold_file = os.path.join(opt.data_dir, f"{lang}.test.gold.v1.1.txt")

        if not os.path.exists(pred_file):
            print(f"Warning: Prediction file not found: {pred_file}")
            continue

        if not os.path.exists(gold_file):
            print(f"Warning: Gold file not found: {gold_file}")
            continue

        eval_tasks.append((lang, pred_file, gold_file, opt.metrics))

    # Parallel evaluation across languages
    print(f"\nEvaluating {len(eval_tasks)} languages in parallel (workers={opt.workers})...")

    with ThreadPoolExecutor(max_workers=opt.workers) as executor:
        futures = {executor.submit(evaluate_language_wrapper, task): task[0]
                  for task in eval_tasks}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            lang, lang_results, error = future.result()

            if error:
                print(f"\nError evaluating {lang.upper()}: {error}")
                continue

            metric_dict.update(lang_results)

            # Print language results
            print(f"\n{lang.upper()} Results:")
            print(f"  Hit@1: {lang_results[f'hit_official/{lang}']:.4f}")
            print(f"  MRR: {lang_results[f'mrr_official/{lang}']:.4f}")

    # Compute macro-averages across languages
    evaluated_langs = [l for l in opt.languages
                      if f'hit_official/{l}' in metric_dict]

    if len(evaluated_langs) > 0:
        for metric_name in ['mrr_official', 'hit_official'] + opt.metrics:
            values = [metric_dict[f"{metric_name}/{lang}"]
                     for lang in evaluated_langs
                     if f"{metric_name}/{lang}" in metric_dict]
            if values:
                metric_dict[f"{metric_name}/avg"] = sum(values) / len(values)

    # Save results
    os.makedirs(os.path.dirname(opt.output_file), exist_ok=True)

    if os.path.exists(opt.output_file):
        with open(opt.output_file, 'r') as f:
            metric_all = [json.loads(line) for line in f if line.strip()]
    else:
        metric_all = []

    metric_all.append(metric_dict)

    with open(opt.output_file, 'w') as f:
        f.write("\n".join([json.dumps(m) for m in metric_all]))

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    df = pd.DataFrame(metric_all)
    print(df.to_markdown(index=False))
    print("\n" + "="*60)
    print(f"Results saved to: {opt.output_file}")


if __name__ == '__main__':
    main()
