""" 
Portable Image Evaluator
Can be used in other projects by copying the vwsd_utils folder.
"""
import argparse
import logging
import os
from os.path import join as pj
import glob
import sys

# Add current directory to path to allow relative imports if run as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from vwsd_utils import CLIP, data_loader, plot

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def run_evaluation(
    image_dir: str,
    output_dir: str = "result",
    data_dir: str = "dataset",
    language: str = "en",
    model_clip: str = None,
    batch_size: int = None,
    plot_results: bool = False,
    experiment_name: str = "generated_images",
    image_pattern: str = "*.jpg"
):
    """
    Run evaluation using generated images.
    """
    
    # Load dataset
    try:
        data = data_loader(data_dir)[language]
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return

    # Get generated images
    generated_images = sorted(glob.glob(pj(image_dir, image_pattern)))
    
    if not generated_images:
        logging.error(f"No images found in {image_dir} matching pattern {image_pattern}")
        return
    
    logging.info(f"Found {len(generated_images)} generated images")
    logging.info(f"Dataset has {len(data)} test samples")
    
    # Check count
    if len(generated_images) != len(data):
        logging.warning(f"Number of generated images ({len(generated_images)}) does not match dataset size ({len(data)})")
        logging.warning("Will use available images up to the minimum of both")
        num_samples = min(len(generated_images), len(data))
    else:
        num_samples = len(data)

    # Load CLIP model
    model_name = model_clip if model_clip is not None else 'openai/clip-vit-large-patch14-336'
    logging.info(f"Loading CLIP model: {model_name}")
    clip = CLIP(model_name)

    # Run inference
    result = []
    for n in range(num_samples):
        d = data[n]
        query_image = generated_images[n]
        
        logging.info(f"{n+1}/{num_samples}: {os.path.basename(query_image)} -> {d['target phrase']}")
        
        # Calculate image-to-image similarity
        sim = clip.get_image_similarity(
            query_images=query_image, 
            reference_images=d['candidate images'], 
            batch_size=batch_size
        )
        
        # sim is a list of lists: [[score1, score2, ...]]
        similarity_scores = sim[0]
        
        if plot_results:
            plot(
                similarity=[similarity_scores],
                texts=[f"Query: {os.path.basename(query_image)}"],
                images=d['candidate images'],
                export_file=pj(output_dir, "visualization", language, experiment_name, f'similarity.{n}.png')
            )
        
        # Sort by similarity score (descending)
        tmp = sorted(zip(similarity_scores, d['candidate images']), key=lambda x: x[0], reverse=True)
        result.append({
            'language': language,
            'data': n,
            'candidate': [os.path.basename(i[1]) for i in tmp],
            'relevance': sorted(similarity_scores, reverse=True),
            'query_image': os.path.basename(query_image),
            'target_phrase': d['target phrase'],
            'target_word': d['target word']
        })

    # Save results
    df = pd.DataFrame(result)
    path = pj(output_dir, experiment_name)
    os.makedirs(path, exist_ok=True)
    
    # Save prediction file
    pred_file = pj(path, f'prediction.{language}.txt')
    with open(pred_file, 'w') as f:
        f.write('\n'.join(['\t'.join(x) for x in df.sort_values(by=['data'])['candidate'].to_list()]))
    
    # Save full results
    full_res_file = pj(path, f'full_result.{language}.csv')
    df.to_csv(full_res_file, index=False)
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Results saved to: {path}")
    logging.info(f"Prediction file: {pred_file}")
    logging.info(f"Full results: {full_res_file}")
    logging.info(f"{'='*60}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Solve V-WSD using generated images")
    parser.add_argument('-d', '--data-dir', help='directory of dataset', default='dataset', type=str)
    parser.add_argument('-l', '--language', help='language', default='en', type=str)
    parser.add_argument('-m', '--model-clip', help='clip model', default=None, type=str)
    parser.add_argument('-o', '--output-dir', help='output directory', default="result", type=str)
    parser.add_argument('-i', '--image-dir', help='directory containing generated images', type=str, required=True)
    parser.add_argument('-b', '--batch-size', help='batch size', default=None, type=int)
    parser.add_argument('--plot', help='generate visualization', action='store_true')
    parser.add_argument('--experiment-name', help='name for this experiment', default='generated_images', type=str)
    parser.add_argument('--image-pattern', help='pattern to match generated images', default='*.jpg', type=str)
    
    opt = parser.parse_args()
    
    run_evaluation(
        image_dir=opt.image_dir,
        output_dir=opt.output_dir,
        data_dir=opt.data_dir,
        language=opt.language,
        model_clip=opt.model_clip,
        batch_size=opt.batch_size,
        plot_results=opt.plot,
        experiment_name=opt.experiment_name,
        image_pattern=opt.image_pattern
    )
