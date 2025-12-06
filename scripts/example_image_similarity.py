"""
Image-Image Similarity Calculation Example
Demonstrates how to use the modified CLIP model to calculate similarity between images
"""
import logging
from vwsd import CLIP

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def example_single_query():
    """Example 1: Similarity between a single query image and multiple candidate images"""
    print("\n=== Example 1: Single Query Image vs Multiple Candidate Images ===")
    
    # Load model
    clip = CLIP('laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
    
    # Define image paths
    query_image = "image/my_generated_image.jpg"  # Your query image
    candidate_images = [
        "dataset/image/test_images_resized/image1.jpg",
        "dataset/image/test_images_resized/image2.jpg",
        "dataset/image/test_images_resized/image3.jpg",
    ]
    
    # Calculate similarity
    similarity = clip.get_image_similarity(query_image, candidate_images)
    
    # Output results
    print(f"\nQuery Image: {query_image}")
    print(f"Number of Candidate Images: {len(candidate_images)}")
    print("\nSimilarity Scores (Higher is more similar):")
    for i, (cand, score) in enumerate(zip(candidate_images, similarity[0])):
        print(f"  {i+1}. {cand}: {score:.2f}")
    
    # Find the most similar image
    best_idx = similarity[0].index(max(similarity[0]))
    print(f"\nMost Similar Image: {candidate_images[best_idx]} (Score: {similarity[0][best_idx]:.2f})")


def example_multiple_queries():
    """Example 2: Similarity between multiple query images and multiple candidate images"""
    print("\n=== Example 2: Multiple Query Images vs Multiple Candidate Images ===")
    
    clip = CLIP('laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
    
    # Multiple query images
    query_images = [
        "image/query1.jpg",
        "image/query2.jpg"
    ]
    
    # Multiple candidate images
    candidate_images = [
        "dataset/image/test_images_resized/image1.jpg",
        "dataset/image/test_images_resized/image2.jpg",
        "dataset/image/test_images_resized/image3.jpg",
    ]
    
    # Calculate similarity matrix
    similarity = clip.get_image_similarity(query_images, candidate_images, batch_size=4)
    
    # Output results
    print(f"\nNumber of Query Images: {len(query_images)}")
    print(f"Number of Candidate Images: {len(candidate_images)}")
    print("\nSimilarity Matrix (Row=Query Image, Column=Candidate Image):")
    for i, query in enumerate(query_images):
        print(f"\nQuery Image {i+1}: {query}")
        for j, (cand, score) in enumerate(zip(candidate_images, similarity[i])):
            print(f"  Candidate {j+1}: {score:.2f}")


def example_vwsd_task():
    """Example 3: Using Image-Image Similarity in V-WSD Task"""
    print("\n=== Example 3: V-WSD Task Application ===")
    from vwsd import data_loader
    
    clip = CLIP('laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
    
    # Load dataset
    data = data_loader('dataset')['en']
    
    # Assume you generated an image for each test sample
    # Images named: generated_0.jpg, generated_1.jpg, ...
    
    results = []
    for n, d in enumerate(data[:3]):  # Process only first 3 samples as example
        print(f"\nProcessing Sample {n+1}: {d['target phrase']}")
        
        # Your generated query image
        query_image = f"image/generated_{n}.jpg"
        
        # Candidate images
        candidate_images = d['candidate images']
        
        # Calculate similarity
        similarity = clip.get_image_similarity(query_image, candidate_images)
        
        # Rank candidate images
        ranked = sorted(zip(similarity[0], candidate_images), key=lambda x: x[0], reverse=True)
        
        print(f"Top 3 Most Similar Candidate Images:")
        for i, (score, img) in enumerate(ranked[:3]):
            print(f"  {i+1}. {img}: {score:.2f}")
        
        results.append({
            'sample_id': n,
            'target_phrase': d['target phrase'],
            'best_match': ranked[0][1],
            'score': ranked[0][0]
        })
    
    return results


def example_batch_processing():
    """Example 4: Batch Process Entire V-WSD Dataset"""
    print("\n=== Example 4: Batch Process V-WSD Dataset ===")
    import os
    from vwsd import data_loader
    import pandas as pd
    
    clip = CLIP('laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
    
    # Load dataset
    data = data_loader('dataset')['en']
    
    # Generated images directory
    generated_image_dir = "image"
    
    results = []
    for n, d in enumerate(data):
        # Construct generated image path (adjust according to your naming convention)
        query_image = os.path.join(generated_image_dir, f"generated_{n}.jpg")
        
        # Check if file exists
        if not os.path.exists(query_image):
            print(f"Warning: Image does not exist {query_image}")
            continue
        
        # Calculate similarity
        similarity = clip.get_image_similarity(query_image, d['candidate images'])
        
        # Rank
        ranked = sorted(zip(similarity[0], d['candidate images']), 
                       key=lambda x: x[0], reverse=True)
        
        results.append({
            'sample_id': n,
            'best_match': os.path.basename(ranked[0][1]),
            'score': ranked[0][0],
            'all_candidates': '\t'.join([os.path.basename(img) for _, img in ranked])
        })
        
        if (n + 1) % 50 == 0:
            print(f"Processed {n+1}/{len(data)} samples")
    
    # Save results
    df = pd.DataFrame(results)
    output_dir = "result/image_similarity_experiment"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save prediction results
    with open(os.path.join(output_dir, 'prediction.en.txt'), 'w') as f:
        f.write('\n'.join(df['all_candidates'].tolist()))
    
    # Save full results
    df.to_csv(os.path.join(output_dir, 'full_result.en.csv'), index=False)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Average Similarity Score: {df['score'].mean():.2f}")
    
    return df


if __name__ == '__main__':
    # Run the example you want
    
    # example_single_query()
    # example_multiple_queries()
    # example_vwsd_task()
    # example_batch_processing()
    
    print("\nPlease uncomment the function calls above to run the corresponding examples")
    print("\nUsage:")
    print("1. example_single_query() - Single image query")
    print("2. example_multiple_queries() - Multiple image query")
    print("3. example_vwsd_task() - V-WSD task example")
    print("4. example_batch_processing() - Batch process full dataset")
