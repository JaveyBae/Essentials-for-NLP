"""
Image Generation using Google Gemini API
This script generates images from text prompts using the Gemini Imagen API.
"""

import google.generativeai as genai
import os
from PIL import Image
from io import BytesIO
import time
from pathlib import Path

# ================= Configuration =================
# Configure your Google Gemini API Key here
API_KEY = "YOUR_API_KEY_HERE"

# Output directory for generated images
OUTPUT_DIR = "./generated_images"

# Configure Gemini API
genai.configure(api_key=API_KEY)


def generate_image_from_text(prompt: str, output_filename: str = None, num_images: int = 1):
    """
    Generate images from text prompt using Gemini Imagen API
    
    Args:
        prompt (str): Text description for image generation
        output_filename (str): Optional filename for saving the image
        num_images (int): Number of images to generate (default: 1)
    
    Returns:
        list: List of generated image paths
    """
    try:
        # Create output directory if it doesn't exist
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        
        # Initialize the Imagen model
        model = genai.ImageGenerationModel("imagen-3.0-generate-001")
        
        print(f"Generating {num_images} image(s) for prompt: '{prompt}'")
        
        # Generate images
        response = model.generate_images(
            prompt=prompt,
            number_of_images=num_images,
            aspect_ratio="1:1",  # Options: "1:1", "16:9", "9:16", "4:3", "3:4"
            safety_filter_level="block_some",
            person_generation="allow_adult"
        )
        
        saved_paths = []
        
        # Save generated images
        for i, image in enumerate(response.images):
            # Generate filename
            if output_filename:
                if num_images > 1:
                    filename = f"{output_filename}_{i+1}.png"
                else:
                    filename = f"{output_filename}.png"
            else:
                timestamp = int(time.time())
                filename = f"generated_{timestamp}_{i+1}.png"
            
            output_path = os.path.join(OUTPUT_DIR, filename)
            
            # Save image
            image._pil_image.save(output_path)
            saved_paths.append(output_path)
            print(f"✓ Image saved: {output_path}")
        
        return saved_paths
        
    except Exception as e:
        print(f"✗ Error generating image: {str(e)}")
        return []


def batch_generate_images(prompts: list, output_prefix: str = "batch"):
    """
    Generate multiple images from a list of text prompts
    
    Args:
        prompts (list): List of text descriptions
        output_prefix (str): Prefix for output filenames
    
    Returns:
        dict: Dictionary mapping prompts to their generated image paths
    """
    results = {}
    
    for idx, prompt in enumerate(prompts):
        print(f"\n[{idx+1}/{len(prompts)}] Processing prompt...")
        filename = f"{output_prefix}_{idx+1}"
        paths = generate_image_from_text(prompt, filename)
        results[prompt] = paths
        
        # Add delay to avoid rate limiting
        if idx < len(prompts) - 1:
            time.sleep(2)
    
    return results


def main():
    """
    Main function with example usage
    """
    print("=" * 60)
    print("Gemini Image Generation Tool")
    print("=" * 60)
    
    # Check if API key is configured
    if API_KEY == "YOUR_API_KEY_HERE":
        print("\n⚠ Warning: Please configure your Gemini API Key in the script!")
        print("Set the API_KEY variable at the top of the file.\n")
        return
    
    # Example 1: Single image generation
    print("\n--- Example 1: Single Image Generation ---")
    prompt = "A serene mountain landscape at sunset with a crystal clear lake reflecting the sky"
    generate_image_from_text(prompt, "mountain_sunset")
    
    # Example 2: Multiple images from one prompt
    print("\n--- Example 2: Multiple Images from One Prompt ---")
    prompt = "A futuristic city with flying cars and neon lights"
    generate_image_from_text(prompt, "futuristic_city", num_images=3)
    
    # Example 3: Batch generation
    print("\n--- Example 3: Batch Generation ---")
    prompts = [
        "A cute cat sitting on a windowsill looking outside",
        "A steaming cup of coffee on a wooden table with morning light",
        "An astronaut floating in space with Earth in the background"
    ]
    batch_generate_images(prompts, "batch")
    
    print("\n" + "=" * 60)
    print("Image generation completed!")
    print(f"Check the '{OUTPUT_DIR}' directory for generated images.")
    print("=" * 60)


if __name__ == "__main__":
    # Interactive mode: uncomment the following lines for custom input
    # user_prompt = input("Enter your image prompt: ")
    # generate_image_from_text(user_prompt, "custom_image")
    
    # Run examples
    main()
