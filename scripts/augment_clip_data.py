import google.generativeai as genai
import time
import os
from tqdm import tqdm

# ================= Configuration Area =================
# Please enter your Google Gemini API Key here
API_KEY = "YOur API Key Here" 

# Input and Output File Paths
INPUT_FILE = r"f:\刑法大模型\en.test.data.v1.1.txt"
OUTPUT_FILE = r"f:\刑法大模型\en.test.data.v1.1.augmented.txt"

# Batch Size
BATCH_SIZE = 5

# Configure Gemini
genai.configure(api_key=API_KEY)
# Use gemini-1.5-flash, fast and stable
model = genai.GenerativeModel("gemini-2.5-flash")

def generate_descriptions_batch(batch_items):
    """
    Batch generate descriptions
    batch_items: list of (word, phrase) tuples
    """
    if not batch_items:
        return []

    # Build Batch Prompt
    items_str = ""
    for i, (word, phrase) in enumerate(batch_items):
        items_str += f"{i+1}. Word: {word}, Phrase: {phrase}\n"

    prompt = f"""
    Task: Generate a concise, visual description (10-20 words) for each of the following concepts.
    Context: These descriptions will be used for CLIP image retrieval to distinguish specific meanings.
    Format: Return ONLY the descriptions, one per line, corresponding strictly to the numbered items. Do not include the numbers or prefixes in the output.
    
    Items:
    {items_str}
    
    Descriptions:
    """
    
    try:
        response = model.generate_content(prompt)
        if response.text:
            # Split by line and clean
            lines = [line.strip() for line in response.text.strip().split('\n') if line.strip()]
            return lines
        else:
            return None
    except Exception as e:
        print(f"Batch API Error: {e}")
        return None

def main():
    # 1. Read original data
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Starting to process {len(lines)} lines (Batch Size: {BATCH_SIZE})...")
    
    # 2. Check for resume capability
    processed_count = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            processed_count = len(f.readlines())
        print(f"Found {processed_count} lines already processed, continuing from line {processed_count + 1}...")

    # 3. Batch processing loop
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
        # Start from breakpoint, slice by BATCH_SIZE
        for i in tqdm(range(processed_count, len(lines), BATCH_SIZE)):
            batch_lines = lines[i : i + BATCH_SIZE]
            batch_items = []
            valid_indices = [] # Record indices of valid data
            
            # Parse current batch data
            for idx, line in enumerate(batch_lines):
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    batch_items.append((parts[0], parts[1]))
                    valid_indices.append(idx)
            
            if not batch_items:
                # If this batch is all bad data, write as is
                for line in batch_lines:
                    f_out.write(line)
                continue

            # Call API (with retry)
            descriptions = None
            for attempt in range(3):
                descriptions = generate_descriptions_batch(batch_items)
                # Simple validation: check if returned lines match request count roughly
                if descriptions and len(descriptions) == len(batch_items):
                    break
                elif descriptions and len(descriptions) != len(batch_items):
                    print(f"Warning: Batch mismatch (Expected {len(batch_items)}, got {len(descriptions)}). Retrying...")
                time.sleep(2)
            
            # Write results
            desc_idx = 0
            for idx, line in enumerate(batch_lines):
                if idx in valid_indices:
                    # This is a valid line, try to get description
                    parts = line.strip().split('\t')
                    word = parts[0]
                    phrase = parts[1]
                    images = parts[2:]
                    
                    if descriptions and desc_idx < len(descriptions):
                        desc = descriptions[desc_idx]
                        # Simple cleaning, prevent model from outputting "1. Description" format
                        if desc[0].isdigit() and desc[1:3] in ['. ', ') ']:
                             desc = desc.split(' ', 1)[1]
                    else:
                        # Fallback on failure
                        desc = phrase
                    
                    desc_idx += 1
                    
                    # Format output
                    new_col1 = f"{word}, {desc}"
                    new_col2 = f"{phrase}, {desc}"
                    new_line_parts = [new_col1, new_col2] + images
                    f_out.write("\t".join(new_line_parts) + "\n")
                else:
                    # Write invalid line as is
                    f_out.write(line)
            
            f_out.flush()
            time.sleep(1.5) # Wait a bit between batches

    print(f"\nProcessing complete! Results saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
