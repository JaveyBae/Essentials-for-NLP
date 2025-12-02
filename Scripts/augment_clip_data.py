import google.generativeai as genai
import time
import os
from tqdm import tqdm

# ================= 配置区域 =================
# 请在此处填入您的 Google Gemini API Key
API_KEY = "YOur API Key Here" 

# 输入和输出文件路径
INPUT_FILE = r"f:\刑法大模型\en.test.data.v1.1.txt"
OUTPUT_FILE = r"f:\刑法大模型\en.test.data.v1.1.augmented.txt"

# 批处理大小
BATCH_SIZE = 5

# 配置 Gemini
genai.configure(api_key=API_KEY)
# 使用 gemini-1.5-flash，速度快且稳定
model = genai.GenerativeModel("gemini-2.5-flash")

def generate_descriptions_batch(batch_items):
    """
    批量生成描述
    batch_items: list of (word, phrase) tuples
    """
    if not batch_items:
        return []

    # 构建批量 Prompt
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
            # 按行分割并清理
            lines = [line.strip() for line in response.text.strip().split('\n') if line.strip()]
            return lines
        else:
            return None
    except Exception as e:
        print(f"Batch API Error: {e}")
        return None

def main():
    # 1. 读取原始数据
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到输入文件 {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"开始处理 {len(lines)} 行数据 (Batch Size: {BATCH_SIZE})...")
    
    # 2. 检查断点续传
    processed_count = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            processed_count = len(f.readlines())
        print(f"发现已处理 {processed_count} 行，将从第 {processed_count + 1} 行继续...")

    # 3. 批处理循环
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
        # 从断点处开始，按 BATCH_SIZE 切分
        for i in tqdm(range(processed_count, len(lines), BATCH_SIZE)):
            batch_lines = lines[i : i + BATCH_SIZE]
            batch_items = []
            valid_indices = [] # 记录有效数据的索引
            
            # 解析当前批次的数据
            for idx, line in enumerate(batch_lines):
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    batch_items.append((parts[0], parts[1]))
                    valid_indices.append(idx)
            
            if not batch_items:
                # 如果这一批全是坏数据，直接原样写入
                for line in batch_lines:
                    f_out.write(line)
                continue

            # 调用 API (带重试)
            descriptions = None
            for attempt in range(3):
                descriptions = generate_descriptions_batch(batch_items)
                # 简单的校验：返回行数是否与请求数大致匹配
                if descriptions and len(descriptions) == len(batch_items):
                    break
                elif descriptions and len(descriptions) != len(batch_items):
                    print(f"Warning: Batch mismatch (Expected {len(batch_items)}, got {len(descriptions)}). Retrying...")
                time.sleep(2)
            
            # 写入结果
            desc_idx = 0
            for idx, line in enumerate(batch_lines):
                if idx in valid_indices:
                    # 这是一个有效行，尝试获取描述
                    parts = line.strip().split('\t')
                    word = parts[0]
                    phrase = parts[1]
                    images = parts[2:]
                    
                    if descriptions and desc_idx < len(descriptions):
                        desc = descriptions[desc_idx]
                        # 简单的清洗，防止模型输出 "1. Description" 这种格式
                        if desc[0].isdigit() and desc[1:3] in ['. ', ') ']:
                             desc = desc.split(' ', 1)[1]
                    else:
                        # 失败降级
                        desc = phrase
                    
                    desc_idx += 1
                    
                    # 格式化输出
                    new_col1 = f"{word}, {desc}"
                    new_col2 = f"{phrase}, {desc}"
                    new_line_parts = [new_col1, new_col2] + images
                    f_out.write("\t".join(new_line_parts) + "\n")
                else:
                    # 无效行原样写入
                    f_out.write(line)
            
            f_out.flush()
            time.sleep(1.5) # 批次间稍微等待

    print(f"\n处理完成！结果已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
