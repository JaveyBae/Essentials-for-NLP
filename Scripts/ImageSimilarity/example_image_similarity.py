"""
图像-图像相似度计算示例
演示如何使用修改后的 CLIP 模型计算图像之间的相似度
"""
import logging
from vwsd import CLIP

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def example_single_query():
    """示例1: 单张查询图像与多张候选图像的相似度"""
    print("\n=== 示例1: 单张查询图像 vs 多张候选图像 ===")
    
    # 加载模型
    clip = CLIP('laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
    
    # 定义图像路径
    query_image = "image/my_generated_image.jpg"  # 你的查询图像
    candidate_images = [
        "dataset/image/test_images_resized/image1.jpg",
        "dataset/image/test_images_resized/image2.jpg",
        "dataset/image/test_images_resized/image3.jpg",
    ]
    
    # 计算相似度
    similarity = clip.get_image_similarity(query_image, candidate_images)
    
    # 输出结果
    print(f"\n查询图像: {query_image}")
    print(f"候选图像数量: {len(candidate_images)}")
    print("\n相似度分数 (越高越相似):")
    for i, (cand, score) in enumerate(zip(candidate_images, similarity[0])):
        print(f"  {i+1}. {cand}: {score:.2f}")
    
    # 找到最相似的图像
    best_idx = similarity[0].index(max(similarity[0]))
    print(f"\n最相似的图像: {candidate_images[best_idx]} (分数: {similarity[0][best_idx]:.2f})")


def example_multiple_queries():
    """示例2: 多张查询图像与多张候选图像的相似度"""
    print("\n=== 示例2: 多张查询图像 vs 多张候选图像 ===")
    
    clip = CLIP('laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
    
    # 多张查询图像
    query_images = [
        "image/query1.jpg",
        "image/query2.jpg"
    ]
    
    # 多张候选图像
    candidate_images = [
        "dataset/image/test_images_resized/image1.jpg",
        "dataset/image/test_images_resized/image2.jpg",
        "dataset/image/test_images_resized/image3.jpg",
    ]
    
    # 计算相似度矩阵
    similarity = clip.get_image_similarity(query_images, candidate_images, batch_size=4)
    
    # 输出结果
    print(f"\n查询图像数量: {len(query_images)}")
    print(f"候选图像数量: {len(candidate_images)}")
    print("\n相似度矩阵 (行=查询图像, 列=候选图像):")
    for i, query in enumerate(query_images):
        print(f"\n查询图像 {i+1}: {query}")
        for j, (cand, score) in enumerate(zip(candidate_images, similarity[i])):
            print(f"  候选 {j+1}: {score:.2f}")


def example_vwsd_task():
    """示例3: 在 V-WSD 任务中使用图像-图像相似度"""
    print("\n=== 示例3: V-WSD 任务应用 ===")
    from vwsd import data_loader
    
    clip = CLIP('laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
    
    # 加载数据集
    data = data_loader('dataset')['en']
    
    # 假设你为每个测试样本生成了一张图像
    # 图像命名为: generated_0.jpg, generated_1.jpg, ...
    
    results = []
    for n, d in enumerate(data[:3]):  # 只处理前3个样本作为示例
        print(f"\n处理样本 {n+1}: {d['target phrase']}")
        
        # 你生成的查询图像
        query_image = f"image/generated_{n}.jpg"
        
        # 候选图像
        candidate_images = d['candidate images']
        
        # 计算相似度
        similarity = clip.get_image_similarity(query_image, candidate_images)
        
        # 排序候选图像
        ranked = sorted(zip(similarity[0], candidate_images), key=lambda x: x[0], reverse=True)
        
        print(f"Top 3 最相似的候选图像:")
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
    """示例4: 批量处理整个 V-WSD 数据集"""
    print("\n=== 示例4: 批量处理 V-WSD 数据集 ===")
    import os
    from vwsd import data_loader
    import pandas as pd
    
    clip = CLIP('laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
    
    # 加载数据集
    data = data_loader('dataset')['en']
    
    # 生成的图像文件夹
    generated_image_dir = "image"
    
    results = []
    for n, d in enumerate(data):
        # 构造生成图像路径 (根据你的命名方式调整)
        query_image = os.path.join(generated_image_dir, f"generated_{n}.jpg")
        
        # 检查文件是否存在
        if not os.path.exists(query_image):
            print(f"警告: 图像不存在 {query_image}")
            continue
        
        # 计算相似度
        similarity = clip.get_image_similarity(query_image, d['candidate images'])
        
        # 排序
        ranked = sorted(zip(similarity[0], d['candidate images']), 
                       key=lambda x: x[0], reverse=True)
        
        results.append({
            'sample_id': n,
            'best_match': os.path.basename(ranked[0][1]),
            'score': ranked[0][0],
            'all_candidates': '\t'.join([os.path.basename(img) for _, img in ranked])
        })
        
        if (n + 1) % 50 == 0:
            print(f"已处理 {n+1}/{len(data)} 个样本")
    
    # 保存结果
    df = pd.DataFrame(results)
    output_dir = "result/image_similarity_experiment"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存预测结果
    with open(os.path.join(output_dir, 'prediction.en.txt'), 'w') as f:
        f.write('\n'.join(df['all_candidates'].tolist()))
    
    # 保存完整结果
    df.to_csv(os.path.join(output_dir, 'full_result.en.csv'), index=False)
    
    print(f"\n结果已保存到: {output_dir}")
    print(f"平均相似度分数: {df['score'].mean():.2f}")
    
    return df


if __name__ == '__main__':
    # 运行你想要的示例
    
    # example_single_query()
    # example_multiple_queries()
    # example_vwsd_task()
    # example_batch_processing()
    
    print("\n请取消注释上面的函数调用来运行对应的示例")
    print("\n使用方法:")
    print("1. example_single_query() - 单张图像查询")
    print("2. example_multiple_queries() - 多张图像查询")
    print("3. example_vwsd_task() - V-WSD 任务示例")
    print("4. example_batch_processing() - 批量处理完整数据集")
