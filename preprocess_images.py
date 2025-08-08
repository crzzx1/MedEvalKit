import os
import argparse
from PIL import Image
from glob import glob
from tqdm import tqdm

def preprocess_dataset_images(dataset_path):
    """
    遍历指定路径及其所有子文件夹，将非RGBA格式的图片转换为RGBA格式并覆盖保存。
    """
    # 检查路径是否存在
    if not os.path.isdir(dataset_path):
        print(f"错误：路径 '{dataset_path}' 不存在或不是一个文件夹。")
        return

    # 定义要查找的图片扩展名（包括大小写）
    extensions = [
        '*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp',
        '*.PNG', '*.JPG', '*.JPEG', '*.GIF', '*.BMP'
    ]
    
    image_files = []
    print(f"正在路径 '{dataset_path}' 及其子文件夹中递归搜索图片...")
    
    # 使用递归方式查找所有图片文件
    for ext in extensions:
        # os.path.join(dataset_path, '**', ext) 表示在所有子目录中查找
        # recursive=True 是启用递归搜索的关键
        image_files.extend(glob(os.path.join(dataset_path, '**', ext), recursive=True))

    # 去重，防止因大小写等原因重复添加
    image_files = sorted(list(set(image_files)))

    if not image_files:
        print("未找到任何图片文件。请检查路径和图片格式。")
        return

    print(f"找到了 {len(image_files)} 张图片，开始预处理...")

    for img_path in tqdm(image_files, desc="处理进度"):
        try:
            with Image.open(img_path) as img:
                # 如果图像模式不是 RGBA，则进行转换
                if img.mode != 'RGBA':
                    # 转换为 RGBA 并保存，覆盖原文件
                    img.convert('RGBA').save(img_path)
        except Exception as e:
            # 打印错误信息，并继续处理下一张图片
            print(f"\n处理图片 {img_path} 时出错: {e}")

    print("预处理完成！")

if __name__ == "__main__":
    # --- 使用命令行参数，使脚本更灵活 ---
    # 1. 创建参数解析器
    parser = argparse.ArgumentParser(
        description="一个用于将数据集中图片预处理为RGBA格式的工具。"
    )
    # 2. 添加 --path 参数，用于指定图片文件夹路径
    parser.add_argument(
        '--path', 
        type=str, 
        required=True, 
        help="包含图片的数据集文件夹的路径。"
    )
    # 3. 解析命令行传入的参数
    args = parser.parse_args()
    
    # 4. 调用主函数，传入从命令行获取的路径
    preprocess_dataset_images(args.path)