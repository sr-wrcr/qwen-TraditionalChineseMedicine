import json
import pandas as pd
import os
import sys
from pathlib import Path

# 设置路径
BASE_DIR = Path(__file__).parent.parent  # scripts目录的上一级
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "data"

print(f"📁 基础目录: {BASE_DIR}")
print(f"📁 数据目录: {DATA_DIR}")
print(f"📁 输出目录: {OUTPUT_DIR}")


def convert_tcm_dataset():
    """
    转换中医数据集，按train/val/test分别保存
    """
    # 存储不同划分的数据
    split_data = {
        'train': [],
        'val': [],
        'test': []
    }

    split_counts = {'train': 0, 'val': 0, 'test': 0}

    # 定义数据类型
    data_types = ['face', 'hand', 'tongue']

    print("\n📁 开始扫描数据集...")

    for split in split_data.keys():
        for data_type in data_types:
            data_path = DATA_DIR / split / data_type

            if not data_path.exists():
                print(f"  ⚠️  目录不存在: {data_path}")
                continue

            print(f"  🔍 处理 {split}/{data_type}...")

            # 获取所有图片文件
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.PNG']
            image_files = []
            for ext in image_extensions:
                image_files.extend(list(data_path.glob(f"*{ext}")))

            print(f"    找到 {len(image_files)} 张图片")

            # 对每张图片，查找同名的CSV文件
            for img_file in image_files:
                img_stem = img_file.stem  # 图片文件名（不含扩展名），如：0dfs02-hand001

                # 查找同名CSV
                csv_file = data_path / f"{img_stem}.csv"

                if not csv_file.exists():
                    print(f"    ⚠️  图片 {img_stem} 没有对应的CSV文件")
                    continue

                try:
                    # 读取CSV文件
                    df = pd.read_csv(csv_file, encoding='utf-8-sig')

                    if len(df) != 1:
                        print(f"    ⚠️  CSV {img_stem}.csv 有 {len(df)} 行，期望1行")
                        continue

                    # 获取CSV中的数据
                    row = df.iloc[0]

                    # 设置中文名称
                    tcm_type_map = {
                        'face': '面相',
                        'hand': '手相',
                        'tongue': '舌诊'
                    }
                    tcm_type = tcm_type_map.get(data_type, '图片')

                    # 构建训练数据
                    data_entry = {
                        "id": f"{split}_{data_type}_{img_stem}",
                        "conversations": [
                            {
                                "from": "user",
                                "value": f"<|vision_start|>{img_file}<|vision_end|>请分析这张中医{tcm_type}图片，给出详细的体质分析和健康建议，需要包括：体质类型、健康风险、饮食建议、运动建议、穿衣建议、外疗建议、作息建议这7个方面。"
                            },
                            {
                                "from": "assistant",
                                "value": f"根据中医{tcm_type}分析：\n\n一、体质类型\n{row['体质类型']}\n\n二、健康风险\n{row['健康风险']}\n\n三、饮食建议\n{row['饮食建议']}\n\n四、运动建议\n{row['运动建议']}\n\n五、穿衣建议\n{row['穿衣建议']}\n\n六、外疗建议\n{row['外疗建议']}\n\n七、作息建议\n{row['作息建议']}"                            }
                        ]
                    }

                    split_data[split].append(data_entry)
                    split_counts[split] += 1
                    print(f"    ✓ 成功处理: {img_stem}")

                except Exception as e:
                    print(f"    ✗ 处理 {img_stem} 时出错: {e}")
                    continue

    # 保存数据
    print(f"\n💾 保存转换后的数据...")

    # 确保输出目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 分别保存每个划分
    for split in split_data.keys():
        if split_data[split]:
            output_file = OUTPUT_DIR / f"tcm_{split}_data.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for entry in split_data[split]:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            print(f"  ✅ {split}: {split_counts[split]} 条 -> {output_file}")

    # 创建合并的训练+验证集（用于实际训练）
    train_val_data = split_data['train'] + split_data['val']
    if train_val_data:
        output_file = OUTPUT_DIR / "tcm_trainval_data.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in train_val_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"  ✅ train+val: {len(train_val_data)} 条 -> {output_file}")

    # 创建数据集配置文件
    create_dataset_config(split_counts)

    return split_counts


def create_dataset_config(split_counts):
    """创建数据集配置文件"""
    config = {
        "dataset_name": "TCM_Image_Diagnosis",
        "description": "中医图片诊疗数据集",
        "language": "zh",
        "license": "research",
        "splits": split_counts,
        "features": {
            "id": "string",
            "conversations": [
                {
                    "from": "string",
                    "value": "string"
                }
            ]
        }
    }

    config_file = OUTPUT_DIR / "dataset_info.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"  📄 数据集配置: {config_file}")

    # 创建训练配置
    create_training_config()


def create_training_config():
    """创建训练配置文件"""
    config_content = f"""# Qwen2VL中医训练配置
model_name_or_path: "Qwen/Qwen2-VL-2B-Instruct"
train_dataset_path: "{OUTPUT_DIR}/tcm_trainval_data.jsonl"
eval_dataset_path: "{OUTPUT_DIR}/tcm_val_data.jsonl"
test_dataset_path: "{OUTPUT_DIR}/tcm_test_data.jsonl"
output_dir: "./qwen2vl_tcm_lora"

# 训练参数
num_train_epochs: 5
per_device_train_batch_size: 1
per_device_eval_batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 2e-4
warmup_ratio: 0.03
logging_steps: 10
save_steps: 100
eval_steps: 100
save_total_limit: 2

# LoRA配置
use_lora: true
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.1

# 硬件配置
fp16: true
gradient_checkpointing: true
"""

    config_file = BASE_DIR / "train_config.yaml"
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_content)
    print(f"  ⚙️  训练配置: {config_file}")


def verify_data():
    """验证生成的数据"""
    print(f"\n🔍 验证生成的数据...")

    splits = ['train', 'val', 'test', 'trainval']

    for split in splits:
        data_file = OUTPUT_DIR / f"tcm_{split}_data.jsonl"
        if data_file.exists():
            with open(data_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            print(f"  {split}: {len(lines)} 条记录")

            # 显示第一条记录
            if lines:
                data = json.loads(lines[0])
                print(f"    示例ID: {data['id']}")
                user_value = data['conversations'][0]['value']
                if '<|vision_start|>' in user_value:
                    parts = user_value.split('<|vision_start|>')
                    img_info = parts[1].split('<|vision_end|>')[0]
                    print(f"    图片路径: {Path(img_info).name}")
                    print(f"    对应CSV: {Path(img_info).stem}.csv")


if __name__ == "__main__":
    print("=== 中医数据转换开始 ===")
    counts = convert_tcm_dataset()
    verify_data()

    print(f"\n🎉 转换完成！")
    print(f"📊 统计:")
    for split, count in counts.items():
        print(f"  {split}: {count} 条")

    print(f"\n📁 输出文件位置: {OUTPUT_DIR}")