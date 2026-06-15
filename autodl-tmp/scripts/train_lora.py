#!/usr/bin/env python3
"""
train_success_original.py - 回到最初的成功版本
"""

import os
import sys
import torch
import gc
from pathlib import Path
from PIL import Image

# 设置临时目录
os.makedirs("/root/autodl-tmp/temp", exist_ok=True)
os.environ['TMPDIR'] = "/root/autodl-tmp/temp"

print("🚀 ========== 回到最初的成功版本 ==========")

# 导入
import yaml
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, TaskType, get_peft_model

# 检查qwen_vl_utils
try:
    from qwen_vl_utils import process_vision_info
    print("✅ qwen_vl_utils 可用")
except ImportError:
    print("❌ 缺少 qwen_vl_utils")
    sys.exit(1)

# 路径
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "Qwen" / "Qwen2-VL-2B-Instruct"
OUTPUT_DIR = ROOT_DIR / "qwen2vl_tcm_lora"

def load_config():
    """加载训练配置"""
    config_file = ROOT_DIR / "train_config.yaml"
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {
        'train_data': str(DATA_DIR / "tcm_trainval_data.jsonl"),
        'eval_data': str(DATA_DIR / "tcm_val_data.jsonl"),
        'output_dir': str(OUTPUT_DIR),
        'num_train_epochs': 3,
        'lora_rank': 16,
        'max_length': 8192,
    }

def resize_image_to_max(image_path, max_dimension=448):
    """缩小图片到最大尺寸不超过指定值"""
    try:
        img = Image.open(image_path).convert('RGB')
        width, height = img.size

        # 计算缩放比例
        max_current = max(width, height)
        if max_current > max_dimension:
            scale = max_dimension / max_current
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"  缩小: {width}x{height} -> {new_width}x{new_height}")

        return img
    except Exception as e:
        print(f"❌ 加载图片失败 {image_path}: {e}")
        # 返回默认图片
        return Image.new('RGB', (224, 224), color='white')

def process_func_success_with_resize(example, processor, tokenizer, max_length=8192):
    """
    成功版本的预处理函数，添加图片缩小
    """
    conversation = example["conversations"]
    input_content = conversation[0]["value"]
    output_content = conversation[1]["value"]

    # 提取图片路径
    file_path = input_content.split("<|vision_start|>")[1].split("<|vision_end|>")[0]
    print(f"📷 处理图片: {Path(file_path).name}")

    # 缩小图片
    resized_image = resize_image_to_max(file_path, max_dimension=448)

    # 构建消息 - 使用缩小后的Image对象
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": resized_image,  # 使用缩小后的图片
                },
                {"type": "text", "text": "请分析这张中医图片"}
            ]
        }
    ]

    # 使用process_vision_info处理
    image_inputs, video_inputs = process_vision_info(messages)

    # 应用聊天模板
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 处理器输入 - 关键：和成功版本完全一样
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # 转换为list以便拼接 - 和成功版本完全一样
    inputs_dict = {key: value.tolist() for key, value in inputs.items()}

    # 处理输出 - 不截断
    response = tokenizer(
        f"{output_content}",
        add_special_tokens=False
    )

    # 构建input_ids - 和成功版本完全一样
    input_ids = (
            inputs_dict["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    )

    attention_mask = inputs_dict["attention_mask"][0] + response["attention_mask"] + [1]
    labels = (
            [-100] * len(inputs_dict["input_ids"][0])
            + response["input_ids"]
            + [tokenizer.pad_token_id]
    )

    # 截断检查（只警告，不实际截断）
    if len(input_ids) > max_length:
        print(f"⚠️  序列过长: {len(input_ids)} > {max_length}")
        # 可以稍微截断，但保持核心内容
        if len(input_ids) > max_length + 500:
            # 只截断一点，保持大部分内容
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]

    # 转换为tensor - 和成功版本完全一样
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels)

    # 关键：正确处理pixel_values和image_grid_thw - 和成功版本完全一样
    inputs_dict['pixel_values'] = torch.tensor(inputs_dict['pixel_values'])
    inputs_dict['image_grid_thw'] = torch.tensor(inputs_dict['image_grid_thw']).squeeze(0)

    # 检查patch数量
    num_patches = inputs_dict['pixel_values'].shape[0]
    print(f"  ✅ patches: {num_patches}")

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": inputs_dict['pixel_values'],
        "image_grid_thw": inputs_dict['image_grid_thw']
    }

def test_success_version(model, processor, tokenizer):
    """测试成功版本"""
    print("\n🧪 测试成功版本处理...")

    # 使用第一条数据测试
    train_path = DATA_DIR / "tcm_trainval_data.jsonl"
    train_raw = load_dataset('json', data_files=str(train_path), split='train')

    test_result = process_func_success_with_resize(train_raw[0], processor, tokenizer)

    print(f"✅ 测试处理成功")
    for key, value in test_result.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")

    # 测试前向传播
    print("\n🧪 测试模型前向传播...")
    test_inputs = {k: v.unsqueeze(0).to(model.device) if isinstance(v, torch.Tensor) else v
                   for k, v in test_result.items()}

    print("测试输入形状:")
    for key, value in test_inputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")

    try:
        with torch.no_grad():
            outputs = model(**test_inputs)
        print(f"✅ 前向传播成功！logits形状: {outputs.logits.shape}")
        return True
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  使用设备: {device}")

    # 加载配置
    config = load_config()

    # 加载模型和处理器
    print("\n🔧 加载模型和处理器...")
    processor = AutoProcessor.from_pretrained(
        str(MODEL_DIR),
        trust_remote_code=True
    )
    tokenizer = processor.tokenizer

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    model.enable_input_require_grads()
    print("✅ 模型加载成功")

    # 测试成功版本
    if not test_success_version(model, processor, tokenizer):
        print("❌ 测试失败，退出")
        return

    # 加载数据
    print("\n📂 加载数据...")
    train_path = config.get('train_data', str(DATA_DIR / "tcm_train_data.jsonl"))
    eval_path = config.get('eval_data', str(DATA_DIR / "tcm_val_data.jsonl"))

    # 全量加载
    train_raw = load_dataset('json', data_files=train_path, split='train[:2000]')

    if os.path.exists(eval_path):
        eval_raw = load_dataset('json', data_files=eval_path, split='train[200:600]')
    else:
        split = train_raw.train_test_split(test_size=0.1, seed=42)
        train_raw = split['train[:2000]']
        eval_raw = split['test[200:600]']

    print(f"训练数据: {len(train_raw)} 条")
    print(f"验证数据: {len(eval_raw)} 条")

    # 分批处理数据（避免内存问题）
    def process_in_batches(raw_data, name="训练集", batch_size=200):
        """分批处理"""
        print(f"\n🔄 处理{name}...")

        all_processed = []
        num_batches = (len(raw_data) + batch_size - 1) // batch_size

        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(raw_data))

            print(f"  批次 {i + 1}/{num_batches}: {start}-{end}")

            batch = []
            for j in range(start, end):
                try:
                    result = process_func_success_with_resize(raw_data[j], processor, tokenizer)
                    batch.append(result)
                except Exception as e:
                    print(f"    跳过第{j}条: {e}")
                    continue

            all_processed.extend(batch)

            # 清理内存
            gc.collect()
            torch.cuda.empty_cache()

        return Dataset.from_list(all_processed)

    # 处理训练数据
    train_dataset = process_in_batches(train_raw, "训练集", 100)
    eval_dataset = process_in_batches(eval_raw, "验证集", 50)

    print(f"\n📊 处理后数据:")
    print(f"  训练集: {len(train_dataset)} 条")
    print(f"  验证集: {len(eval_dataset)} 条")

    # 配置LoRA
    print("\n🎯 配置LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=config.get('lora_rank', 16),
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 输出目录
    output_dir = config.get('output_dir', str(OUTPUT_DIR))
    os.makedirs(output_dir, exist_ok=True)

    # 训练参数 - 简化版
    training_args = TrainingArguments(
        # output_dir=str(output_dir),
        output_dir=str(OUTPUT_DIR.absolute()),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        eval_strategy="steps",
        save_total_limit=1,
        load_best_model_at_end=False,
        fp16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to="none",
        push_to_hub=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    # 数据收集器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
    )

    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 开始训练
    print(f"\n🔥 开始训练...")
    print(f"  训练集: {len(train_dataset)} 条")
    print(f"  输出: {output_dir}")

    try:
        trainer.train()
        print("\n✅ 训练完成！")

        # 保存模型
        print("\n💾 保存模型...")
        trainer.save_model()
        processor.save_pretrained(output_dir)

        print(f"\n🎉 ========== 训练完成！ ==========")
        print(f"📁 模型保存在: {output_dir}")

    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()