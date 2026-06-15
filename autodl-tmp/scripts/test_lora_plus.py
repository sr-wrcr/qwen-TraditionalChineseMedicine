#!/usr/bin/env python3
"""
test_lora_from_jsonl.py - 从JSONL文件中提取图片进行批量测试
"""

import os
import sys
import json
import torch
import random
from pathlib import Path
from PIL import Image
from datetime import datetime

# 设置临时目录
os.makedirs("/root/autodl-tmp/temp", exist_ok=True)
os.environ['TMPDIR'] = "/root/autodl-tmp/temp"

print("🧪 ========== 从JSONL文件提取图片测试LoRA模型 ==========")

# 导入
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

# 路径配置
ROOT_DIR = Path("/root/autodl-tmp")
MODEL_DIR = ROOT_DIR / "Qwen" / "Qwen2-VL-2B-Instruct"
LORA_DIR = ROOT_DIR / "qwen2vl_tcm_lora"
DATA_DIR = ROOT_DIR / "data"
OUTPUT_FILE = ROOT_DIR / "batch_test_result.txt"


def load_model_with_lora():
    """加载带LoRA权重的模型"""
    print("\n🔧 加载模型和处理器...")

    # 检查LoRA目录
    if not LORA_DIR.exists():
        print(f"❌ LoRA权重目录不存在: {LORA_DIR}")
        print("   请确保已经训练并保存了LoRA权重")
        sys.exit(1)

    # 加载处理器
    processor = AutoProcessor.from_pretrained(
        str(MODEL_DIR),
        trust_remote_code=True
    )

    # 加载基础模型
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # 检查LoRA文件
    lora_files = list(LORA_DIR.glob("*"))
    print(f"📂 LoRA目录中的文件: {[f.name for f in lora_files]}")

    # 加载LoRA权重
    try:
        print(f"📂 加载LoRA权重: {LORA_DIR}")
        model = PeftModel.from_pretrained(model, str(LORA_DIR))
        print("✅ LoRA权重加载成功")
    except Exception as e:
        print(f"❌ 加载LoRA失败: {e}")
        print("⚠️  使用基础模型（未微调）")

    model.eval()
    print(f"📊 模型设备: {model.device}")

    return model, processor


def extract_images_from_jsonl(jsonl_file, num_samples=6):
    """从JSONL文件中提取测试图片"""
    images = []

    print(f"\n📂 从 {jsonl_file.name} 提取图片...")

    # 读取JSONL文件
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"📊 文件中有 {len(lines)} 条记录")

    # 随机选择指定数量的样本
    if len(lines) > num_samples:
        selected_lines = random.sample(lines, num_samples)
    else:
        selected_lines = lines

    for i, line in enumerate(selected_lines):
        try:
            data = json.loads(line.strip())
            conversation = data["conversations"][0]["value"]

            # 提取图片路径
            if '<|vision_start|>' in conversation:
                img_path = conversation.split('<|vision_start|>')[1].split('<|vision_end|>')[0]

                # 提取图片类型（从ID或路径判断）
                data_id = data.get("id", "")
                if "hand" in data_id or "手" in img_path:
                    img_type = "手相"
                elif "face" in data_id or "面" in img_path:
                    img_type = "面相"
                elif "tongue" in data_id or "舌" in img_path:
                    img_type = "舌诊"
                else:
                    img_type = "中医图片"

                images.append({
                    "type": img_type,
                    "path": Path(img_path),
                    "id": data_id,
                    "index": i + 1
                })
                print(f"  {i + 1}. {img_type}: {Path(img_path).name}")

        except Exception as e:
            print(f"  解析第{i + 1}行失败: {e}")
            continue

    return images


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

        return img
    except Exception as e:
        print(f"❌ 加载图片失败 {image_path}: {e}")
        return Image.new('RGB', (224, 224), color='white')


def analyze_image(model, processor, image_path, image_type):
    """分析单张图片"""
    print(f"\n📷 分析{image_type}图片: {Path(image_path).name}")

    # 加载并缩小图片
    image = resize_image_to_max(image_path, max_dimension=448)

    # 标准提示词
    prompt = f"""请分析这张中医{image_type}图片，给出详细的体质分析和健康建议，需要包括以下7个方面：
1. 体质类型
2. 健康风险
3. 饮食建议
4. 运动建议
5. 穿衣建议
6. 外疗建议
7. 作息建议

请用中文回答，内容要详细专业。"""

    # 构建消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    try:
        # 处理视觉信息
        image_inputs, video_inputs = process_vision_info(messages)

        # 应用聊天模板
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 准备模型输入
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        print(f"  正在生成分析...")
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        # 解码回复
        generated_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        # 提取模型回复部分
        if "assistant" in generated_text:
            response = generated_text.split("assistant\n")[-1].strip()
        else:
            response = generated_text

        return response, True

    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return f"分析失败: {str(e)}", False


def check_response_completeness(response, image_type):
    """检查回答的完整性"""
    section_keywords = {
        "体质类型": ["体质", "体质类型", "体风", "一、", "1."],
        "健康风险": ["风险", "健康风险", "风险提示", "二、", "2."],
        "饮食建议": ["饮食", "饮食建议", "膳食", "三、", "3."],
        "运动建议": ["运动", "运动建议", "锻炼", "活动", "四、", "4."],
        "穿衣建议": ["穿衣", "穿衣建议", "衣着", "服装", "五、", "5."],
        "外疗建议": ["外疗", "外疗建议", "按摩", "刮痧", "拔罐", "艾灸", "六、", "6."],
        "作息建议": ["作息", "作息建议", "睡眠", "休息", "起居", "作息时间", "七、", "7."]
    }

    found_sections = []
    missing_sections = []

    for section, keywords in section_keywords.items():
        found = any(keyword in response for keyword in keywords)
        if found:
            found_sections.append(section)
        else:
            missing_sections.append(section)

    return found_sections, missing_sections


def save_results(all_results):
    """保存所有结果到文件"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(f"{'=' * 60}\n")
        f.write(f"🧬 中医图片批量分析报告\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"📅 分析时间: {timestamp}\n")
        f.write(f"📁 模型配置: Qwen2-VL-2B-Instruct + LoRA微调\n")
        f.write(f"📂 LoRA路径: {LORA_DIR}\n")
        f.write(f"📊 测试图片: {len(all_results)} 张\n")
        f.write(f"📄 数据源: {DATA_DIR / 'tcm_test_data.jsonl'}\n")
        f.write(f"\n{'=' * 60}\n\n")

        total_success = sum(1 for r in all_results if r["success"])
        f.write(f"📈 总体统计:\n")
        f.write(f"  • 成功分析: {total_success}/{len(all_results)}\n")
        f.write(f"  • 成功率: {total_success / len(all_results) * 100:.1f}%\n\n")

        # 按类型统计
        type_stats = {}
        for result in all_results:
            if result['success']:
                img_type = result['type']
                type_stats[img_type] = type_stats.get(img_type, 0) + 1

        if type_stats:
            f.write(f"📊 按类型统计:\n")
            for img_type, count in type_stats.items():
                f.write(f"  • {img_type}: {count} 张\n")
            f.write(f"\n")

        # 写入每个结果
        for i, result in enumerate(all_results, 1):
            f.write(f"{'=' * 60}\n")
            f.write(f"📷 测试 {i}/{len(all_results)}\n")
            f.write(f"{'=' * 60}\n\n")

            f.write(f"📋 图片信息:\n")
            f.write(f"  • 类型: {result['type']}\n")
            f.write(f"  • ID: {result['id']}\n")
            f.write(f"  • 文件名: {result['filename']}\n")
            f.write(f"  • 路径: {result['image_path']}\n")
            f.write(f"  • 状态: {'✅ 成功' if result['success'] else '❌ 失败'}\n\n")

            if result['success']:
                # 完整性检查
                found, missing = check_response_completeness(result['response'], result['type'])
                f.write(f"📊 完整性检查 ({len(found)}/7):\n")
                f.write(f"  • 包含的方面: {', '.join(found) if found else '无'}\n")
                if missing:
                    f.write(f"  • 缺失的方面: {', '.join(missing)}\n")
                f.write(f"\n")

                # 回答内容
                f.write(f"📝 模型回答 ({len(result['response'])} 字符):\n")
                f.write(f"{'-' * 40}\n")
                f.write(f"{result['response']}\n")
                f.write(f"{'-' * 40}\n\n")
            else:
                f.write(f"❌ 错误信息:\n")
                f.write(f"{result['response']}\n\n")

            f.write(f"\n")

        f.write(f"{'=' * 60}\n")
        f.write(f"✅ 批量测试完成\n")
        f.write(f"{'=' * 60}\n")

    print(f"\n💾 结果已保存到: {OUTPUT_FILE}")

    # 显示摘要
    print(f"\n📊 测试摘要:")
    print(f"  • 总测试图片: {len(all_results)} 张")
    print(f"  • 成功分析: {total_success} 张")
    print(f"  • 成功率: {total_success / len(all_results) * 100:.1f}%")


def main():
    """主函数"""
    print(f"\n🎯 目标: 从JSONL文件提取6张图片测试LoRA模型")
    print(f"{'=' * 60}")

    # 1. 检查测试数据文件
    test_jsonl = DATA_DIR / "tcm_test_data.jsonl"
    if not test_jsonl.exists():
        print(f"❌ 测试数据文件不存在: {test_jsonl}")
        print("请先运行 convert_tcm_data.py 生成测试数据")
        return

    # 2. 加载模型
    model, processor = load_model_with_lora()

    # 3. 从JSONL提取图片
    test_images = extract_images_from_jsonl(test_jsonl, num_samples=6)

    if not test_images:
        print("❌ 未能从JSONL文件中提取测试图片")
        return

    # 4. 分析每张图片
    all_results = []

    print(f"\n🔍 开始分析图片...")
    for i, img_info in enumerate(test_images, 1):
        print(f"\n[{i}/{len(test_images)}] 分析 {img_info['type']}: {img_info['path'].name}")

        # 检查图片文件是否存在
        if not img_info['path'].exists():
            print(f"  ⚠️  图片文件不存在: {img_info['path']}")
            all_results.append({
                "type": img_info['type'],
                "id": img_info['id'],
                "filename": img_info['path'].name,
                "image_path": str(img_info['path']),
                "response": "图片文件不存在",
                "success": False,
                "length": 0
            })
            continue

        response, success = analyze_image(
            model, processor,
            img_info["path"],
            img_info["type"]
        )

        all_results.append({
            "type": img_info['type'],
            "id": img_info['id'],
            "filename": img_info['path'].name,
            "image_path": str(img_info['path']),
            "response": response,
            "success": success,
            "length": len(response) if success else 0
        })

        # 显示摘要
        if success:
            found, missing = check_response_completeness(response, img_info['type'])
            print(f"  ✅ 分析完成 ({len(response)} 字符, 包含{len(found)}/7个方面)")
        else:
            print(f"  ❌ 分析失败")

    # 5. 保存结果
    save_results(all_results)

    print(f"\n{'=' * 60}")
    print(f"🎉 批量测试完成!")
    print(f"📁 详细结果保存在: {OUTPUT_FILE}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()