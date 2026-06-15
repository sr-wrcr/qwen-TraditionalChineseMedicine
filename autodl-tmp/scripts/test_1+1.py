#!/usr/bin/env python3
"""
正确测试Qwen2VL模型
"""

from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch
import numpy as np

def test_qwen2vl():
    print("🧪 开始测试Qwen2VL模型...")
    
    try:
        # 加载模型和处理器
        model_id = "../Qwen/Qwen2-VL-2B-Instruct"
        print(f"加载模型: {model_id}")
        
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("✅ 模型加载成功！")
        
        # 测试1: 纯文本数学问题
        print("\n📝 测试1: 纯文本数学问题")
        question = "1+1等于多少？"
        print(f"提问: {question}")
        
        # 创建对话
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        # 准备输入
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 由于是纯文本，没有图像
        inputs = processor(
            text=[text],
            images=None,
            padding=True,
            return_tensors="pt"
        ).to(model.device)
        
        # 生成回答
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False
        )
        
        # 解码回答
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"🤖 模型回答: {response}")
        
        # 测试2: 创建简单图像测试
        print("\n🖼️  测试2: 创建简单图像测试")
        
        # 创建一个红色正方形图像
        img_array = np.zeros((100, 100, 3), dtype=np.uint8)
        img_array[25:75, 25:75] = [255, 0, 0]  # 红色
        image = Image.fromarray(img_array)
        image.save("test_red_square.jpg")
        
        # 图像问题
        messages_with_image = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "请描述这张图片中的颜色"}
                ]
            }
        ]
        
        text_with_image = processor.apply_chat_template(
            messages_with_image,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs_with_image = processor(
            text=[text_with_image],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(model.device)
        
        generated_ids_image = model.generate(
            **inputs_with_image,
            max_new_tokens=100,
            do_sample=False
        )
        
        generated_ids_image = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs_with_image.input_ids, generated_ids_image)
        ]
        
        response_image = processor.batch_decode(generated_ids_image, skip_special_tokens=True)[0]
        print(f"🤖 视觉回答: {response_image}")
        
        print("\n🎉 测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_qwen2vl()
