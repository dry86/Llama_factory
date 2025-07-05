"""
run_meme_pipeline.py
多轮 Qwen2-VL 推理：Q-Generator ➜ VQA ➜ Reasoner
"""
import os, re, json, argparse, time 
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import torch
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

# ----------------------- 0. 统一加载 Qwen2-VL -----------------------
MODEL_ID = "/newdisk/public/wws/01-AIGC-GPRO/LLaMA-Factory/output/qwen2vl_7b/lora_450v2_sft_4e_lr1e-4"  # 本地路径

# 加载处理器用于图像处理和聊天模板
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

def get_devices_from_str(devices_str: str):
    """从字符串解析设备列表
    
    Args:
        devices_str: 设备字符串，例如 "0,1" 或 "1"
        
    Returns:
        设备数量
    """
    if not devices_str:
        # 尝试获取可见的GPU数量
        try:
            import torch
            return torch.cuda.device_count()
        except Exception:
            return 1
    
    # 如果提供了gpus参数，返回GPU数量
    return len(devices_str.split(","))

# 使用vLLM加载模型
def load_vllm_model(model_id, dtype, gpus=None, tp_size=1, gpu_memory_utilization=0.95):
    """加载vLLM模型
    
    Args:
        model_id: 模型ID
        dtype: 数据类型
        gpus: 要使用的GPU设备ID字符串，例如 "0,1" 或 "1"
        tp_size: 张量并行大小
        gpu_memory_utilization: GPU内存使用率
        
    Returns:
        加载的模型
    """
    # 设置CUDA_VISIBLE_DEVICES环境变量
    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        print(f"设置CUDA_VISIBLE_DEVICES = {gpus}")
    
    # 清理GPU缓存
    import torch
    torch.cuda.empty_cache()
    
    kwargs = {
        "model": model_id,
        "dtype": dtype,
        "tensor_parallel_size": tp_size,
        "max_model_len": 32768,
        "limit_mm_per_prompt": {"image": 1, "video": 0},
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": True
    }
    
    print(f"使用参数: tensor_parallel_size={tp_size}, gpu_memory_utilization={gpu_memory_utilization}")
    
    return LLM(**kwargs)

# ----------------------- 1. 工具函数 -----------------------
def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def _generate(model, messages: List[dict], max_new_tokens=512, temperature=0.7) -> str:
    
    # 应用聊天模板
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    image_data, _ = process_vision_info(messages)
    
    # 设置采样参数
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.9
    )
    
    vllm_inputs = {
        "prompt": prompt,
        "multi_modal_data": {
            "image": image_data,
        },
    }

    # 生成响应
    outputs = model.generate(vllm_inputs, sampling_params)
    generated = outputs[0].outputs[0].text
    return generated.strip()

def _generate_text(model, messages: List[dict], max_new_tokens=512, temperature=0.7) -> str:
    
    # 应用聊天模板
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    # image_data, _ = process_vision_info(messages)
    
    # 设置采样参数
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.9
    )
    
    vllm_inputs = {
        "prompt": prompt,
    }

    # 生成响应
    outputs = model.generate(vllm_inputs, sampling_params)
    generated = outputs[0].outputs[0].text
    return generated.strip()

def extract_harm_answer(text: str) -> str | None:
    """
    从文本中提取 hateful/not-hateful 答案
    
    Args:
        text: 包含答案的文本
        
    Returns:
        提取出的答案 (hateful/not-hateful) 或 None
    """
    # 规范化文本，处理可能的大小写和空格问题
    text = text.lower().strip()
    
    # 如果直接是"hateful"或"non-hateful"，直接返回结果
    if text == "hateful":
        return "hateful"
    elif text == "non-hateful":
        return "not-hateful"
    
    # 1. 首先尝试从 <answer> 标签中提取
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.IGNORECASE)
    if answer_match:
        answer = answer_match.group(1).strip().lower()
        if answer in ['hateful', 'not-hateful']:
            return answer
    
    # 2. 尝试从非标准格式中提取
    patterns = [
        r'答案是\s*(hateful|not-hateful)\b',
        r'answer is\s*(hateful|not-hateful)\b',
        r'answer:\s*(hateful|not-hateful)\b',
        r'the answer is\s*(hateful|not-hateful)\b',
        r'final answer:\s*(hateful|not-hateful)\b',
        r'conclusion:\s*(hateful|not-hateful)\b',
        r'result:\s*(hateful|not-hateful)\b',
        r'output:\s*(hateful|not-hateful)\b',
        r'judgment:\s*(hateful|not-hateful)\b',
        r'verdict:\s*(hateful|not-hateful)\b',
        r'\s*(hateful|not-hateful)\b'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer = match.group(1).strip().lower()
            if answer in ['hateful', 'not-hateful']:
                return answer
    
    return None

# ----------------------- 2. 数据加载 -----------------------
def load_data_from_jsonl(jsonl_path: str, img_base_dir: str = None):
    """从jsonl文件加载数据
    
    Args:
        jsonl_path: jsonl文件路径
        img_base_dir: 图片基础目录，如果提供，将与jsonl中的img路径拼接
        
    Returns:
        数据列表，每项包含id、img_path、label、text
    """
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            # 处理图片路径
            if img_base_dir:
                img_path = os.path.join(img_base_dir, item['img'])
            else:
                # 假设jsonl文件与图片目录在同一目录
                jsonl_dir = os.path.dirname(jsonl_path)
                img_path = os.path.join(jsonl_dir, item['img'])
            
            data.append({
                'id': item['id'],
                'img_path': img_path,
                'label': item['label'],
                'text': item['text']
            })
    return data

# ----------------------- 3. Q-Generator -----------------------


def generate_text_analysis(model, text: str = None) -> List[str]:
    Text_Analysis_TEMPLATE = (
        f"You are a meme text analysis expert. "
        f"Please provide a comprehensive analysis of the following meme text: \"{text}\" based on the following aspects:\n\n"
        f"1. Literal meaning: Explain the most direct, surface-level meaning of the text.\n"
        f"2. Possible metaphors, puns, or cultural references: Identify any implicit meanings, wordplay, or specific cultural associations that may be present in the text.\n"
        f"3. Emotional tone or attitude: Assess the emotions expressed by the text (such as humor, sarcasm, anger, joy, etc.).\n"
        f"4. Risk assessment: Does the text potentially contain offensive, discriminatory, or sensitive content? If so, please specify the reasons; if not, state \"No relevant risks identified.\"\n"
        f"5. Possible image context: Please reasonably speculate what kind of image might accompany this text, and what kind of message or emotional effect the combination of text and image might create.\n"
        f"Summarize your analysis into a concise and coherent paragraph of no more than 5 sentences."
    )

    # Text_Analysis_TEMPLATE = (
    #     '你是一位meme文本分析专家。'
    #     '请针对以下meme文本：“{text}”，按照如下要求进行综合分析：\n\n'
    #     '1. 字面含义：解释该文本最直接、表层的意思。\n'
    #     '2. 可能的隐喻、双关或文化指涉：指出文本中可能包含的隐含意义、特定文化关联。\n'
    #     '3. 情感色彩或态度：判断该文本表达的情绪（如幽默、讽刺、愤怒、喜悦等）。\n'
    #     '4. 风险评估：该文本是否可能包含冒犯性、歧视性或敏感含义？如有，请具体说明原因；如无，请写“未发现相关风险”。\n'
    #     '5. 可能的配图情境：请合理推测该文本可能会配合怎样的图片，以及图文可能形成怎样的信息或情感效果。\n'
    #     '请将所有分析内容整合为一段连贯、详细的自然语言描述。'
    # )
    # 如果有文本，添加到提示中

    messages = [
        {"role": "user",
         "content": [
             {"type": "text", "text": Text_Analysis_TEMPLATE}
         ]}
    ]
    raw = _generate_text(model, messages, max_new_tokens=512)
    print("-"*50)
    print("text analysis:")
    print(raw)
    print("-"*50)
    return raw


# ----------------------- 5. Reasoner -----------------------
# REASON_TEMPLATE = """你是仇恨言论检测专家。\n以下给出图像、其文字内容及 (问题, 回答) 组。\n\
# 1) 请在 <think> </think> 标签中写下推理过程；\n\
# 2) 在 <answer> </answer> 中只给出 hateful 或 non-hateful。"""


def reason(model, img: Image.Image, text_analysis: str, text: str = None) -> str:

    # problem = (
    #     f"As a hateful speech detection expert, you are presented with a meme sample containing an image and the overlaid text: \"{text}\".\n\n"
    #     f"Hateful speech is defined as a direct or indirect attack on people based on characteristics such as ethnicity, race, nationality, immigration status, religion, caste, sex, gender identity, sexual orientation, disability, or disease. "
    #     f"Such attacks may include dehumanizing comparisons (e.g., likening people to animals), statements of inferiority, calls for exclusion or segregation, or mocking hate crimes.\n\n"
    #     f"Exceptions: Attacks targeting individuals (e.g., public figures) that are not based on protected characteristics, or criticisms directed at hate groups (e.g., terrorist organizations), are **not** considered hateful.\n\n"
    #     f"Instructions:\n"
    #     f"1. Carefully analyze the visual and textual content.\n"
    #     f"2. Consider both overt and subtle hateful expressions based on the definition above.\n"
    #     f"3. Provide your reasoning process within <think> </think>.\n"
    #     f"4. Give your final decision using either 'hateful' or 'not-hateful', enclosed in <answer> </answer> tags."
    #     f"先分析图像内容，注意图像中的细节元素，然后综合文字的分析内容，最后综合判断是否属于仇恨言论。"
    # )
    problem = (
        f"As a hateful speech detection expert, you are presented with a meme sample containing an image and the overlaid text: \"{text}\".\n\n"
        f"Hateful speech is defined as a direct or indirect attack on people based on characteristics such as ethnicity, race, nationality, immigration status, religion, caste, sex, gender identity, sexual orientation, disability, or disease. "
        f"Such attacks may include dehumanizing comparisons (e.g., likening people to animals), statements of inferiority, calls for exclusion or segregation, or mocking hate crimes.\n\n"
        f"Exceptions: Attacks targeting individuals (e.g., public figures) that are not based on protected characteristics, or criticisms directed at hate groups (e.g., terrorist organizations), are **not** considered hateful.\n\n"
        f"Instructions:\n"
        f"1. Carefully analyze the visual content, paying close attention to detailed elements in the image.\n"
        f"2. Refer to the following text analysis for the overlaid text:\n"
        f"---\n"
        f"{text_analysis}\n"
        f"---\n"
        f"3. Consider both overt and subtle hateful expressions based on the definition above.\n"
        f"4. Provide your reasoning process within <think> </think>.\n"
        f"5. Give your final decision using either 'hateful' or 'not-hateful', enclosed in <answer> </answer> tags.\n"
        f"Make sure your reasoning integrates both the visual elements and the detailed text analysis before making the final judgment."
    )
    
    messages = [
        # {"role": "system", "content": REASON_TEMPLATE},
        {"role": "user",
         "content": [
             {"type": "image", "image": img},
             {"type": "text",
              "text": problem}
         ]}
    ]
    return _generate(model, messages, max_new_tokens=512, temperature=0.3)

# ----------------------- 6. 处理单个样本 -----------------------
def process_sample(model, item, verbose: bool = True):
    """处理单个样本
    
    Args:
        model: vLLM模型
        item: 样本数据，包含img_path、text、label等
        verbose: 是否打印详细信息
        
    Returns:
        预测结果，包含预测标签、问答对、推理结果等
    """
    img_path = item['img_path']
    text = item.get('text', '')
    ground_truth = item.get('label', None)
    
    if verbose:
        print(f"\n=== 处理样本 {item['id']} ===")
        print(f"图片路径: {img_path}")
        print(f"文本内容: {text}")
        print(f"真实标签: {ground_truth}")
    
    img = load_image(img_path)
    
    # Step-1: 生成text analysis
    if verbose:
        print(f"\n=== Step-1 生成text analysis ===")
    text_analysis = generate_text_analysis(model, text)


    # Step-3: Reasoner 综合判断
    if verbose:
        print("=== Step-3 Reasoner 综合判断 ===")
    result = reason(model, img, text_analysis, text)
    if verbose:
        print(result)
    
    # 提取最终结果
    prediction = None
    answer_match = extract_harm_answer(result)
    if answer_match:
        prediction_text = answer_match.lower()
        prediction = 1 if prediction_text == "hateful" else 0
    
    return {
        "id": item['id'],
        "prediction": prediction,
        "ground_truth": ground_truth,
        "text_analysis": text_analysis,
        "reasoning": result
    }

# ----------------------- 7. CLI 演示 -----------------------
def run_pipeline(model, image_path: str, k: int = 3):
    """运行单个图片的管道（向后兼容）"""
    img = load_image(image_path)
    print(f"\n=== Step-1 生成 {k} 个 probing 问题 ===")
    questions = generate_questions(model, img, k=k)
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q}")

    print("\n=== Step-2 VQA 回答 ===")
    qas = []
    for q in questions:
        a = answer_question(model, img, q)
        qas.append((q, a))
        print(f"Q: {q}\nA: {a}\n")

    print("=== Step-3 Reasoner 综合判断 ===")
    result = reason(model, img, qas)
    print(result)

def run_jsonl_pipeline(model, jsonl_path: str, img_base_dir: str = None, k: int = 3, verbose: bool = True, 
                       output_path: str = None, max_samples: int = None):
    """从jsonl文件运行管道
    
    Args:
        model: vLLM模型
        jsonl_path: jsonl文件路径
        img_base_dir: 图片基础目录
        k: 生成问题数量
        verbose: 是否打印详细信息
        output_path: 结果输出JSON文件路径
        max_samples: 最大处理样本数
    """
    # 加载数据
    data = load_data_from_jsonl(jsonl_path, img_base_dir)
    
    if max_samples:
        data = data[:max_samples]
    
    results = []
    correct = 0
    
    # 确保输出目录存在
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 初始化输出文件，写入开始的JSON数组标记
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('[\n')
    
    # 处理每个样本
    for i, item in enumerate(data):
        try:
            print(f"\n处理第 {i+1}/{len(data)} 个样本...")
            result = process_sample(model, item, verbose=verbose)
            results.append(result)
            
            # 每处理一个样本就追加写入文件
            if output_path:
                with open(output_path, 'a', encoding='utf-8') as f:
                    # 将结果转换为JSON字符串
                    result_json = json.dumps(result, ensure_ascii=False, indent=2)
                    # 写入这个结果，如果不是最后一个结果，添加逗号
                    if i < len(data) - 1:
                        f.write(result_json + ',\n')
                    else:
                        f.write(result_json + '\n')
                print(f"样本 {result['id']} 结果已追加到 {output_path}")
            
            # 统计准确率
            if result['ground_truth'] is not None and result['prediction'] is not None:
                if result['ground_truth'] == result['prediction']:
                    correct += 1
                    
            # 实时打印准确率
            if (i + 1) % 10 == 0 and i > 0:
                print(f"\n当前准确率: {correct}/{i+1} = {correct/(i+1):.4f}")
                
        except Exception as e:
            print(f"处理样本 {item['id']} 时出错: {e}")
            # 记录错误信息到日志
            if output_path:
                with open(output_path + ".error.log", 'a', encoding='utf-8') as f:
                    f.write(f"样本 {item['id']} 处理错误: {e}\n")
    
    # 完成JSON数组
    if output_path:
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(']\n')
    
    # 计算总体准确率
    if results:
        valid_results = [r for r in results if r['ground_truth'] is not None and r['prediction'] is not None]
        if valid_results:
            accuracy = correct / len(valid_results)
            print(f"\n总体准确率: {correct}/{len(valid_results)} = {accuracy:.4f}")
    
    # 保存汇总结果
    if output_path:
        summary_path = output_path + ".summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            summary = {
                "total_samples": len(data),
                "processed_samples": len(results),
                "valid_samples": len([r for r in results if r['ground_truth'] is not None and r['prediction'] is not None]),
                "correct_predictions": correct,
                "accuracy": correct / len([r for r in results if r['ground_truth'] is not None and r['prediction'] is not None]) if results else 0,
                "jsonl_path": jsonl_path,
                "img_base_dir": img_base_dir,
                "num_questions": k
            }
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"汇总结果已保存到 {summary_path}")
    
    return results

if __name__ == "__main__":
    '''
    此版本尝试如下： 见prompt
    给一个语言大模型 text 让其分析文本的隐喻等信息，输出的分析 + clean_img给VLM 
    两轮信息判断这个memes是否存在hatefulness
    '''
    time_start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("-image", type=str, default="/newdisk/public/wws/01-AIGC-GPRO/LLaMA-Factory/examples/memes_pipeline/01726.png", help="path to meme image (.jpg/.png)")
    parser.add_argument("-k", "--num_questions", type=int, default=3)
    parser.add_argument("-jsonl", type=str, default="/newdisk/public/wws/00-Dataset-AIGC/FHM_new/test_seen.jsonl", help="path to jsonl file")
    parser.add_argument("-img_dir", type=str, default="/newdisk/public/wws/00-Dataset-AIGC/FHM_new", help="base directory for images")
    parser.add_argument("-max_samples", type=int, default=None, help="maximum number of samples to process")
    parser.add_argument("-output", type=str, default=f"/newdisk/public/wws/01-AIGC-GPRO/LLaMA-Factory/examples/memes_pipeline/outputs/resultsv2_{time_start}.json", help="output JSON file path to save results")
    parser.add_argument("-mode", type=str, choices=["single", "jsonl"], default="jsonl", help="run mode: single image or jsonl file")
    parser.add_argument("-verbose", action="store_true", help="print detailed information")
    
    # GPU 设置
    parser.add_argument("--gpus", type=str, default="2", help="Visible GPU ids, e.g. '0,1'. Empty = all visible")
    parser.add_argument("--tp", type=int, default=1, help="Tensor‑parallel world size (default = #GPUs)")
    parser.add_argument("--gpu_mem", type=float, default=0.9, help="GPU memory utilization (0.0-1.0)")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    
    args = parser.parse_args()
    
    # 确定tensor_parallel_size
    gpu_count = get_devices_from_str(args.gpus)
    tp_size = args.tp or max(1, gpu_count)  # 默认TP = GPU数量
    
    # 加载模型
    model = load_vllm_model(
        model_id=MODEL_ID, 
        dtype=args.dtype, 
        gpus=args.gpus, 
        tp_size=tp_size,
        gpu_memory_utilization=args.gpu_mem
    )
    
    if args.mode == "single":
        run_pipeline(model, args.image, k=args.num_questions)
    else:
        run_jsonl_pipeline(
            model,
            args.jsonl, 
            img_base_dir=args.img_dir, 
            k=args.num_questions, 
            max_samples=args.max_samples,
            verbose=args.verbose,
            output_path=args.output
        )

    
    time_end = time.time()
    print(f"Total time: {(time_end - time_start)/60:.2f} minutes")
