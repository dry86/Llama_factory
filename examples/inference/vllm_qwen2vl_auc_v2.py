import argparse
import json
import time

import os
from pathlib import Path
from typing import List, Dict, Any
import re
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from PIL import Image
from transformers import AutoTokenizer, AutoProcessor

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import random

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_json(fp: Path) -> List[Dict[str, Any]]:
    """Load evaluation data from a JSON file.

    The file can either be a list of objects or a dict that contains the list
    under the key ``data``.
    """
    with fp.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    # Accept two common layouts: list[dict] or {"data": list[dict]}
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and "data" in obj:
        return obj["data"]
    raise ValueError("Unrecognized JSON format – expected a list or a dict with key 'data'.")

def load_jsonl(fp: Path) -> List[Dict[str, Any]]:
    """Load evaluation data from a JSONL file.

    The file can either be a list of objects or a dict that contains the list
    under the key ``data``.
    """
    data = []
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    
    return data

def build_llm(model_name: str,
              adapter_path: str = None,
              dtype: str = "bfloat16",
              tp_size: int = 1,
              max_model_len: int = 4096,
              gpu_memory_utilization: float = 0.9,
              max_lora_rank: int = 32,
              ) -> LLM:
    """Create a vLLM engine for Qwen2-VL-2B."""
    import torch
    torch.cuda.empty_cache()

    engine_args = {
        "model": model_name,
        "trust_remote_code": True,
        "dtype": dtype,
        "tensor_parallel_size": tp_size,
        "max_model_len": max_model_len,
        "limit_mm_per_prompt": {"image": 1, "video": 0},  # Qwen2‑VL supports 1 img/prompt by default
        "gpu_memory_utilization": gpu_memory_utilization,
        "enable_lora": adapter_path is not None,  # 启用 LoRA 支持
        "max_lora_rank": max_lora_rank,  # 设置最大 LoRA rank
    }

    return LLM(**engine_args)


def make_analysis_prompt(processor: AutoProcessor, text: str) -> str:
    """Wrap raw prompt text with ChatML tags expected by Qwen‑family models."""
    analysis_prompt = (
        f"As a hateful speech detection expert, you are presented with a meme sample containing an image and the overlaid text: \"{text}\".\n\n"
        f"Hateful speech is defined as a direct or indirect attack on people based on characteristics such as ethnicity, race, nationality, immigration status, religion, caste, sex, gender identity, sexual orientation, disability, or disease. "
        f"Such attacks may include dehumanizing comparisons (e.g., likening people to animals), statements of inferiority, calls for exclusion or segregation, or mocking hate crimes.\n\n"
        f"Exceptions: Attacks targeting individuals (e.g., public figures) that are not based on protected characteristics, or criticisms directed at hate groups (e.g., terrorist organizations), are **not** considered hateful.\n\n"
        f"Instructions:\n"
        f"1. Carefully analyze the visual and textual content.\n"
        f"2. Consider both overt and subtle hateful expressions based on the definition above.\n"
        f"3. Provide only your reasoning process, enclosed in <think> </think> tags.\n"
        f"4. **Do NOT give a final decision yet.**"
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},                       # 只写 type 即可，真正图片走 mm_data
                {"type": "text", "text": analysis_prompt},
            ],
        }
    ]
    chatml_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return chatml_prompt

# ------------- NEW: decision-prompt 构造 ----------------
def make_decision_prompt(processor: AutoProcessor, analysis: str) -> str:
    """把 step-1 产出的 <think> … </think> 作为上下文，引导模型给出 <answer>。"""
    decision_prompt = (
        "You are a hateful-speech classification assistant.\n"
        "Below is the reasoning produced in the previous step:\n"
        f"{analysis}\n"
        "Based on this reasoning, is it hateful? "
        "Output format: yes | no \n"
        "Do not repeat the reasoning or add any extra text."
    )
    # 用统一 chat_template，此处只有纯文本
    messages = [
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": decision_prompt}
            ]
        }
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def extract_harm_answer(text: str) -> str | None:
    """
    从文本中提取 Yes/No 答案
    
    Args:
        text: 包含答案的文本
        
    Returns:
        提取出的答案 (Yes/No) 或 None
    """
    text = text.lower().strip()
    if text == "yes":
        return "yes"
    elif text == "no":
        return "no"
    
    # 尝试从非标准格式中提取
    patterns = [
        r'答案是\s*(Yes|No)\b',
        r'answer is\s*(Yes|No)\b',
        r'answer:\s*(Yes|No)\b',
        r'the answer is\s*(Yes|No)\b',
        r'final answer:\s*(Yes|No)\b',
        r'conclusion:\s*(Yes|No)\b',
        r'result:\s*(Yes|No)\b',
        r'output:\s*(Yes|No)\b',
        r'\n(Yes|No)$',
        r'(Yes|No)$'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            return answer
    
    return None

def calculate_metrics(predictions: List[Dict[str, Any]]) -> Dict[str, float]:
    """计算评估指标"""
    y_true = []
    y_pred = []
    
    for pred in predictions:
        try:
            gt = pred["gt"].lower()
            pred_answer = pred["pred"].lower()
            
            if gt in ["yes", "no"] and pred_answer in ["yes", "no"]:
                    y_true.append(1 if gt == "yes" else 0)
                    y_pred.append(1 if pred_answer == "yes" else 0)
        except Exception as e:
            print(f"处理错误: {e}")
            continue
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array([p["p_yes"] for p in predictions])
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'auroc': roc_auc_score(y_true, y_prob)
    }
    
    return metrics


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch inference with Qwen2‑VL‑2B on vLLM – JSON I/O version")
    parser.add_argument("--dataset", type=Path, default=Path("/newdisk/public/wws/01-AIGC-GPRO/LLaMA-Factory/data_use/test_seen.jsonl"), help="Path to evaluation JSON file")
    parser.add_argument("--img_dir", type=str, default="/newdisk/public/wws/00-Dataset-AIGC/FHM_new/img", help="base directory for images")
    parser.add_argument("--save", type=Path, default=Path("predictions_auc_v2.json"), help="Path to output JSON file")
    parser.add_argument("--model", type=str, default="/newdisk/public/wws/01-AIGC-GPRO/LLaMA-Factory/output/qwen2vl_7b/lora_450v2_sft_4e_lr1e-4", help="HF hub id or local dir")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--adapter", type=str, default=None, help="Path to LoRA adapter")
    parser.add_argument("--max_lora_rank", type=int, default=8, help="Maximum LoRA rank (should match the rank used in SFT)")

    # GPU & performance
    parser.add_argument("--gpus", type=str, default="2", help="Visible GPU ids, e.g. '0,1'. Empty = all visible")
    parser.add_argument("--tp", type=int, default=1, help="Tensor‑parallel world size (default = #GPUs)")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ----------------------------------------------------------
    # GPU visibility & TP size
    # ----------------------------------------------------------
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus  # must be set *before* vLLM loads CUDA context
        gpu_count = len(args.gpus.split(","))
    else:
        # Count GPUs from nvidia‑smi visibility (fallback 1)
        try:
            import torch
            gpu_count = torch.cuda.device_count()
        except Exception:
            gpu_count = 1

    tp_size = args.tp or max(1, gpu_count)  # default TP = number of visible GPUs

    # Build engine & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model)
    llm = build_llm(args.model, 
                    args.adapter, 
                    dtype=args.dtype, 
                    tp_size=tp_size, 
                    max_lora_rank=args.max_lora_rank, 
                    gpu_memory_utilization=args.gpu_memory_utilization
                )

    sampler = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    # Load dataset (list of dict)
    samples = load_jsonl(args.dataset)

    # Construct vLLM request objects
    analysis_requests = []
    for ex in samples:
        text = ex["text"]
        chatml_prompt = make_analysis_prompt(processor, text)
        prompt_ids = tokenizer.encode(chatml_prompt)

        
        img_path = Path(args.img_dir, ex["img"])
        if not img_path.is_file():
            raise FileNotFoundError(img_path)
        image = Image.open(img_path).convert("RGB")

        request = {
            "prompt_token_ids": prompt_ids,
            "multi_modal_data": {"image": [image]},
        }
        
        # 如果使用了 LoRA adapter，添加 LoRA 请求
        if args.adapter:
            request["lora_request"] = LoRARequest("default", 1, args.adapter)
            
        analysis_requests.append(request)

    # Inference in chunks
    analysis_outputs = []
    for i in range(0, len(analysis_requests), args.batch_size):
        chunk = analysis_requests[i : i + args.batch_size]
        results = llm.generate(chunk, sampler)
        analysis_outputs.extend(results)
        # break

    # -------- NEW: decision pass ----------
    decision_requests = []
    for res in analysis_outputs:
        analysis_text = res.outputs[0].text      # step-1 产出的 <think>…</think>
        dec_prompt    = make_decision_prompt(processor, analysis_text)
        prompt_ids    = tokenizer.encode(dec_prompt)

        decision_requests.append({
            "prompt_token_ids": prompt_ids,
            "multi_modal_data": {},              # ★ 无图片
        })

    yes_ids = tokenizer.encode("yes", add_special_tokens=False)
    no_ids  = tokenizer.encode("no",  add_special_tokens=False)
    assert len(yes_ids) == len(no_ids) == 1, "分词不是单 token 时需改写掩码逻辑"
    YES_ID, NO_ID = yes_ids[0], no_ids[0]

    # yes_no_mask = LogitsProcessorSpec(
    #     name="vocab_mask",                # 内置处理器
    #     # keep == True → 只保留列表里的 token
    #     # keep == False → 屏蔽列表里的 token
    #     kwargs={"vocab": [YES_ID, NO_ID], "keep": True},
    # )

    # 可用更小 max_tokens 和低 temperature
    decision_sampler = SamplingParams(
        temperature=0.0,
        max_tokens=1,                     # 只要第一个 token
        logprobs=2,                       # 要求返回 top-2 词的对数概率
        stop_token_ids=[tokenizer.eos_token_id],
        allowed_token_ids=[YES_ID, NO_ID],   # ★ 关键行：词表硬裁成 {yes,no}
    )

    decision_outputs = []
    for i in range(0, len(decision_requests), args.batch_size):
        chunk = decision_requests[i : i + args.batch_size]
        results = llm.generate(chunk, decision_sampler)


        decision_outputs.extend(results)

    # ---------------- collect ----------------
    predictions = []
    for sample, ana_res, dec_res in zip(samples, analysis_outputs, decision_outputs):
        think = ana_res.outputs[0].text

        chosen_id = dec_res.outputs[0].token_ids[0]
        lp_dict = dec_res.outputs[0].logprobs[0]                  # {id: logp, ...}
        lp_yes = lp_dict.get(YES_ID, float("-inf")).logprob
        lp_no  = lp_dict.get(NO_ID,  float("-inf")).logprob
        # p_yes  = np.exp(lp_yes) / (np.exp(lp_yes) + np.exp(lp_no))
        p_yes = np.exp(lp_yes - np.logaddexp(lp_yes, lp_no))

        pred = "yes" if chosen_id == YES_ID else "no"
        gt = "yes" if sample["label"] == 1 else "no"

        predictions.append({
            "image": sample["img"],
            "text":  sample["text"],
            "think": think,
            "answer_raw": dec_res.outputs[0].text,
            "p_yes": p_yes,
            "gt": gt,
            "pred": pred,
            "is_correct": gt == pred,
        })
        

    # 计算评估指标
    metrics = calculate_metrics(predictions)
    print("\n评估指标:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # 将指标添加到结果的最前面
    final_results = {
        "metrics": metrics,
        "predictions": predictions
    }

    args.save.parent.mkdir(parents=True, exist_ok=True)
    with args.save.open("w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    print(f"\n保存了 {len(predictions)} 条预测结果 → {args.save}")


if __name__ == "__main__":
    # 设置运行时间
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"运行时间: {(end_time - start_time) / 60:.2f} 分钟")
