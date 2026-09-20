"""
CPU 评估脚本：在 GSM8K 测试集子集上对比 Base Model vs MemGen

使用数据集自带的标准答案（ground truth），计算准确率。
"""
import sys
import os
import copy
import time
import re
import json

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from datasets import load_dataset

from memgen.model.configuration_memgen import MemGenConfig
from memgen.model.modeling_memgen import MemGenModel, _remap_lora_adapter_key
from safetensors.torch import load_file as safe_load_file

BASE_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models/Qwen2.5-1.5B-Instruct")
CHECKPOINT_PATH = os.path.join(
    _PROJECT_ROOT,
    "models/memgen-checkpoints/Qwen2.5-1.5B-Instruct/gsm8k/weaver-sft/pn=1_pl=8_in=3_il=8/model"
)

# 评估题目数量（CPU 太慢，先跑 20 题看趋势）
NUM_EVAL_SAMPLES = 20


def extract_answer(text: str) -> str:
    """从模型输出中提取最终答案（\boxed{} 或 #### 后面的数字）"""
    # 尝试匹配 \boxed{...}
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip()
    
    # 尝试匹配 #### 后面的数字（GSM8K 格式）
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '')
    
    # 尝试匹配最后一个数字
    numbers = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
    if numbers:
        return numbers[-1]
    
    return ""


def extract_ground_truth(answer_str: str) -> str:
    """从 GSM8K 标准答案中提取最终数值"""
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', answer_str)
    if match:
        return match.group(1).replace(',', '')
    return ""


def build_prompt(tokenizer, question: str) -> str:
    """构建 GSM8K 风格的 prompt"""
    content = (
        "Solve the math problem step by step, and put the FINAL ANSWER in \\boxed{}.\n\n"
        f"Question: {question}"
    )
    messages = [{"role": "user", "content": content}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return text


def load_base_model_and_tokenizer():
    print("=" * 70)
    print("加载基础模型 (Qwen2.5-1.5B-Instruct)")
    print("=" * 70)
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
    
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.float32
    )
    model.eval()
    print(f"  模型加载完成，耗时 {time.time()-t0:.1f}s")
    print(f"  参数量: {sum(p.numel() for p in model.parameters())/1e6:.1f}M\n")
    return model, tokenizer


def load_memgen_model(base_model, tokenizer):
    print("=" * 70)
    print("加载 MemGen 模型 (Base + Weaver)")
    print("=" * 70)
    
    memgen_config = MemGenConfig.from_pretrained(
        BASE_MODEL_PATH,
        max_prompt_aug_num=1,
        max_inference_aug_num=3,
        prompt_latents_len=8,
        inference_latents_len=8,
        weaver_lora_config={
            "r": 16, "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"],
            "lora_dropout": 0.1, "bias": "none", "task_type": "CAUSAL_LM",
        },
        trigger_active=False,
        trigger_lora_config={
            "r": 16, "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"],
            "lora_dropout": 0.1, "bias": "none", "task_type": "CAUSAL_LM",
        },
    )
    
    print("  克隆基础模型...")
    reasoner = copy.deepcopy(base_model)
    weaver = copy.deepcopy(base_model)
    trigger = copy.deepcopy(base_model)
    
    model = MemGenModel(
        config=memgen_config,
        base_tokenizer=tokenizer,
        reasoner_base_model=reasoner,
        weaver_base_model=weaver,
        trigger_base_model=trigger,
    )
    
    print(f"  加载检查点: {CHECKPOINT_PATH}")
    t0 = time.time()
    
    # 加载投影层
    proj_state = torch.load(
        os.path.join(CHECKPOINT_PATH, "projs.bin"), map_location="cpu", weights_only=True
    )
    model.reasoner_to_weaver.load_state_dict(proj_state["reasoner_to_weaver"])
    model.weaver_to_reasoner.load_state_dict(proj_state["weaver_to_reasoner"])
    
    # 加载 weaver 状态
    weaver_state = torch.load(
        os.path.join(CHECKPOINT_PATH, "weaver.bin"), map_location="cpu", weights_only=True
    )
    model.weaver.prompt_query_latents.data.copy_(weaver_state["prompt_query_latents"])
    model.weaver.inference_query_latents.data.copy_(weaver_state["inference_query_latents"])
    model.weaver.prompt_latent_ln.load_state_dict(weaver_state["prompt_latent_ln"])
    model.weaver.inference_latent_ln.load_state_dict(weaver_state["inference_latent_ln"])
    model.weaver.prompt_latent_scale.data.copy_(weaver_state["prompt_latent_scale"])
    model.weaver.inference_latent_scale.data.copy_(weaver_state["inference_latent_scale"])
    
    # 加载 trigger 状态
    trigger_state = torch.load(
        os.path.join(CHECKPOINT_PATH, "trigger.bin"), map_location="cpu", weights_only=True
    )
    model.trigger.output_layer.load_state_dict(trigger_state["output_layer"])
    
    # 加载 weaver LoRA
    weaver_ckpt_sd = safe_load_file(
        os.path.join(CHECKPOINT_PATH, "weaver", "weaver", "adapter_model.safetensors"),
        device="cpu"
    )
    weaver_model_sd = model.weaver.model.state_dict()
    weaver_final = {}
    for k, v in weaver_ckpt_sd.items():
        new_key = _remap_lora_adapter_key(k, "weaver")
        if new_key in weaver_model_sd:
            weaver_final[new_key] = v
    model.weaver.model.load_state_dict(weaver_final, strict=False)
    
    # 加载 trigger LoRA
    trigger_ckpt_sd = safe_load_file(
        os.path.join(CHECKPOINT_PATH, "trigger", "trigger", "adapter_model.safetensors"),
        device="cpu"
    )
    trigger_model_sd = model.trigger.model.state_dict()
    trigger_final = {}
    for k, v in trigger_ckpt_sd.items():
        new_key = _remap_lora_adapter_key(k, "trigger")
        if new_key in trigger_model_sd:
            trigger_final[new_key] = v
    model.trigger.model.load_state_dict(trigger_final, strict=False)
    
    model.eval()
    print(f"  检查点加载完成，耗时 {time.time()-t0:.1f}s\n")
    return model


def generate_base(model, tokenizer, question, max_new_tokens=256):
    """用基础模型生成回答"""
    prompt = build_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=gen_config,
        )
    
    new_ids = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def generate_memgen(model, tokenizer, question, max_new_tokens=256):
    """用 MemGen 模型生成回答"""
    prompt = build_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        weaver_do_sample=False,
        trigger_do_sample=False,
    )
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=gen_config,
        )
    
    new_ids = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def main():
    print("=" * 70)
    print(f"GSM8K 评估 (CPU) - 测试 {NUM_EVAL_SAMPLES} 题")
    print("=" * 70)
    
    # 加载 GSM8K 测试集
    print("\n加载 GSM8K 测试集...")
    dataset = load_dataset("gsm8k", "main", split="test")
    samples = dataset.select(range(NUM_EVAL_SAMPLES))
    print(f"  已加载 {len(samples)} 道题目\n")
    
    # 加载模型
    base_model, tokenizer = load_base_model_and_tokenizer()
    memgen_model = load_memgen_model(base_model, tokenizer)
    
    # 评估
    base_correct = 0
    memgen_correct = 0
    results = []
    
    print("=" * 70)
    print("开始评估")
    print("=" * 70)
    
    for i, sample in enumerate(samples):
        question = sample["question"]
        ground_truth = extract_ground_truth(sample["answer"])
        
        print(f"\n[{i+1}/{NUM_EVAL_SAMPLES}]")
        print(f"  题目: {question[:80]}...")
        print(f"  标准答案: {ground_truth}")
        
        # Base model
        t0 = time.time()
        base_output = generate_base(base_model, tokenizer, question)
        base_time = time.time() - t0
        base_answer = extract_answer(base_output)
        base_is_correct = base_answer == ground_truth
        if base_is_correct:
            base_correct += 1
        
        # MemGen
        t0 = time.time()
        memgen_output = generate_memgen(memgen_model, tokenizer, question)
        memgen_time = time.time() - t0
        memgen_answer = extract_answer(memgen_output)
        memgen_is_correct = memgen_answer == ground_truth
        if memgen_is_correct:
            memgen_correct += 1
        
        print(f"  Base: {base_answer} {'[OK]' if base_is_correct else '[X]'} ({base_time:.1f}s)")
        print(f"  MemGen: {memgen_answer} {'[OK]' if memgen_is_correct else '[X]'} ({memgen_time:.1f}s)")
        
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "base_answer": base_answer,
            "base_correct": base_is_correct,
            "memgen_answer": memgen_answer,
            "memgen_correct": memgen_is_correct,
        })
    
    # 汇总
    print("\n" + "=" * 70)
    print("评估结果汇总")
    print("=" * 70)
    print(f"  总题数: {NUM_EVAL_SAMPLES}")
    print(f"  Base Model 正确: {base_correct}/{NUM_EVAL_SAMPLES} ({100*base_correct/NUM_EVAL_SAMPLES:.1f}%)")
    print(f"  MemGen 正确: {memgen_correct}/{NUM_EVAL_SAMPLES} ({100*memgen_correct/NUM_EVAL_SAMPLES:.1f}%)")
    
    # 保存详细结果
    output_path = os.path.join(_PROJECT_ROOT, "reproduction/eval_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  详细结果已保存: {output_path}")


if __name__ == "__main__":
    main()
