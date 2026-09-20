"""
2×2 Sanity Check: 语言 (中/英) × 模型 (Base/MemGen)

用 10 道题（5 简单 + 5 难）的中英文版本，快速验证：
- Weaver 是否只在英文上有效（训练分布内）
- 还是跨语言都能工作
- 难题能否揭示 Base 和 MemGen 的差异
"""
import sys
import os
import copy
import time

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from memgen.model.configuration_memgen import MemGenConfig
from memgen.model.modeling_memgen import MemGenModel, _remap_lora_adapter_key
from safetensors.torch import load_file as safe_load_file

BASE_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models/Qwen2.5-1.5B-Instruct")
CHECKPOINT_PATH = os.path.join(
    _PROJECT_ROOT,
    "models/memgen-checkpoints/Qwen2.5-1.5B-Instruct/gsm8k/weaver-sft/pn=1_pl=8_in=3_il=8/model"
)

# 10 道题目：5 道简单 + 5 道更难（多步推理）
QUESTIONS = [
    # === 简单题 ===
    {
        "en": "A book costs $5. If you buy 3 books, how much do you pay?",
        "zh": "一本书 5 美元。买 3 本书要付多少钱？",
        "answer": "15",
    },
    {
        "en": "There are 10 apples. You eat 2. How many apples are left?",
        "zh": "有 10 个苹果，吃了 2 个，还剩几个？",
        "answer": "8",
    },
    {
        "en": "A box has 12 candies. You share them equally among 3 children. How many candies does each child get?",
        "zh": "一个盒子里有 12 颗糖，平均分给 3 个小孩，每个小孩分到几颗？",
        "answer": "4",
    },
    {
        "en": "Tom has 20 dollars. He spends 8 dollars on a shirt. How much money does he have left?",
        "zh": "Tom 有 20 美元，买衬衫花了 8 美元，还剩多少钱？",
        "answer": "12",
    },
    {
        "en": "A car travels 60 miles per hour. How far does it travel in 2 hours?",
        "zh": "一辆车每小时开 60 英里，开 2 小时能开多远？",
        "answer": "120",
    },
    # === 难题（多步推理）===
    {
        # 之前做错的荣誉榜题
        "en": "In a class of 20 students, 3/5 of the students are girls. If 2/3 of the girls and all of the boys are on the honor roll, how many students are on the honor roll?",
        "zh": "一个班级有 20 名学生。3/5 的学生是女生。如果 2/3 的女生和所有男生都在荣誉榜上，荣誉榜上有多少名学生？",
        "answer": "16",
    },
    {
        # 比例 + 差值
        "en": "The ratio of boys to girls in a class is 3:5. If there are 40 students in total, how many more girls than boys are there?",
        "zh": "一个班级中男生与女生的比例是 3:5。如果总共有 40 名学生，女生比男生多多少人？",
        "answer": "10",
    },
    {
        # 连续百分比
        "en": "A store increases the price of a $50 item by 20%, then offers a 10% discount on the new price. What is the final price in dollars?",
        "zh": "一家商店把 50 美元的商品先提价 20%，再对新价格打九折（即降 10%）。最终价格是多少美元？",
        "answer": "54",
    },
    {
        # 多步购物 + 百分比折扣
        "en": "Alice buys 3 notebooks at $4 each and 2 pens at $3 each. If she gets a 15% discount on her total purchase, how much does she pay in dollars?",
        "zh": "Alice 买了 3 本笔记本，每本 4 美元，还有 2 支笔，每支 3 美元。如果总价打八五折（即优惠 15%），她一共要付多少钱？",
        "answer": "15.3",
    },
    {
        # 工程问题
        "en": "It takes 4 workers 6 days to paint a house. How many days would it take 8 workers to paint the same house, assuming they all work at the same rate?",
        "zh": "4 个工人花 6 天能粉刷一栋房子。如果 8 个工人以同样的速度工作，粉刷同一栋房子需要多少天？",
        "answer": "3",
    },
]


def build_prompt(tokenizer, question: str) -> str:
    content = (
        "Solve the math problem step by step, and put the FINAL ANSWER in \\boxed{}.\n\n"
        f"Question: {question}"
    )
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_models():
    print("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
    
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.float32)
    base_model.eval()
    
    # MemGen
    memgen_config = MemGenConfig.from_pretrained(
        BASE_MODEL_PATH,
        max_prompt_aug_num=1, max_inference_aug_num=3,
        prompt_latents_len=8, inference_latents_len=8,
        weaver_lora_config={"r": 16, "lora_alpha": 32, "target_modules": ["q_proj", "v_proj"],
                            "lora_dropout": 0.1, "bias": "none", "task_type": "CAUSAL_LM"},
        trigger_active=False,
        trigger_lora_config={"r": 16, "lora_alpha": 32, "target_modules": ["q_proj", "v_proj"],
                             "lora_dropout": 0.1, "bias": "none", "task_type": "CAUSAL_LM"},
    )
    
    reasoner = copy.deepcopy(base_model)
    weaver = copy.deepcopy(base_model)
    trigger = copy.deepcopy(base_model)
    
    memgen_model = MemGenModel(
        config=memgen_config, base_tokenizer=tokenizer,
        reasoner_base_model=reasoner, weaver_base_model=weaver, trigger_base_model=trigger,
    )
    
    # 加载检查点
    proj_state = torch.load(os.path.join(CHECKPOINT_PATH, "projs.bin"), map_location="cpu", weights_only=True)
    memgen_model.reasoner_to_weaver.load_state_dict(proj_state["reasoner_to_weaver"])
    memgen_model.weaver_to_reasoner.load_state_dict(proj_state["weaver_to_reasoner"])
    
    weaver_state = torch.load(os.path.join(CHECKPOINT_PATH, "weaver.bin"), map_location="cpu", weights_only=True)
    memgen_model.weaver.prompt_query_latents.data.copy_(weaver_state["prompt_query_latents"])
    memgen_model.weaver.inference_query_latents.data.copy_(weaver_state["inference_query_latents"])
    memgen_model.weaver.prompt_latent_ln.load_state_dict(weaver_state["prompt_latent_ln"])
    memgen_model.weaver.inference_latent_ln.load_state_dict(weaver_state["inference_latent_ln"])
    memgen_model.weaver.prompt_latent_scale.data.copy_(weaver_state["prompt_latent_scale"])
    memgen_model.weaver.inference_latent_scale.data.copy_(weaver_state["inference_latent_scale"])
    
    trigger_state = torch.load(os.path.join(CHECKPOINT_PATH, "trigger.bin"), map_location="cpu", weights_only=True)
    memgen_model.trigger.output_layer.load_state_dict(trigger_state["output_layer"])
    
    weaver_ckpt_sd = safe_load_file(os.path.join(CHECKPOINT_PATH, "weaver", "weaver", "adapter_model.safetensors"), device="cpu")
    weaver_model_sd = memgen_model.weaver.model.state_dict()
    weaver_final = {_remap_lora_adapter_key(k, "weaver"): v for k, v in weaver_ckpt_sd.items() if _remap_lora_adapter_key(k, "weaver") in weaver_model_sd}
    memgen_model.weaver.model.load_state_dict(weaver_final, strict=False)
    
    trigger_ckpt_sd = safe_load_file(os.path.join(CHECKPOINT_PATH, "trigger", "trigger", "adapter_model.safetensors"), device="cpu")
    trigger_model_sd = memgen_model.trigger.model.state_dict()
    trigger_final = {_remap_lora_adapter_key(k, "trigger"): v for k, v in trigger_ckpt_sd.items() if _remap_lora_adapter_key(k, "trigger") in trigger_model_sd}
    memgen_model.trigger.model.load_state_dict(trigger_final, strict=False)
    
    memgen_model.eval()
    print("模型加载完成\n")
    return base_model, memgen_model, tokenizer


def generate(model, tokenizer, question, is_memgen=False):
    prompt = build_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    
    gen_config = GenerationConfig(
        max_new_tokens=512, do_sample=False,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    if is_memgen:
        gen_config.use_cache = True
        gen_config.weaver_do_sample = False
        gen_config.trigger_do_sample = False
    
    with torch.no_grad():
        output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, generation_config=gen_config)
    
    new_ids = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def extract_answer(text):
    import re
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip().replace(',', '')
    numbers = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
    return numbers[-1] if numbers else ""


def main():
    # 分阶段加载模型，减少内存占用
    
    # 阶段 1: Base Model
    print("加载 Base Model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
    
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.float32)
    base_model.eval()
    
    print("\n" + "=" * 80)
    print("阶段 1/2: Base Model 测试")
    print("=" * 80)
    
    results = {"en": {"base": 0, "memgen": 0}, "zh": {"base": 0, "memgen": 0}}
    
    for i, q in enumerate(QUESTIONS):
        print(f"\n[题目 {i+1}/{len(QUESTIONS)}] 正确答案: {q['answer']}")
        print("-" * 80)

        for lang in ["en", "zh"]:
            question = q[lang]
            print(f"\n  [{lang.upper()}] {question[:60]}...")

            # Base
            t0 = time.time()
            base_out = generate(base_model, tokenizer, question, is_memgen=False)
            base_time = time.time() - t0
            base_ans = extract_answer(base_out)
            base_ok = base_ans == q['answer']
            if base_ok: results[lang]["base"] += 1
            print(f"    Base:   {base_ans:5s} {'[OK]' if base_ok else '[X]'} ({base_time:.1f}s)")
    
    # 释放 Base Model
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    # 阶段 2: MemGen Model
    print("\n\n" + "=" * 80)
    print("阶段 2/2: MemGen Model 测试")
    print("=" * 80)
    
    print("\n加载 MemGen Model...")
    memgen_config = MemGenConfig.from_pretrained(
        BASE_MODEL_PATH,
        max_prompt_aug_num=1, max_inference_aug_num=3,
        prompt_latents_len=8, inference_latents_len=8,
        weaver_lora_config={"r": 16, "lora_alpha": 32, "target_modules": ["q_proj", "v_proj"],
                            "lora_dropout": 0.1, "bias": "none", "task_type": "CAUSAL_LM"},
        trigger_active=False,
        trigger_lora_config={"r": 16, "lora_alpha": 32, "target_modules": ["q_proj", "v_proj"],
                             "lora_dropout": 0.1, "bias": "none", "task_type": "CAUSAL_LM"},
    )
    
    base_for_clone = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.float32)
    reasoner = copy.deepcopy(base_for_clone)
    weaver = copy.deepcopy(base_for_clone)
    trigger = copy.deepcopy(base_for_clone)
    del base_for_clone
    gc.collect()
    
    memgen_model = MemGenModel(
        config=memgen_config, base_tokenizer=tokenizer,
        reasoner_base_model=reasoner, weaver_base_model=weaver, trigger_base_model=trigger,
    )
    
    # 加载检查点
    proj_state = torch.load(os.path.join(CHECKPOINT_PATH, "projs.bin"), map_location="cpu", weights_only=True)
    memgen_model.reasoner_to_weaver.load_state_dict(proj_state["reasoner_to_weaver"])
    memgen_model.weaver_to_reasoner.load_state_dict(proj_state["weaver_to_reasoner"])
    
    weaver_state = torch.load(os.path.join(CHECKPOINT_PATH, "weaver.bin"), map_location="cpu", weights_only=True)
    memgen_model.weaver.prompt_query_latents.data.copy_(weaver_state["prompt_query_latents"])
    memgen_model.weaver.inference_query_latents.data.copy_(weaver_state["inference_query_latents"])
    memgen_model.weaver.prompt_latent_ln.load_state_dict(weaver_state["prompt_latent_ln"])
    memgen_model.weaver.inference_latent_ln.load_state_dict(weaver_state["inference_latent_ln"])
    memgen_model.weaver.prompt_latent_scale.data.copy_(weaver_state["prompt_latent_scale"])
    memgen_model.weaver.inference_latent_scale.data.copy_(weaver_state["inference_latent_scale"])
    
    trigger_state = torch.load(os.path.join(CHECKPOINT_PATH, "trigger.bin"), map_location="cpu", weights_only=True)
    memgen_model.trigger.output_layer.load_state_dict(trigger_state["output_layer"])
    
    weaver_ckpt_sd = safe_load_file(os.path.join(CHECKPOINT_PATH, "weaver", "weaver", "adapter_model.safetensors"), device="cpu")
    weaver_model_sd = memgen_model.weaver.model.state_dict()
    weaver_final = {_remap_lora_adapter_key(k, "weaver"): v for k, v in weaver_ckpt_sd.items() if _remap_lora_adapter_key(k, "weaver") in weaver_model_sd}
    memgen_model.weaver.model.load_state_dict(weaver_final, strict=False)
    
    trigger_ckpt_sd = safe_load_file(os.path.join(CHECKPOINT_PATH, "trigger", "trigger", "adapter_model.safetensors"), device="cpu")
    trigger_model_sd = memgen_model.trigger.model.state_dict()
    trigger_final = {_remap_lora_adapter_key(k, "trigger"): v for k, v in trigger_ckpt_sd.items() if _remap_lora_adapter_key(k, "trigger") in trigger_model_sd}
    memgen_model.trigger.model.load_state_dict(trigger_final, strict=False)
    
    memgen_model.eval()
    print("MemGen 加载完成\n")
    
    for i, q in enumerate(QUESTIONS):
        print(f"\n[题目 {i+1}/{len(QUESTIONS)}] 正确答案: {q['answer']}")
        print("-" * 80)

        for lang in ["en", "zh"]:
            question = q[lang]
            print(f"\n  [{lang.upper()}] {question[:60]}...")

            # MemGen
            t0 = time.time()
            memgen_out = generate(memgen_model, tokenizer, question, is_memgen=True)
            memgen_time = time.time() - t0
            memgen_ans = extract_answer(memgen_out)
            memgen_ok = memgen_ans == q['answer']
            if memgen_ok: results[lang]["memgen"] += 1
            print(f"    MemGen: {memgen_ans:5s} {'[OK]' if memgen_ok else '[X]'} ({memgen_time:.1f}s)")
    
    total = len(QUESTIONS)
    print("\n" + "=" * 80)
    print(f"汇总 (正确数/{total})")
    print("=" * 80)
    print(f"         | Base | MemGen")
    print(f"  英文   |  {results['en']['base']}   |   {results['en']['memgen']}")
    print(f"  中文   |  {results['zh']['base']}   |   {results['zh']['memgen']}")
    print()
    
    if results['en']['memgen'] > results['en']['base'] and results['zh']['memgen'] <= results['zh']['base']:
        print("结论: Weaver 只在英文（训练分布内）有效，中文 OOD 导致失效")
    elif results['en']['memgen'] > results['en']['base'] and results['zh']['memgen'] > results['zh']['base']:
        print("结论: Weaver 跨语言都有效")
    elif results['en']['memgen'] <= results['en']['base']:
        print("结论: MemGen 在英文上也没有明显提升，需要检查 LoRA 加载或其他问题")
    else:
        print("结论: 需要更多数据判断")


if __name__ == "__main__":
    main()
