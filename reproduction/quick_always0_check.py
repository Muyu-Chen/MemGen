#!/usr/bin/env python3
"""
快速验证：直接查看 MemGen always_0 的原始输出
"""
import sys
import os
import copy
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memgen.model.configuration_memgen import MemGenConfig
from memgen.model.modeling_memgen import MemGenModel

# 测试 5 道题
QUESTIONS = [
    {"question": "What is 2+3?", "answer": "5"},
    {"question": "What is 10-4?", "answer": "6"},
    {"question": "What is 3*5?", "answer": "15"},
    {"question": "John has 5 apples and buys 2 more. How many apples does John have?", "answer": "7"},
    {"question": "What is 7+8?", "answer": "15"},
]

BASE_MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"
CHECKPOINT_PATH = os.path.join(
    _PROJECT_ROOT,
    "models/memgen-checkpoints/Qwen2.5-1.5B-Instruct/gsm8k/weaver-sft/pn=1_pl=8_in=3_il=8/model"
)

WEAVER_LORA_CONFIG = {"r": 8, "lora_alpha": 16, "lora_dropout": 0.0, "task_type": "CAUSAL_LM", "target_modules": ["q_proj", "v_proj"]}
TRIGGER_LORA_CONFIG = {"r": 8, "lora_alpha": 16, "lora_dropout": 0.0, "task_type": "CAUSAL_LM", "target_modules": ["q_proj", "v_proj"]}

def main():
    print("=" * 80)
    print("快速验证：MemGen always_0 原始输出")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH, trust_remote_code=True, local_files_only=True)

    # 加载 MemGen 模型
    print("\n加载 MemGen 模型...")
    config = MemGenConfig.from_pretrained(
        BASE_MODEL_PATH,
        max_prompt_aug_num=1, max_inference_aug_num=3,
        prompt_latents_len=8, inference_latents_len=8,
        weaver_lora_config=WEAVER_LORA_CONFIG,
        trigger_active=True,
        trigger_lora_config=TRIGGER_LORA_CONFIG,
    )

    base_for_clone = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.float32)
    reasoner = copy.deepcopy(base_for_clone)
    weaver = copy.deepcopy(base_for_clone)
    trigger = copy.deepcopy(base_for_clone)
    del base_for_clone
    gc.collect()

    model = MemGenModel(
        config=config, base_tokenizer=tokenizer,
        reasoner_base_model=reasoner, weaver_base_model=weaver, trigger_base_model=trigger,
    )

    # 加载 projections
    proj_state = torch.load(os.path.join(CHECKPOINT_PATH, "projs.bin"), map_location="cpu", weights_only=True)
    model.reasoner_to_weaver.load_state_dict(proj_state["reasoner_to_weaver"])
    model.weaver_to_reasoner.load_state_dict(proj_state["weaver_to_reasoner"])

    # 加载 weaver
    weaver_state = torch.load(os.path.join(CHECKPOINT_PATH, "weaver.bin"), map_location="cpu", weights_only=True)
    model.weaver.prompt_query_latents.data.copy_(weaver_state["prompt_query_latents"])
    model.weaver.inference_query_latents.data.copy_(weaver_state["inference_query_latents"])

    # 加载 trigger
    trigger_state = torch.load(os.path.join(CHECKPOINT_PATH, "trigger.bin"), map_location="cpu", weights_only=True)
    model.trigger.output_layer.load_state_dict(trigger_state["output_layer"])
    model.trigger.load_adapter(
        os.path.join(CHECKPOINT_PATH, "trigger"),
        adapter_name="trigger",
    )
    model.set_trigger_active(True)
    model.eval()

    print("模型加载完成。\n")

    # Monkey-patch: always_0
    original_forward = model.trigger.forward
    def always_0_forward(**kwargs):
        input_ids = kwargs["input_ids"]
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros(batch_size, seq_len, 2, device=input_ids.device)
        logits[..., 0] = 10.0
        logits[..., 1] = -10.0
        return logits
    model.trigger.forward = always_0_forward

    # 测试 5 道题
    print("=" * 80)
    print("MemGen with always_0 (force no augmentation)")
    print("=" * 80)

    for i, q in enumerate(QUESTIONS, 1):
        prompt = f"Question: {q['question']}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        with torch.no_grad():
            output_ids, aug_mask = model.generate(
                input_ids,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_ids = output_ids[0, input_ids.shape[1]:]
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        aug_count = (aug_mask[0, :len(new_ids)] == 1).sum().item()

        print(f"\n[{i}] {q['question']}")
        print(f"    正确答案: {q['answer']}")
        print(f"    增强次数: {aug_count}")
        print(f"    生成内容:")
        for line in text.split('\n'):
            print(f"      {line}")

if __name__ == "__main__":
    main()
