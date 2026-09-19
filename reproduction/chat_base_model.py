"""
Interactive chat with base model (Qwen2.5-1.5B-Instruct) only.

Usage:
    python reproduction/chat_base_model.py
    python reproduction/chat_base_model.py --max-new-tokens 256
"""
import sys
import os
import argparse
import time

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

DEFAULT_MODEL = os.path.join(_PROJECT_ROOT, "models/Qwen2.5-1.5B-Instruct")
if not os.path.isdir(DEFAULT_MODEL):
    DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def main():
    parser = argparse.ArgumentParser(description="Chat with base model")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Model name or path (default: %(default)s)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

    dtype = torch.float32 if not torch.cuda.is_available() else torch.bfloat16
    model_kwargs = {"torch_dtype": dtype}
    if torch.cuda.is_available():
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s "
          f"({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)\n")

    print("=" * 60)
    print(f"  Base Model Chat ({os.path.basename(args.model)})")
    print("  Type your question and press Enter.")
    print("  Commands: 'quit'/'exit' to stop, 'clear' to reset history.")
    print("=" * 60)

    messages = [{"role": "system", "content": "请用中文回答用户的问题。"}]

    while True:
        try:
            user_input = input("\n[You] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if user_input.lower() == "clear":
            messages = [{"role": "system", "content": "请用中文回答用户的问题。"}]
            print("[History cleared]")
            continue

        messages.append({"role": "user", "content": user_input})

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        gen_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        t0 = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=gen_config,
            )
        elapsed = time.time() - t0

        new_ids = output_ids[0, input_ids.shape[1]:]
        reply = tokenizer.decode(new_ids, skip_special_tokens=True)
        n_tokens = new_ids.shape[0]

        print(f"\n[Model] ({n_tokens} tokens, {elapsed:.1f}s)")
        print(reply)

        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
