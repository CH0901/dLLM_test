import os
import json
import math
import random
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_decode(tokenizer, ids):
    try:
        return tokenizer.decode(ids, skip_special_tokens=False)
    except:
        return str(ids)


def build_masked_input(tokenizer, prompt, gen_length, mask_token_id):
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"][0].tolist()
    attn = enc["attention_mask"][0].tolist()

    suffix = [mask_token_id] * gen_length
    full_ids = input_ids + suffix
    full_attn = attn + [1] * gen_length

    prompt_len = len(input_ids)
    total_len = len(full_ids)
    gen_positions = list(range(prompt_len, total_len))
    return full_ids, full_attn, prompt_len, gen_positions


@torch.no_grad()
def simple_masked_decode(
    model,
    tokenizer,
    prompt,
    gen_length=32,
    steps=32,
    temperature=0.0,
    topk_per_step=1,
    device="cuda",
):
    """
    레포 내부 전용 sampler를 강제하지 않고,
    'step-wise unmask trajectory' 관찰용으로 간단한 masked decode를 수행.
    목적은 benchmark가 아니라 FP/Q trajectory 비교.
    """

    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        # 일부 tokenizer는 [MASK] 등록이 없을 수 있음
        # LLaDA라면 보통 존재하지만, 없으면 문자열로 강제 추가
        tokenizer.add_special_tokens({"additional_special_tokens": ["[MASK]"]})
        if tokenizer.mask_token_id is None:
            mask_token_id = tokenizer.convert_tokens_to_ids("[MASK]")
        else:
            mask_token_id = tokenizer.mask_token_id

    ids, attn, prompt_len, gen_positions = build_masked_input(
        tokenizer, prompt, gen_length, mask_token_id
    )

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.tensor([attn], dtype=torch.long, device=device)

    traj = []
    remaining = set(gen_positions)

    for step in range(steps):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0]  # [seq, vocab]

        candidate_info = []
        for pos in list(remaining):
            pos_logits = logits[pos]
            probs = F.softmax(pos_logits, dim=-1)
            conf, tok = torch.max(probs, dim=-1)

            candidate_info.append({
                "position": int(pos),
                "token_id": int(tok.item()),
                "token_str": tokenizer.decode([int(tok.item())], skip_special_tokens=False),
                "confidence": float(conf.item()),
            })

        if len(candidate_info) == 0:
            break

        # confidence 높은 순으로 정렬
        candidate_info = sorted(candidate_info, key=lambda x: x["confidence"], reverse=True)

        # step마다 몇 개 unmask할지 결정
        k = min(topk_per_step, len(candidate_info))
        chosen = candidate_info[:k]

        for item in chosen:
            pos = item["position"]
            tok = item["token_id"]
            input_ids[0, pos] = tok
            remaining.remove(pos)

        current_ids = input_ids[0].tolist()

        traj.append({
            "step": step,
            "num_remaining_masks": len(remaining),
            "newly_unmasked": chosen,
            "current_ids": current_ids,
            "current_text_full": safe_decode(tokenizer, current_ids),
            "generated_region_ids": current_ids[prompt_len:prompt_len+gen_length],
            "generated_region_text": safe_decode(
                tokenizer,
                current_ids[prompt_len:prompt_len+gen_length]
            ),
        })

        if len(remaining) == 0:
            break

    return {
        "prompt": prompt,
        "prompt_len": prompt_len,
        "gen_length": gen_length,
        "steps_requested": steps,
        "steps_executed": len(traj),
        "trajectory": traj,
    }


def load_model_and_tokenizer(model_path, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp_model_path", type=str, required=True)
    parser.add_argument("--q_model_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--gen_length", type=int, default=32)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--topk_per_step", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    prompts = [
        "Write one short paragraph about why regular exercise is helpful.",
        "Explain the greenhouse effect in simple terms.",
        "Describe a student preparing for an exam the night before.",
        "Write two sentences about the advantages of public transportation.",
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading FP model...")
    fp_model, fp_tok = load_model_and_tokenizer(args.fp_model_path, device=device)

    print("Loading Q model...")
    q_model, q_tok = load_model_and_tokenizer(args.q_model_path, device=device)

    for i, prompt in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] prompt: {prompt}")

        fp_out = simple_masked_decode(
            fp_model, fp_tok, prompt,
            gen_length=args.gen_length,
            steps=args.steps,
            topk_per_step=args.topk_per_step,
            device=device,
        )

        q_out = simple_masked_decode(
            q_model, q_tok, prompt,
            gen_length=args.gen_length,
            steps=args.steps,
            topk_per_step=args.topk_per_step,
            device=device,
        )

        with open(os.path.join(args.save_dir, f"sample_{i}_fp.json"), "w", encoding="utf-8") as f:
            json.dump(fp_out, f, ensure_ascii=False, indent=2)

        with open(os.path.join(args.save_dir, f"sample_{i}_q.json"), "w", encoding="utf-8") as f:
            json.dump(q_out, f, ensure_ascii=False, indent=2)

    print("✅ trajectory 저장 완료")


if __name__ == "__main__":
    main()
