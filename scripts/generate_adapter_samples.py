from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_prompts(path: str, count: int, seed: int):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = obj.get("prompt")
            if isinstance(prompt, str) and prompt:
                rows.append((row_index, int(obj.get("prompt_id", row_index)), prompt))
    if not rows:
        raise ValueError(f"No prompts found in {path}")
    rng = random.Random(seed)
    selected = sorted(rng.sample(range(len(rows)), k=min(count, len(rows))))
    return [rows[i] for i in selected]


def dominant_share(values) -> float:
    if not values:
        return 0.0
    return Counter(values).most_common(1)[0][1] / len(values)


def dominant_value(values):
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--prompts_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--num_prompts", type=int, default=50)
    parser.add_argument("--num_responses", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--prompt_seed", type=int, default=20260810)
    parser.add_argument("--generation_seed", type=int, default=777123)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    prompts = read_prompts(args.prompts_path, args.num_prompts, args.prompt_seed)

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    old_padding_side = tok.padding_side
    tok.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=None,
    ).to("cuda:0")
    adapter_path = args.adapter_path.strip()
    if adapter_path:
        model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    else:
        model = base
    model.eval()

    random.seed(args.generation_seed)
    torch.manual_seed(args.generation_seed)
    torch.cuda.manual_seed_all(args.generation_seed)

    output_rows = []
    for start in range(0, len(prompts), max(1, args.batch_size)):
        chunk = prompts[start : start + max(1, args.batch_size)]
        texts = [row[2] for row in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        input_ids = enc["input_ids"].to("cuda:0")
        attention_mask = enc["attention_mask"].to("cuda:0")
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=args.num_responses,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        prompt_width = input_ids.shape[1]
        for local_index, (row_index, prompt_id, prompt) in enumerate(chunk):
            for response_index in range(args.num_responses):
                out_index = local_index * args.num_responses + response_index
                raw_ids = outputs[out_index, prompt_width:].detach().cpu().tolist()
                eos_position = None
                if tok.eos_token_id in raw_ids:
                    eos_position = raw_ids.index(tok.eos_token_id)
                    generated_ids = raw_ids[:eos_position]
                else:
                    generated_ids = raw_ids
                response = tok.decode(generated_ids, skip_special_tokens=True).strip()
                nonspace_chars = [char for char in response if not char.isspace()]
                normalized = " ".join(response.lower().split())
                all_star = bool(nonspace_chars) and set(nonspace_chars) == {"*"}
                dominant_token_id = dominant_value(generated_ids)
                output_rows.append(
                    {
                        "stage": args.stage,
                        "source_row_index": row_index,
                        "prompt_id": prompt_id,
                        "response_index": response_index,
                        "generated_token_count": len(generated_ids),
                        "hit_max_new_tokens": int(eos_position is None),
                        "response_length_chars": len(response),
                        "all_star": int(all_star),
                        "dominant_char_share": dominant_share(nonspace_chars),
                        "dominant_token_share": dominant_share(generated_ids),
                        "dominant_token_id": dominant_token_id,
                        "dominant_token_text": (
                            tok.decode([dominant_token_id]) if dominant_token_id is not None else ""
                        ),
                        "unique_token_ratio": (
                            len(set(generated_ids)) / len(generated_ids) if generated_ids else 0.0
                        ),
                        "normalized_response": normalized,
                        "prompt": prompt,
                        "response": response,
                    }
                )

    csv_path = os.path.join(args.output_dir, f"open_generation_{args.stage}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    n = len(output_rows)
    exact_counts = Counter(row["normalized_response"] for row in output_rows)
    summary = {
        "stage": args.stage,
        "model_path": args.model_path,
        "adapter_path": adapter_path or None,
        "num_prompts": len(prompts),
        "num_responses": n,
        "exact_unique_responses": len(exact_counts),
        "exact_unique_ratio": len(exact_counts) / n,
        "all_star_count": sum(row["all_star"] for row in output_rows),
        "all_star_fraction": sum(row["all_star"] for row in output_rows) / n,
        "hit_max_new_tokens_count": sum(row["hit_max_new_tokens"] for row in output_rows),
        "hit_max_new_tokens_fraction": sum(row["hit_max_new_tokens"] for row in output_rows) / n,
        "dominant_char_ge_0_95_count": sum(
            row["dominant_char_share"] >= 0.95 for row in output_rows
        ),
        "dominant_char_ge_0_95_fraction": sum(
            row["dominant_char_share"] >= 0.95 for row in output_rows
        )
        / n,
        "dominant_token_ge_0_95_count": sum(
            row["dominant_token_share"] >= 0.95 for row in output_rows
        ),
        "dominant_token_ge_0_95_fraction": sum(
            row["dominant_token_share"] >= 0.95 for row in output_rows
        )
        / n,
        "top_dominant_tokens": [
            {
                "token_id": token_id,
                "token_text": tok.decode([token_id]),
                "response_count": count,
            }
            for token_id, count in Counter(
                row["dominant_token_id"]
                for row in output_rows
                if row["dominant_token_id"] is not None
            ).most_common(10)
        ],
        "mean_generated_token_count": sum(row["generated_token_count"] for row in output_rows) / n,
        "mean_unique_token_ratio": sum(row["unique_token_ratio"] for row in output_rows) / n,
        "top_exact_responses": [
            {"count": count, "preview": text[:200]}
            for text, count in exact_counts.most_common(10)
        ],
        "csv_path": csv_path,
    }
    summary_path = os.path.join(args.output_dir, f"open_generation_{args.stage}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    tok.padding_side = old_padding_side


if __name__ == "__main__":
    main()
