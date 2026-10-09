"""CPU regression probe against the actual locked judge tokenizers; no GPU allocation."""
import argparse
import json
from pathlib import Path

from common import judge_token_ids, load_json, verify_model_lock


def main():
    from transformers import AutoTokenizer
    import torch
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    a = p.parse_args()
    plan = load_json(a.plan)
    lock = load_json(plan['model_lock'])
    verify_model_lock(lock)
    results = []
    for name in ('nemotron', 'skywork'):
        tok = AutoTokenizer.from_pretrained(lock[name]['path'], local_files_only=True)
        ids = judge_token_ids(tok, 'Hello', 'Hi')
        tensor = torch.tensor([ids], dtype=torch.long)
        assert tensor.shape == (1, len(ids)) and len(ids) > 2
        results.append(dict(judge=name, explicit_type=type(ids).__name__,
                            tensor_shape=list(tensor.shape), passed=True))
    print(json.dumps(dict(tokenizer_probe=results, gpu_inference_tested=False)), flush=True)


if __name__ == '__main__':
    main()
