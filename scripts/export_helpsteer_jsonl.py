from datasets import load_from_disk
import json, os

src = r"C:\yuhe32\dpo\ipo\data\raw\HelpSteer"   # 改成你 save_to_disk 的实际目录
out = r"C:\yuhe32\dpo\ipo\data\raw\helpsteer.jsonl"

ds = load_from_disk(src)["train"]

os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    for ex in ds:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print("Wrote:", out, "rows=", len(ds))