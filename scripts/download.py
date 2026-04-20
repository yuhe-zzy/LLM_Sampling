from datasets import load_dataset

cache_dir = r"C:\yuhe32\dpo\ipo\data\hf_cache"        # 下载缓存放这里
save_dir  = r"C:\yuhe32\dpo\ipo\data\raw\HelpSteer"   # 固化后的数据放这里

ds = load_dataset("nvidia/HelpSteer", cache_dir=cache_dir)
ds.save_to_disk(save_dir)

print(ds)  # DatasetDict(train=..., validation=...)