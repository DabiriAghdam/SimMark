from datasets import load_dataset

train_dataset = load_dataset("ctr4si/reddit_tifu", "long", split=f"train")
val_dataset = load_dataset("ctr4si/reddit_tifu", "short", split=f"train")
train_dataset.save_to_disk("data/tifu-train")
val_dataset.save_to_disk("data/tifu-val")
