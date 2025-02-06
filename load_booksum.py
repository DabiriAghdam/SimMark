from datasets import load_dataset

train_dataset = load_dataset("kmfoda/booksum", split=f"train")
val_dataset = load_dataset("kmfoda/booksum", split=f"validation")
train_dataset.save_to_disk("data/booksum-train")
val_dataset.save_to_disk("data/booksum-val")
