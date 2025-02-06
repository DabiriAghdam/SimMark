from datasets import load_dataset, Dataset

train_streaming_dataset = load_dataset("allenai/c4", "realnewslike", streaming=True, split="train")
val_streaming_dataset = load_dataset("allenai/c4", "realnewslike", streaming=True, split="validation")
train_texts = list()
i = 0
for text in train_streaming_dataset:
    train_texts.append(text['text'])
    i += 1
    if (i >= 8000): break

val_texts = list()
i = 0
for text in val_streaming_dataset:
    val_texts.append(text['text'])
    i += 1
    if (i >= 1000): break
train_dataset = Dataset.from_dict({'text': train_texts}).save_to_disk("data/c4-train")
val_dataset = Dataset.from_dict({'text': val_texts}).save_to_disk("data/c4-val")