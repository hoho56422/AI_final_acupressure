# import os
# import json
# import pandas as pd
# from datasets import Dataset, DatasetDict
# import evaluate
# from transformers import (
#     AutoTokenizer,
#     AutoModelForSequenceClassification,
#     TrainingArguments,
#     Trainer,
#     EarlyStoppingCallback
# )
# from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
# import numpy as np
# import torch
# import random
# import matplotlib.pyplot as plt
# from sklearn.utils.class_weight import compute_class_weight


# os.makedirs("./trained_model/tmp_output", exist_ok=True)
# os.makedirs("./trained_model/logs", exist_ok=True)
# # === 1. 設定隨機種子確保可重現性 ===
# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

# set_seed(42)

# # === 2. 讀取 JSON 資料 ===
# with open("sympton_dataset.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# df = pd.DataFrame(data)

# # === 3. Label 編碼 ===
# label_encoder = LabelEncoder()
# df["label_id"] = label_encoder.fit_transform(df["label"])
# label2id = {label: int(i) for label, i in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}
# id2label = {int(i): label for label, i in label2id.items()}

# # === 4. 計算類別權重處理不平衡資料 ===
# class_weights = compute_class_weight(
#     "balanced",
#     classes=np.unique(df["label_id"]),
#     y=df["label_id"]
# )
# class_weights = torch.tensor(class_weights, dtype=torch.float32)

# # === 5. 分層分割資料集 ===
# sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
# train_idx, temp_idx = next(sss.split(df["text"], df["label_id"]))
# train_df = df.iloc[train_idx]
# temp_df = df.iloc[temp_idx]

# sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
# valid_idx, test_idx = next(sss.split(temp_df["text"], temp_df["label_id"]))
# valid_df = temp_df.iloc[valid_idx]
# test_df = temp_df.iloc[test_idx]

# datasets = DatasetDict({
#     "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
#     "validation": Dataset.from_pandas(valid_df.reset_index(drop=True)),
#     "test": Dataset.from_pandas(test_df.reset_index(drop=True))
# })

# # === 6. Tokenizer 與預處理 ===
# model_name = "./bert-base-chinese"
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# def preprocess(example):
#     return tokenizer(
#         example["text"],
#         truncation=True,
#         padding="max_length",
#         max_length=128,
#         return_tensors="pt"
#     )

# datasets = datasets.map(preprocess, batched=True)
# datasets = datasets.rename_column("label_id", "labels")
# datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# # === 7. 模型定義 ===
# model = AutoModelForSequenceClassification.from_pretrained(
#     model_name,
#     num_labels=len(label2id),
#     id2label=id2label,
#     label2id=label2id,
#     problem_type="single_label_classification"
# )

# # === 8. 自定義加權 Trainer 處理類別不平衡 ===
# class WeightedTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         labels = inputs.pop("labels")
#         outputs = model(**inputs)
#         logits = outputs.logits
#         loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
#         loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
#         return (loss, outputs) if return_outputs else loss

# # === 9. 評估指標 ===
# accuracy_metric = evaluate.load("accuracy")
# f1_metric = evaluate.load("f1")

# def compute_metrics(p):
#     preds = np.argmax(p.predictions, axis=1)
#     labels = p.label_ids
#     acc = accuracy_metric.compute(predictions=preds, references=labels)
#     f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")
#     precision = precision_score(labels, preds, average="macro")
#     recall = recall_score(labels, preds, average="macro")
#     return {
#         "accuracy": acc["accuracy"],
#         "f1": f1["f1"],
#         "precision": precision,
#         "recall": recall
#     }

# # === 10. 訓練參數設定 ===
# training_args = TrainingArguments(
#     output_dir="./trained_model/tmp_output",
#     logging_dir="./trained_model/logs",
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     logging_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=64,
#     num_train_epochs=10,
#     weight_decay=0.01,
#     load_best_model_at_end=True,
#     metric_for_best_model="f1",
#     save_total_limit=2,
#     fp16=torch.cuda.is_available(),
#     warmup_ratio=0.1,
#     gradient_accumulation_steps=2,
#     report_to=["tensorboard"],
# )

# # === 11. 初始化 Trainer ===
# trainer = WeightedTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=datasets["train"],
#     eval_dataset=datasets["validation"],
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics,
#     callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
# )

# # === 12. 開始訓練 ===
# trainer.train()

# # === 13. 儲存模型與相關檔案 ===
# trainer.save_model("./trained_model")
# tokenizer.save_pretrained("./trained_model")

# # 儲存 label mapping
# with open("./trained_model/label_map.json", "w", encoding="utf-8") as f:
#     json.dump({
#         "id2label": id2label,
#         "label2id": label2id,
#         "label_encoder_classes": label_encoder.classes_.tolist()
#     }, f, ensure_ascii=False, indent=2)

# # === 14. 測試集評估 ===
# test_results = trainer.evaluate(eval_dataset=datasets["test"])

# # 儲存測試結果
# with open("./trained_model/test_results.json", "w", encoding="utf-8") as f:
#     json.dump(test_results, f, ensure_ascii=False, indent=2)

# # 生成並儲存混淆矩陣
# test_pred = trainer.predict(datasets["test"])
# preds = np.argmax(test_pred.predictions, axis=1)
# labels = test_pred.label_ids

# cm = confusion_matrix(labels, preds)
# disp = ConfusionMatrixDisplay(
#     confusion_matrix=cm,
#     display_labels=label_encoder.classes_
# )
# disp.plot(xticks_rotation="vertical")
# plt.tight_layout()
# plt.savefig("./trained_model/confusion_matrix.png", dpi=300, bbox_inches="tight")

# print("\n=== 訓練完成 ===")
# print("測試集結果:")
# print(f"準確率: {test_results['eval_accuracy']:.4f}")
# print(f"F1 分數: {test_results['eval_f1']:.4f}")
# print(f"精確率: {test_results['eval_precision']:.4f}")
# print(f"召回率: {test_results['eval_recall']:.4f}")
# print(f"混淆矩陣已保存至: ./trained_model/confusion_matrix.png")