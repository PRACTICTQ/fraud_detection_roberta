#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""评估机制A：原版 Fraud-R1 增强数据的分类性能"""
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

MODEL_PATH = "./models/roberta_binary"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
MAX_LENGTH = 256

# 指向机制A生成的文件
datasets_config = [
    ("原始子集", "data/test_fraud_subset.csv", "specific_dialogue_content"),
    ("TextFooler攻击", "data/textfooler_attacked.csv", "attacked_text"),
    ("Fraud-R1 标准版 第1轮", "data/fraud_r1_standard.csv", "round1"),
    ("Fraud-R1 标准版 第2轮", "data/fraud_r1_standard.csv", "round2"),
    ("Fraud-R1 标准版 第3轮", "data/fraud_r1_standard.csv", "round3"),
    ("PromptAttack 策略1", "data/prompt_attacked.csv", "prompt_strategy1"),
    ("PromptAttack 策略2", "data/prompt_attacked.csv", "prompt_strategy2"),
    ("PromptAttack 策略3", "data/prompt_attacked.csv", "prompt_strategy3"),
]

def load_model_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

def predict_batch(texts, tokenizer, model, batch_size=BATCH_SIZE, max_length=MAX_LENGTH):
    predictions = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_texts = [t if isinstance(t, str) and t.strip() != "" else " " for t in batch_texts]
        inputs = tokenizer(batch_texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        predictions.extend(preds)
    return predictions

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型路径不存在: {MODEL_PATH}")
        return
    tokenizer, model = load_model_tokenizer()
    print(f"模型加载成功，设备: {DEVICE}")

    df_orig = pd.read_csv("data/test_fraud_subset.csv")
    true_labels = [1] * len(df_orig)

    results = []
    for name, filepath, text_col in datasets_config:
        if not os.path.exists(filepath):
            print(f"⚠️ 跳过 {name}，文件不存在: {filepath}")
            continue
        df = pd.read_csv(filepath)
        if text_col not in df.columns:
            print(f"⚠️ 跳过 {name}，缺少列 {text_col}")
            continue
        texts = df[text_col].fillna("").tolist()
        if len(texts) != len(true_labels):
            print(f"⚠️ 跳过 {name}，样本数不一致")
            continue
        preds = predict_batch(texts, tokenizer, model)
        acc = accuracy_score(true_labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average="binary")
        results.append({"Dataset": name, "Accuracy": f"{acc:.4f}", "Precision": f"{precision:.4f}",
                        "Recall": f"{recall:.4f}", "F1": f"{f1:.4f}"})
        print(f"已完成: {name}")

    print("\n" + "="*60)
    print("性能对比表 - 机制A（原版 Fraud-R1）")
    print("="*60)
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    df_results.to_csv("performance_standard.csv", index=False, encoding="utf-8-sig")
    print("\n结果已保存至 performance_standard.csv")

if __name__ == "__main__":
    main()

