#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RoBERTa 在欺诈通话数据集上的训练脚本
支持二分类 (is_fraud) 和多分类 (fraud_type)
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
MODEL_NAME = 'roberta-base'       # 英文用 roberta-base，中英文混合可用 'xlm-roberta-base'
MAX_LENGTH = 256
BATCH_SIZE = 2
EPOCHS = 1
LEARNING_RATE = 2e-5
TEST_SIZE = 0.2                   # 如果不分开训练/测试文件，按此比例划分
RANDOM_SEED = 42

# 数据文件路径（根据实际情况修改）
TRAIN_CSV = 'data/训练集结果.csv'
TEST_CSV = 'data/测试集结果.csv'

# 输出目录
OUTPUT_DIR_BINARY = './models/roberta_binary'
OUTPUT_DIR_MULTI = './models/roberta_multi'

# ==================== 加载数据 ====================
print("=" * 60)
print("加载数据...")

# 检查文件是否存在
if os.path.exists(TRAIN_CSV) and os.path.exists(TEST_CSV):
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    print(f"使用独立训练集和测试集")
    print(f"训练集大小: {len(train_df)}, 测试集大小: {len(test_df)}")
else:
    # 如果没有分开的文件，则使用单个 CSV 并划分
    print("未找到独立的训练/测试文件，尝试使用单个 CSV 文件...")
    single_csv = 'data/测试集结果.csv'  # 或 训练集结果.csv
    if os.path.exists(single_csv):
        df = pd.read_csv(single_csv)
        print(f"从 {single_csv} 加载数据，大小: {len(df)}")
        # 按比例划分
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=df['is_fraud'])
        print(f"按 {1-TEST_SIZE}/{TEST_SIZE} 比例划分为训练集和测试集")
    else:
        raise FileNotFoundError("找不到数据文件，请检查 data/ 目录")

# 检查必要列
required_cols = ['specific_dialogue_content', 'is_fraud']
for col in required_cols:
    if col not in train_df.columns:
        raise ValueError(f"训练集缺少列: {col}")

# 填充空值
train_df['specific_dialogue_content'] = train_df['specific_dialogue_content'].fillna('')
test_df['specific_dialogue_content'] = test_df['specific_dialogue_content'].fillna('')
train_df['is_fraud'] = train_df['is_fraud'].fillna(False).astype(bool)
test_df['is_fraud'] = test_df['is_fraud'].fillna(False).astype(bool)

print(f"\n训练集: 诈骗={train_df['is_fraud'].sum()}, 非诈骗={len(train_df)-train_df['is_fraud'].sum()}")
print(f"测试集: 诈骗={test_df['is_fraud'].sum()}, 非诈骗={len(test_df)-test_df['is_fraud'].sum()}")

# ==================== 二分类任务 ====================
print("\n" + "=" * 60)
print("【任务1：二分类】is_fraud (诈骗/非诈骗)")

train_df['label_binary'] = train_df['is_fraud'].astype(int)
test_df['label_binary'] = test_df['is_fraud'].astype(int)

# 转换为 HuggingFace Dataset
train_dataset_binary = Dataset.from_pandas(train_df[['specific_dialogue_content', 'label_binary']])
test_dataset_binary = Dataset.from_pandas(test_df[['specific_dialogue_content', 'label_binary']])

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples['specific_dialogue_content'],
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH
    )

train_tokenized = train_dataset_binary.map(tokenize_function, batched=True)
test_tokenized = test_dataset_binary.map(tokenize_function, batched=True)

# 移除原始文本列，保留 labels
train_tokenized = train_tokenized.remove_columns(['specific_dialogue_content'])
test_tokenized = test_tokenized.remove_columns(['specific_dialogue_content'])
train_tokenized = train_tokenized.rename_column('label_binary', 'labels')
test_tokenized = test_tokenized.rename_column('label_binary', 'labels')
train_tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
test_tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# 定义评估指标
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# 模型
model_binary = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# 训练参数
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR_BINARY,
    eval_strategy='epoch',
    save_strategy='epoch',
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    logging_dir='./logs_binary',
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    save_total_limit=2,
    seed=RANDOM_SEED,
)

trainer_binary = Trainer(
    model=model_binary,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# 训练
trainer_binary.train()

# 最终评估
print("\n【二分类最终结果】")
eval_result = trainer_binary.evaluate()
for key, value in eval_result.items():
    print(f"{key}: {value:.4f}")

# 保存模型
trainer_binary.save_model(OUTPUT_DIR_BINARY)
tokenizer.save_pretrained(OUTPUT_DIR_BINARY)
print(f"二分类模型已保存到 {OUTPUT_DIR_BINARY}")

# ==================== 多分类任务（可选） ====================
if 'fraud_type' in train_df.columns and 'fraud_type' in test_df.columns:
    print("\n" + "=" * 60)
    print("【任务2：多分类】fraud_type (诈骗类型)")

    # 过滤掉空值
    train_multi = train_df[train_df['fraud_type'].notna()].copy()
    test_multi = test_df[test_df['fraud_type'].notna()].copy()

    if len(train_multi) > 0 and len(test_multi) > 0:
        # 标签编码
        le = LabelEncoder()
        train_multi['label_multi'] = le.fit_transform(train_multi['fraud_type'])
        test_multi['label_multi'] = test_multi['fraud_type'].apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )
        # 过滤掉测试集中新出现的类别
        test_multi = test_multi[test_multi['label_multi'] != -1]
        num_classes = len(le.classes_)
        print(f"诈骗类型类别数: {num_classes}")
        print(f"类别映射: {dict(zip(le.classes_, range(num_classes)))}")

        # 创建数据集
        train_dataset_multi = Dataset.from_pandas(train_multi[['specific_dialogue_content', 'label_multi']])
        test_dataset_multi = Dataset.from_pandas(test_multi[['specific_dialogue_content', 'label_multi']])

        train_tokenized_multi = train_dataset_multi.map(tokenize_function, batched=True)
        test_tokenized_multi = test_dataset_multi.map(tokenize_function, batched=True)

        train_tokenized_multi = train_tokenized_multi.remove_columns(['specific_dialogue_content'])
        test_tokenized_multi = test_tokenized_multi.remove_columns(['specific_dialogue_content'])
        train_tokenized_multi = train_tokenized_multi.rename_column('label_multi', 'labels')
        test_tokenized_multi = test_tokenized_multi.rename_column('label_multi', 'labels')
        train_tokenized_multi.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        test_tokenized_multi.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

        # 多分类评估函数
        def compute_metrics_multi(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
            acc = accuracy_score(labels, predictions)
            return {'accuracy': acc, 'f1_macro': f1, 'precision_macro': precision, 'recall_macro': recall}

        model_multi = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_classes)

        training_args_multi = TrainingArguments(
            output_dir=OUTPUT_DIR_MULTI,
            eval_strategy='epoch',
            save_strategy='epoch',
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=EPOCHS,
            weight_decay=0.01,
            logging_dir='./logs_multi',
            logging_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model='f1_macro',
            save_total_limit=2,
            seed=RANDOM_SEED,
        )

        trainer_multi = Trainer(
            model=model_multi,
            args=training_args_multi,
            train_dataset=train_tokenized_multi,
            eval_dataset=test_tokenized_multi,
            compute_metrics=compute_metrics_multi,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        trainer_multi.train()

        print("\n【多分类最终结果】")
        eval_result_multi = trainer_multi.evaluate()
        for key, value in eval_result_multi.items():
            print(f"{key}: {value:.4f}")

        trainer_multi.save_model(OUTPUT_DIR_MULTI)
        tokenizer.save_pretrained(OUTPUT_DIR_MULTI)
        print(f"多分类模型已保存到 {OUTPUT_DIR_MULTI}")
    else:
        print("多分类数据不足，跳过多分类任务")
else:
    print("\n数据中缺少 'fraud_type' 列，跳过多分类任务")

print("\n" + "=" * 60)
print("训练完成！")