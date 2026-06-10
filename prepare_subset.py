#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从测试集中抽取诈骗样本子集，用于后续攻击/改写实验
"""

import pandas as pd
import random

# 设置随机种子，保证可复现
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# 路径配置（根据实际情况修改）
TEST_CSV = 'data/测试集结果.csv'
OUTPUT_CSV = 'data/test_fraud_subset.csv'
N_SAMPLES = 500  # 抽取数量，可根据需要调整（如 300 或 1000）

# 1. 读取测试集
print("正在读取测试集...")
df = pd.read_csv(TEST_CSV)

# 2. 筛选诈骗样本
df_fraud = df[df['is_fraud'] == True].copy()
print(f"测试集中诈骗样本总数: {len(df_fraud)}")

# 3. 随机抽样
if len(df_fraud) >= N_SAMPLES:
    df_subset = df_fraud.sample(n=N_SAMPLES, random_state=RANDOM_SEED)
else:
    print(f"警告：诈骗样本不足 {N_SAMPLES} 条，将使用全部 {len(df_fraud)} 条")
    df_subset = df_fraud

# 4. 检查是否有空对话
df_subset['specific_dialogue_content'] = df_subset['specific_dialogue_content'].fillna('')

# 5. 保存子集
df_subset.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
print(f"已保存 {len(df_subset)} 条诈骗样本到 {OUTPUT_CSV}")
print("子集中诈骗类型分布：")
print(df_subset['fraud_type'].value_counts(dropna=False))