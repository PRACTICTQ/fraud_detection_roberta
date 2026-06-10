#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
三种 PromptAttack 策略：
1. 高风险词替换为无害同义词
2. 添加中性填充词并调整句式
3. 改用乐于助人、协商的语气
"""

import pandas as pd
import random

INPUT_CSV = "data/test_fraud_subset.csv"
OUTPUT_CSV = "data/prompt_attacked.csv"

# 策略1：高风险词替换映射
RISK_REPLACE = {
    '转账': '资金划拨',
    '验证码': '安全码',
    '贷款': '资金周转',
    '客服': '服务专员',
    '退款': '资金返还',
    '安全账户': '保障账户',
    '立即': '请尽快',
    '必须': '建议您',
    '逾期': '超时',
    '冻结': '暂停',
    '扣款': '划款',
}

# 策略2：中性填充词
FILLERS = ['哦', '嗯', '是这样的', '那个', '其实', '毕竟', '可以说', '那么', '另外']

# 策略3：语气转换规则
TONE_RULES = [
    ('你必须', '您方便的话可以'),
    ('快点', '麻烦您尽快'),
    ('否则', '如果不处理可能会'),
    ('立即', '请您考虑'),
    ('马上', '请尽快'),
    ('如果不', '若未'),
    ('后果自负', '可能会影响您的权益'),
]

def strategy1(text):
    """无害同义词替换"""
    for k, v in RISK_REPLACE.items():
        text = text.replace(k, v)
    return text

def strategy2(text):
    """添加中性填充词并调整句式"""
    sentences = text.split('。')
    new_sents = []
    for sent in sentences:
        if sent.strip():
            filler = random.choice(FILLERS)
            # 随机决定填充词位置（句首或句中）
            if random.random() < 0.6:
                new_sents.append(f"{filler}，{sent}")
            else:
                # 将填充词插入到第一个逗号后或句末（简单示例）
                if '，' in sent:
                    parts = sent.split('，', 1)
                    new_sents.append(f"{parts[0]}，{filler}，{parts[1]}")
                else:
                    new_sents.append(f"{sent}，{filler}")
    result = '。'.join(new_sents)
    # 如果拆分后变成空，则返回原文本
    return result if result.strip() else text

def strategy3(text):
    """改用乐于助人、协商的语气"""
    for old, new in TONE_RULES:
        text = text.replace(old, new)
    # 添加礼貌收尾
    if not text.endswith('。') and not text.endswith('！'):
        text += '。'
    if random.random() < 0.7:
        text += ' 非常感谢您的配合。'
    return text

def main():
    df = pd.read_csv(INPUT_CSV)
    texts = df["specific_dialogue_content"].tolist()

    df["prompt_strategy1"] = [strategy1(t) for t in texts]
    df["prompt_strategy2"] = [strategy2(t) for t in texts]
    df["prompt_strategy3"] = [strategy3(t) for t in texts]

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ PromptAttack 完成，共处理 {len(texts)} 条对话。")
    print(f"结果保存至 {OUTPUT_CSV}")

if __name__ == "__main__":
    main()