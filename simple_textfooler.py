#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自定义 TextFooler 风格攻击：基于同义词映射表替换关键词
"""

import pandas as pd
import random

# 同义词映射表（可根据需要扩展）
SYNONYM_MAP = {
    '资金': ['经费', '款项', '资产', '资本'],
    '链接': ['联结', '网址', '链接地址', '超链接'],
    '贷款': ['借贷', '借款', '融资', '信贷'],
    '产品': ['物品', '商品', '货物', '货品'],
    '信息': ['讯息', '资料', '数据', '情报'],
    '抵押': ['典押', '担保', '质押', '抵押品'],
    '手机': ['手提电话', '移动电话', '手机设备'],
    '客户': ['主顾', '顾客', '用户', '客户方'],
    '客服': ['客户服务', '客服中心', '服务专员'],
    '验证码': ['校验码', '安全码', '动态码', '短信码'],
    '转账': ['划转', '汇款', '资金划拨', '转出'],
    '退款': ['返款', '资金返还', '退还款项'],
    '安全账户': ['保障账户', '监管账户', '托管账户'],
    '系统': ['制度', '机制', '体系'],
    '操作': ['作业', '处理', '执行', '办理'],
    '解决方案': ['回答方案', '处理方案', '应对措施'],
    '投资': ['投入资金', '资本运作', '理财'],
    '风险': ['隐患', '不确定性', '潜在损失'],
    '安全': ['保险', '稳妥', '保障', '防护'],
    '提醒': ['提示', '温馨提示', '使想起'],
    '办理': ['管理', '处理', '经手'],
    '为了确保': ['为了保证', '为保障', '为确认'],
    '需要': ['没有', '不必', '无需'],  # 注意：这可能导致语义反转，但保留原案例风格
    '应用程序': ['适用程序'],
    '使发生联系': ['使发生接触', '连接'],
    '进口': ['输入'],
    '保卫': ['安全'],
    '解答': ['处理'],
    '短信': ['讯息'],
}

def replace_synonyms(text, prob=0.3):
    """
    对文本中的关键词进行同义词替换
    prob: 每个匹配词被替换的概率（避免过度替换）
    """
    for word, syn_list in SYNONYM_MAP.items():
        if word in text and random.random() < prob:
            new_word = random.choice(syn_list)
            text = text.replace(word, new_word, 1)  # 每词最多替换一次
    return text

def main():
    INPUT_CSV = "data/test_fraud_subset.csv"
    OUTPUT_CSV = "data/textfooler_attacked.csv"

    df = pd.read_csv(INPUT_CSV)
    texts = df["specific_dialogue_content"].tolist()

    attacked_texts = [replace_synonyms(t) for t in texts]

    df["attacked_text"] = attacked_texts
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"✅ TextFooler 攻击完成，共处理 {len(texts)} 条对话。")
    print(f"结果保存至 {OUTPUT_CSV}")

if __name__ == "__main__":
    main()