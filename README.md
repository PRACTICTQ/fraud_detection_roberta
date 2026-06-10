markdown
# 虚假通话检测与鲁棒性分析（基于 RoBERTa + Fraud-R1）

## 项目简介

本项目为《基于 Fraud-R1 方法的虚假通话检测与鲁棒性分析》课程大作业的代码与实验数据仓库。主要工作包括：

- 基于 RoBERTa 构建虚假通话二分类（诈骗/非诈骗）及多分类（7 种诈骗类型）模型
- 参考 Fraud-R1 论文设计多轮诱导策略（信任建立、紧迫感、情感操纵）及自主扩展的“隐蔽式对抗变体”
- 对测试集样本进行攻击/改写，评估模型鲁棒性
- 提供完整的实验脚本、数据集子集、性能对比表

## 目录结构
```text
fraud_detection_roberta/
├── data/ # 数据文件夹
│ ├── 训练集结果.csv # 原始训练集
│ ├── 测试集结果.csv # 原始测试集
│ ├── test_fraud_subset.csv # 抽取的500条诈骗样本
│ ├── textfooler_attacked.csv # TextFooler攻击后文本
│ ├── prompt_attacked.csv # PromptAttack三种策略后文本
│ ├── fraud_r1_standard.csv # 机制A（原生Fraud-R1）增强数据
│ └── fraud_r1_stealth.csv # 机制B（隐蔽式变体）增强数据
├── models/ # 训练好的模型（需自己训练或下载）
│ ├── roberta_binary/ # 二分类模型
│ └── roberta_multi/ # 多分类模型
├── train.py # 模型训练脚本
├── simple_textfooler.py # TextFooler攻击脚本
├── prompt_attack.py # PromptAttack脚本
├── fraud_r1_dual_generation.py # 双策略增强生成脚本（含机制A和机制B）
├── evaluate_standard.py # 评估机制A性能
├── evaluate_stealth.py # 评估机制B性能
├── requirements.txt # Python依赖列表
└── README.md # 本文件
```


## 环境配置

### 依赖安装

```bash
pip install -r requirements.txt
主要依赖：

torch (>=1.9.0)

transformers (>=4.20.0)

datasets (>=2.0.0)

pandas (>=1.3.0)

scikit-learn (>=1.0.0)

zhipuai (用于API调用)

openai (如需使用DeepSeek或OpenAI)

API 配置（如需重新生成增强数据）
本实验使用智谱 AI 的 GLM-4-Flash 模型进行文本改写。你需要：

获取 API Key

在运行 fraud_r1_dual_generation.py 前设置环境变量：
```
set ZHIPU_API_KEY=你的Key   # Windows
export ZHIPU_API_KEY=你的Key # Linux/Mac
或者直接在脚本中替换 ZHIPU_API_KEY 变量（注意不要上传到公开仓库）。

## 运行实验
### 1. 训练分类模型
```bash
python train.py
```
训练完成后，二分类模型保存在 ./models/roberta_binary，多分类模型保存在 ./models/roberta_multi。


### 2. 生成攻击/增强数据
```bash
# 生成 TextFooler 攻击数据
python simple_textfooler.py

# 生成 PromptAttack 数据
python prompt_attack.py

# 生成 Fraud-R1 双版本增强数据（机制A+机制B）
python fraud_r1_dual_generation.py
```

### 3. 评估性能
```bash
# 评估机制A（原生Fraud-R1）
python evaluate_standard.py

# 评估机制B（隐蔽式变体）
python evaluate_stealth.py
```
评估结果会打印在终端，并保存为 performance_standard.csv 和 performance_stealth.csv。

## 实验结果摘要
### 二分类模型（原始测试集）
```text
Accuracy: 96.68%

F1: 96.87%

Recall: 99.28%
```

### 鲁棒性测试（500条诈骗子集）
| 策略 | 最佳 Recall | 最差 Recall | 趋势 |
|------|-------------|-------------|------|
| 原生 Fraud-R1 | 99.8% | 99.6% | 全面上升 |
| 隐蔽式变体 | 99.8% | 97.6% | 先升后降 |

详细性能对比表见 `performance_standard.csv` 和 `performance_stealth.csv`。

## 注意事项

- 由于测试子集只包含诈骗样本，Precision 恒为 1.0，请重点关注 Recall 和 F1 的变化。
- 部分 API 调用可能因内容安全过滤失败，失败时会保留原始文本，不影响整体流程。
- 模型文件较大，未包含在仓库中。请自行运行 `train.py` 生成或联系作者获取。

## 引用

本实验参考了以下论文：
- Liu et al. RoBERTa: A Robustly Optimized BERT Pretraining Approach. EMNLP 2020.
- Yang et al. Fraud-R1: A Multi-Round Benchmark for Assessing the Robustness of LLM Against Augmented Fraud and Phishing Inducements. arXiv 2025.

## 作者与联系方式

- 课程：机器学习 / 自然语言处理
- 日期：2026年6月

## 许可证

本项目仅供课程学习使用，未经许可不得用于商业用途。