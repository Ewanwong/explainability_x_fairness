## Datasets 
Toxicity detection datasets: civil comments, hatexplain, tbd

## Models: 
encoder models: bert, roberta, distilbert
decoder models: llama3.2-3B, Qwen2.5-3B

## Bias Correlation
收集每一个prediction的individual fairness score和reliance score，计算correlation
加入debiased models？

问题：bias是否取绝对值？reliance是否取绝对值？是否按group/prediction分开算correlation？汇报哪一个的correlation/取平均值？
Analysis：是否受到fair washing影响: 加入bias mitigated model

## Model Selection
根据一些model在一小部分validation set上的reliance结果判断哪个更fair
Dimensions: 在同一数据集训练的不同model；在不同数据集上训练的相同model（在该数据集的test set上）；在同一数据集上的同一模型使用不同的debiasing方法
debiasing方法
pre-processing: group balance, group class balance, cda
in-processing: dropout, causal debias, TBD
对比：random ranking和用validation set上的fairness结果

问题：选什么debiasing模型？不同数据集上的model在哪个test set比较？为什么不直接按validation的fairness作指标？decoder模型怎么debias （prompts）？
Analysis: 是否受到fair washing影响: 加入bias mitigated model

## Bias Mitigation
训练时减少模型对sensitive feature的reliance，看是否更fair

问题：用什么regularizer
Analysis：是否受到bias correlation影响

## Analysis
1. vocabulary的影响：同时替换fairness和reliance使用的vocabulary
2. Bias correlation的影响：在reliance时用小的vocabulary，在fairness时用大的vocabulary
3. 与faithfulness关系
