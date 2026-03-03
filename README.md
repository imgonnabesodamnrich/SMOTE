# 论文笔记——基于可学习特征空间的数据增强方法 (Feature-level SMOTE)

> **Paper Title:** Feature-level SMOTE: Augmenting fault samples in learnable feature space for imbalanced fault diagnosis of gas turbines
>
> **Journal:** Expert Systems With Applications (Elsevier), Vol. 238, 2024
>
> **DOI:** [10.1016/j.eswa.2023.122023](https://doi.org/10.1016/j.eswa.2023.122023)
> 
[![DOI](https://img.shields.io/badge/DOI-10.1016/j.eswa.2023.122023-blue)](https://doi.org/10.1016/j.eswa.2023.122023)
![Status](https://img.shields.io/badge/Status-Published-green)
![Topic](https://img.shields.io/badge/Topic-Imbalanced%20Learning-orange)

---
## 目录

1. [背景](#1-背景)
2. [方法论：Feature-level SMOTE](#2-方法论feature-level-smote)
   - 2.1 [空间映射 (Space Mapping)](#21-空间映射-space-mapping)
   - 2.2 [特征级数据增强 (Feature-level Data Augmentation)](#22-特征级数据增强-feature-level-data-augmentation)
   - 2.3 [故障分类](#23-故障分类)
3. [实验验证](#3-实验验证)
4. [结论](#4-结论)
5. [文献来源](#5-文献来源)

---

## 1. 背景

在工业设备（如燃气轮机）的故障诊断中，深度学习模型的训练往往面临两个主要挑战：

1.  **类别不平衡 (Class Imbalance)**：实际运行中，采集到的正常样本数量远多于故障样本。模型容易倾向于将所有样本预测为“正常”，从而忽略少数的故障样本。
2.  **类间重叠 (Inter-class Overlap)**：早期故障的数据特征与正常运行状态非常相似。在原始数据空间中，故障样本与正常样本往往混杂在一起，难以区分。

传统的数据增强方法（如SMOTE及其变体）通常直接在原始数据空间生成合成样本。如果原始空间中两类数据本身重叠严重，SMOTE会在重叠区域生成更多的噪声样本，反而模糊了分类边界，导致诊断性能下降。

针对上述问题，论文提出了一种名为 **Feature-level SMOTE** 的框架。该方法的核心思想是：**先将数据映射到一个类间分离度更高的特征空间，再在该空间内进行数据增强。**

<div align="center">
  <img width="70%" src="https://github.com/user-attachments/assets/15b46f8f-5b6a-409b-baaa-1c8bc9bf8f6d" />
  <p><em>图1 (a)原始空间样本分布 (b)原始空间数据增强后 (c)DSMHSA学习后的特征空间样本分布 (d)特征空间数据增强后 </em></p>
</div>

## 2. 方法论：Feature-level SMOTE

该框架主要包含三个步骤：空间映射、特征级数据增强、故障分类。

### 2.1 空间映射 (Space Mapping)

<div align="center">
  <img width="70%" src="https://github.com/user-attachments/assets/0930b017-d250-4149-a952-98fcc3ec5933" />
  <p><em>图2 多头自注意力机制 (MHSANet) 的内部架构示意图 </em></p>
</div>

为了解决类间重叠问题，首先需要构建一个易于区分不同类别的特征空间。论文采用了 **深度孪生多头自注意力网络 (DSMHSA)** 作为特征提取器。

*   **网络结构**：基于Transformer的多头自注意力机制 (Multi-head Self-attention)，能够有效提取多维时间序列中的长距离依赖关系。
*   **训练策略**：使用孪生网络 (Siamese Network) 结构，并配合 **对比损失函数 (Contrastive Loss)** 进行训练。
    *   当输入的一对样本属于同一类时，损失函数促使它们的距离变小。
    *   当输入的一对样本属于不同类时，损失函数促使它们的距离变大（超过设定的阈值）。

经过这一步处理，原始数据被映射到了一个新的“可学习特征空间”。在这个空间中，同类样本更加聚集，异类样本（正常与故障）被拉开距离，重叠程度显著降低。

### 2.2 特征级数据增强 (Feature-level Data Augmentation)

在获得类间分离度良好的特征空间后，使用 **SMOTE (Synthetic Minority Over-sampling Technique)** 算法对故障样本（少数类）进行过采样。

*   **操作对象**：DSMHSA输出的潜在特征向量 (Latent Feature Representation)，而非原始的时间序列信号。
*   **生成过程**：对于每个故障特征向量，计算其K近邻，在连线上进行线性插值生成新的合成特征。
*   **优势**：由于特征空间中的安全区域（Safe Region）更大，生成的合成样本不再处于类间重叠区，从而避免了生成噪声数据，保证了合成样本的质量。

### 2.3 故障分类

最后，将增强后的平衡特征数据集输入到分类器中。论文使用了简单的 Softmax 分类器 (SMC) 进行训练和测试，完成最终的故障诊断任务。

## 3. 实验验证

<div align="center">
  <img width="70%" src="https://github.com/user-attachments/assets/1001bba4-ffe1-4634-ac33-19a9783c7cb5" />
  <p><em>图3 (a)燃气轮机性能数据采集流程 (b)基于滑动窗口的样本构建方法 </em></p>
</div>

为了验证方法的有效性，在两个数据集上进行了测试：

1.  **某亚洲航空公司燃气轮机实际数据集**：包含正常样本、EGT传感器指示故障、TSIF传感器指示故障。数据极度不平衡且重叠严重。
2.  **机器人执行失败公共数据集 (Robot Execution Failures)**。

**实验结果摘要：**

*   在燃气轮机数据集上，Feature-level SMOTE 的平均平衡准确率 (Balanced Accuracy) 达到 **90.38%**。
*   **对比分析**：与基准方法（无增强）以及7种主流数据增强方法（如标准SMOTE, Borderline-SMOTE, ADASYN等）相比，该方法在准确率、召回率和AUC指标上均取得了显著提升。
*   **可视化分析**：通过t-SNE可视化可以看到，经过DSMHSA映射后，原本混杂的正常与故障样本被有效分离，验证了“先分离再增强”策略的合理性。

## 4. 结论

Feature-level SMOTE 为处理高重叠、不平衡时间序列分类问题提供了一种方案。其关键贡献在于指出了在数据增强前进行特征空间优化的必要性。通过结合对比学习与传统过采样技术，该方法能够显著提升工业设备在小样本故障条件下的诊断能力。

## 5. 文献来源
**Paper Title:** Feature-level SMOTE: Augmenting fault samples in learnable feature space for imbalanced fault diagnosis of gas turbines

**Journal:** Expert Systems With Applications (Elsevier), Vol. 238, 2024

**DOI:** [10.1016/j.eswa.2023.122023](https://doi.org/10.1016/j.eswa.2023.122023)
