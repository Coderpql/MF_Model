# MF_Model

本仓库包含基于矩阵分解的推荐算法，主要为传统矩阵分解：FunkSVD，UBV。以及异质评分跨域矩阵分解：TMF，TCF，ITCF。

## Funk_SVD

矩阵分解算法

设评分矩阵为$R$，用户隐特征矩阵为$P$，物品隐特征矩阵为$Q$，则$R=Q^T P$，用户$u$对商品$i$的评分为$r_{ui}$，那么UV分解算法的目标函数如下：

$\min _{q^{*}, p^{*}} \sum_{(u, i) \in \kappa}\left(r_{u i}-q_{i}^{T} p_{u}\right)^{2}+\lambda\left(\left\|q_{i}\right\|^{2}+\left\|p_{u}\right\|^{2}\right)$

相关更新公式如下：

$e_{u i}=r_{u i}-q_{i}^{T} p_{u}$

$q_{i} \leftarrow q_{i}+\gamma\left(e_{u i} p_{u}-\lambda q_{i}\right)$

$p_{u} \leftarrow p_{u}+\gamma\left(e_{u i} q_{i}-\lambda p_{u}\right)$

## UBV

矩阵三因子分解

设评分矩阵为$R$，用户隐特征矩阵为$U$，物品隐特征矩阵为$V$，则$R=UBV^T$，用户$u$对商品$i$的评分为$r_{ui}$，那么UBV分解算法的目标函数如下：

$\min _{U \geq 0, V \geq 0, B \geq 0} T=\left\|M-U B V^{t}\right\|_{F}^{2} \\
\text { s.t. } \quad U^{t} U=\mathrm{I}, V^{t} V=\mathrm{I}$

相关更新公式如下：

$V_{i j}  \leftarrow V_{i j} \sqrt{\frac{\left(M^{t} U B\right)_{i j}}{\left(V V^{t} M^{t} U B\right)_{i j}}}$

$U_{i j}  \leftarrow U_{i j} \sqrt{\frac{\left(M V B^{t}\right)_{i j}}{\left(U U^{t} M V B^{t}\right)_{i j}}}$

$B_{i j}  \leftarrow B_{i j} \sqrt{\frac{\left(U^{t} M V\right)_{i j}}{\left(U^{t} U B V^{t} V\right)_{i j}}}$

## TMF

参考文献: Mixed factorization for collaborative recommendation with heterogeneous explicit feedbacks

## TCF

参考文献: Transfer Learning to Predict Missing Ratings via Heterogeneous User Feedbacks

## ITCF

参考文献: Interaction-Rich Transfer Learning for Collaborative Filtering with Heterogeneous User Feedback

