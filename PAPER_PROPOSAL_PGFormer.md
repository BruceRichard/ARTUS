# PGFormer: Physics-Guided End-to-End Articulated Generation

## 1. 研究目标
在 ArtFormer 的端到端生成框架上，融合：
- FastVGGT 的“端到端、单次前向减少误差累积”思想；
- Nadeau et al. (2025) 的“可微稳定性成本函数 + 采样/生成引导”思想；

形成适配“可动对象结构 token 生成”的新方法：**PGFormer**。

## 2. 核心创新点
1. **结构生成端到端稳定性建模（E2E Stability Regularization）**  
在 Transformer 一次前向中，对所有有效 part token 同步施加物理成本，减少逐步生成时的误差扩散。

2. **面向 articulation token 的 Nadeau-style 成本函数重构**  
将 placement 任务中的接触/穿透/鲁棒性思想，映射到父子 part 的 AABB 几何关系与关节参数约束，得到可微联合成本。

3. **锚点式关系约束（Anchor-style Relation Constraint）**  
借鉴 FastVGGT 的“保留参考 token”思想，训练时以父节点为结构锚点，显式约束子节点几何与关节参数的物理合法性，提高长链生成稳定性。

4. **FastVGGT 风格 Token 质量增强（Salient Refresh + Collapse Regularization）**  
在每层自注意力后，保留锚点 token，并对高显著 token 做 residual refresh；同时对注意力图相似度加入抑制项，缓解 token collapse，提高部件差异性与细节保真。

## 3. 成本函数（适配 ArtFormer）
对预测 token（子节点）与父节点 token，定义：

\[
\mathcal{L}_{pg} = \lambda_c \mathcal{L}_{contact}
+ \lambda_p \mathcal{L}_{penetration}
+ \lambda_a \mathcal{L}_{axis}
+ \lambda_l \mathcal{L}_{limit}
+ \lambda_o \mathcal{L}_{origin}
\]

其中：
- \(\mathcal{L}_{contact}\)：基于 AABB 外距 \(d\) 的指数接触项  
  \[
  \mathcal{L}_{contact} = \mathbb{E}[1 - \sigma(\log(1+V_p)-\log(1+V_c)) \cdot \exp(-d/d_{max})]
  \]
- \(\mathcal{L}_{penetration}\)：AABB 穿透深度惩罚；
- \(\mathcal{L}_{axis}\)：关节轴单位向量约束；
- \(\mathcal{L}_{limit}\)：关节上下界有序约束（span 非负）；
- \(\mathcal{L}_{origin}\)：关节原点位于（或接近）父包围盒内约束。

总训练目标：
\[
\mathcal{L}_{total} = \mathcal{L}_{ArtFormer} + \lambda_{pg}\mathcal{L}_{pg}
\]

## 4. 相对现有方法的预期提升
1. **相比原 ArtFormer**  
- 父子部件接触更合理、互穿更少；  
- 关节轴与限位参数更物理一致；  
- 长层级生成稳定性更高（误差累积更慢）。

2. **相比仅做后处理/仿真筛选的方法**  
- 不依赖额外仿真器在线筛选；  
- 在训练阶段直接学习“稳定结构偏好”，推理保持原流程与速度优势。

3. **相比纯几何回归损失**  
- 新增物理先验梯度，优化目标从“数值拟合”扩展到“几何-物理一致性”。

## 5. 建议实验指标
- `Valid over penetration-free`（借鉴 Nadeau）  
- Part 间碰撞率 / 穿透深度统计  
- Joint axis norm 误差、joint limit 违反率  
- Instantiation Distance / POR（ArtFormer 已有指标）

## 6. 代码落地点
- `model/Transformer/__init__.py`
  - 新增 `calculate_physics_guided_cost(...)`
  - 在 `step(...)` 中将 `pg_loss` 加入总损失并记录日志
- `configs/3_TF-Diff/text-train.yaml`
  - 新增 `loss_ratio.pg_loss`
  - 新增 `physics_guided_cost` 超参数段
