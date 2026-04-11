# Flow Matching 方案1 使用指南

## 📋 方案概述

**方案1: 双损失混合训练 (Dual Loss Hybrid Training)**

这是一个非侵入式的Flow Matching集成方案，通过配置文件即可启用，无需修改训练代码。

### ✨ 核心特点

1. **完全向后兼容** - 不启用时与原始DDPM完全相同
2. **配置驱动** - 通过YAML配置文件控制所有参数
3. **零代码修改** - 训练脚本无需任何改动
4. **灵活权重** - 可动态调整DDPM和Flow Matching的损失权重

---

## 🚀 快速开始

### 步骤1: 使用新配置文件

```bash
# 使用Flow Matching配置训练
python 2_train_diff.py --config configs/2_Diff/train_with_flow_matching_scheme1.yaml
```

### 步骤2: 查看训练日志

训练开始时会看到：
```
[INFO] Using Flow Matching Scheme 1 (Dual Loss Hybrid Training)
```

如果使用原始配置：
```
[INFO] Using standard DDPM training
```

---

## ⚙️ 配置参数详解

### Flow Matching配置块

```yaml
flow_matching:
  enabled: true                    # 是否启用Flow Matching
  loss_weight: 0.3                 # Flow Matching损失权重
  flow_type: 'optimal_transport'   # Flow类型
  objective: 'pred_velocity'       # 预测目标
  loss_type: 'mse'                 # 损失函数类型
```

### 参数说明

#### `enabled` (bool)
- `true`: 启用Flow Matching，使用双损失训练
- `false`: 禁用Flow Matching，使用标准DDPM训练

#### `loss_weight` (float, 0.0-1.0)
- Flow Matching损失的权重
- 总损失 = `(1 - loss_weight) * DDPM_loss + loss_weight * FM_loss`
- **推荐值**:
  - `0.3`: 保守策略，主要使用DDPM (70% DDPM + 30% FM)
  - `0.5`: 平衡策略，两者权重相同
  - `0.7`: 激进策略，主要使用Flow Matching

#### `flow_type` (string)
- `'optimal_transport'`: 最优传输路径 (推荐)
  - 路径: `x_t = t * x_1 + (1-t) * x_0`
  - 速度: `v_t = x_1 - x_0`
  - 特点: 简单、稳定、采样快
  
- `'conditional'`: 条件流匹配
  - 路径: `x_t = t * x_1 + σ_t * noise`
  - 速度: `v_t = x_1 - (1-σ_min) * noise`
  - 特点: 更灵活，但稍复杂

#### `objective` (string)
- `'pred_velocity'`: 预测速度场 (推荐用于Flow Matching)
- `'pred_x0'`: 预测干净数据 (与DDPM一致)

#### `loss_type` (string)
- `'mse'`: 均方误差损失 (推荐)
- `'l1'`: L1损失

---

## 📊 推荐配置方案

### 方案A: 保守混合 (推荐新手)

```yaml
flow_matching:
  enabled: true
  loss_weight: 0.2              # 20% FM + 80% DDPM
  flow_type: 'optimal_transport'
  objective: 'pred_velocity'
  loss_type: 'mse'
```

**适用场景**: 
- 首次尝试Flow Matching
- 希望保持DDPM的稳定性
- 逐步过渡到Flow Matching

### 方案B: 平衡混合 (推荐)

```yaml
flow_matching:
  enabled: true
  loss_weight: 0.5              # 50% FM + 50% DDPM
  flow_type: 'optimal_transport'
  objective: 'pred_velocity'
  loss_type: 'mse'
```

**适用场景**:
- 希望同时利用两种方法的优势
- 追求更好的泛化能力
- 平衡训练稳定性和采样速度

### 方案C: 激进Flow Matching

```yaml
flow_matching:
  enabled: true
  loss_weight: 0.7              # 70% FM + 30% DDPM
  flow_type: 'optimal_transport'
  objective: 'pred_velocity'
  loss_type: 'mse'
```

**适用场景**:
- 追求最快的采样速度
- 已有稳定的训练流程
- 愿意接受一定的训练不稳定性

---

## 🔍 维度对齐说明

所有维度与原始DDPM完全一致，无需担心维度不匹配：

```python
# 输入维度
x_start: [B, D]      # B=batch_size, D=768 (dim_latentcode)
t: [B]               # 时间步 [0, 1000)
cond: {
    'z_hat': [B, N, D],      # N=z_hat序列长度
    'text': [B, M, D]        # M=text序列长度
}

# 输出维度
loss: scalar         # 标量损失
model_out: [B, D]    # 预测输出
```

---

## 📈 训练监控

### WandB日志

训练时会记录以下指标（与原始DDPM相同）：

```python
{
    'loss': total_loss,              # 总损失
    'vq_loss': vq_loss,              # VQ损失
    'diff_loss_1': diff_loss_1,      # Diffusion损失
    'diff_100_loss_1': ...,          # t<100的损失
    'diff_1000_loss_1': ...,         # t>100的损失
    'z_KL': z_KL,                    # KL散度
    'z_perplexity': z_perplexity,    # 困惑度
}
```

### 如何判断训练是否正常

1. **损失下降**: `loss`应该稳定下降
2. **损失比例**: 如果`loss_weight=0.3`，则FM贡献约30%的损失
3. **生成质量**: 定期检查生成的mesh质量

---

## 🎯 常见问题

### Q1: 启用Flow Matching后训练变慢了？

**A**: 是的，训练时间会增加约30-50%，因为需要计算两种损失。但采样速度会显著提升。

### Q2: 可以中途切换配置吗？

**A**: 可以！从checkpoint恢复训练时，可以修改`flow_matching`配置。

### Q3: 如何禁用Flow Matching？

**A**: 设置`flow_matching.enabled: false`或使用原始配置文件。

### Q4: 损失权重如何选择？

**A**: 建议从0.2开始，逐步增加到0.5。观察生成质量和训练稳定性。

---

## 🔧 故障排除

### 问题1: 训练不稳定

**解决方案**:
- 降低`loss_weight`到0.2或0.1
- 使用`flow_type: 'optimal_transport'`
- 检查学习率是否过大

### 问题2: 生成质量下降

**解决方案**:
- 增加训练epoch
- 调整`loss_weight`平衡
- 检查是否过早启用Flow Matching

### 问题3: 维度不匹配错误

**解决方案**:
- 检查配置文件中的`dim_latentcode`是否为768
- 确保使用正确的数据集
- 查看完整错误堆栈

---

## 📝 完整配置示例

参见: `configs/2_Diff/train_with_flow_matching_scheme1.yaml`

---

## 🎓 进阶使用

### 动态调整权重

可以在训练过程中逐步增加Flow Matching的权重：

```python
# 在Diffusion类中添加
def on_train_epoch_start(self):
    if self.current_epoch > 100:
        # 100 epoch后逐步增加FM权重
        new_weight = min(0.5, 0.2 + (self.current_epoch - 100) * 0.001)
        self.model.flow_matching_weight = new_weight
```

### 使用不同的采样器

训练完成后，可以使用Flow Matching的快速采样：

```python
# 在生成时使用更少的步数
self.model.sampling_timesteps = 50  # 原来是1000
```

---

## 📚 相关文件

- 核心实现: `model/Diffusion/flow_matching.py`
- 方案1实现: `model/Diffusion/flow_matching_scheme1.py`
- 配置文件: `configs/2_Diff/train_with_flow_matching_scheme1.yaml`
- 主训练脚本: `2_train_diff.py` (无需修改)

---

## ✅ 检查清单

使用前请确认：

- [ ] 已安装所有依赖包
- [ ] 配置文件路径正确
- [ ] 数据集路径正确
- [ ] WandB配置正确（如果使用）
- [ ] 有足够的GPU内存（训练开销增加约20%）

---

**祝训练顺利！如有问题，请查看日志或联系开发者。**

