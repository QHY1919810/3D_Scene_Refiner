# 基于 Static Point Routing 的 Observed-View 3DGS Rewrite 训练

本仓库是一个面向工程验证的 **Observed-View 3D Gaussian Splatting Rewrite Training** 原型系统。

当前任务设定不是 SDS，也不是纯文本驱动的无监督生成，而是：

- 从一个已经完成“插车”的 scene 3DGS 初始化出发
- 使用原始 **COLMAP 相机**
- 使用对应视角的 **真实观测 RGB**
- 使用可选的 **车体 mask**
- 使用 **Qwen Image Edit** 生成的图像作为 teacher target
- 对 3DGS 做 **按点路由（point-routed）的 rewrite 训练**

> 当前工程的重点不是通用 3DGS 重建，  
> 而是一个 **已知视角、teacher 引导、可控 point-routed 的 3DGS rewrite 训练框架**。

---

## 1. 项目目标

本项目的核心目标是：

在**已知真实机位**下，对一个已经插车后的 scene 3DGS 进行训练，使其渲染结果逐步朝 rewrite teacher 靠近，并且训练过程具有以下特点：

- teacher 监督可控
- 训练信号尽量稳定
- car 区域和 background 区域能够分流约束
- 优先验证**颜色参数是否真的能被 rewrite loss 拉动**
- 避免训练中出现“点云变稀疏 / 透明 / 躲开监督”的现象

当前阶段最关注的问题是：

- rewrite loss 是否真的有效
- fixed teacher 模式下训练是否可解释
- point-routed 约束是否比纯 pixel-routed 更可靠
- 在不修改几何参数的情况下，颜色是否能先学动

---

## 2. 当前设计概览

### 2.1 当前训练形态

当前主线训练配置为：

- **Observed-view training**
- **不使用 SDS**
- **不使用全局 shared supervision**
- **使用静态 point-tag routed**
- **使用 fixed teacher**
- **无 mask 样本直接跳过**

### 2.2 路由规则

当前按点路由依赖预先计算好的点标签：

- `tag == 1` → car points
- `tag != 1` → background points

训练时的 loss 路由为：

- **mask 内 rewrite + mask 内 HF loss** → 作用到 **car points**
- **mask 外 rewrite** → 作用到 **background points**

### 2.3 当前只更新颜色参数

当前 routed 版本**只让路由损失更新颜色参数**：

- `_features_dc`
- `_features_rest`

不会让 routed loss 直接更新：

- `_xyz`
- `_opacity`
- `_scaling`
- `_rotation`

这样做的目的是：

- 避免点通过“缩小 / 变透明 / 位移”来逃避监督
- 先验证 rewrite 信号是否足以驱动颜色变化
- 先把“能否学动”这个问题隔离出来

---

## 3. 主要文件说明

### 核心系统文件
- `/nfs4/qhy/projects/threestudio/custom/threestudio-3dgs/system/gaussian_splatting.py`

主要负责：

- 前向渲染
- fixed teacher 读取
- rewrite loss 计算
- 高通 / 高频损失计算
- 两次 backward
- 按点裁梯度
- skip no-mask 逻辑

### 几何文件
- `/nfs4/qhy/projects/threestudio/custom/threestudio-3dgs/geometry/gaussian_base.py`

主要负责：

- 点标签读取
- 支持 `.npy` / `.npz["tags"]`
- `car_point_mask`
- `bg_point_mask`
- 梯度 clone / 局部保留 / 合并
- prune 后同步维护标签

### 配置文件
- `/nfs4/qhy/projects/threestudio/custom/threestudio-3dgs/configs/gaussian_splatting_static_routed_fixed_teacher_merged.yaml`

当前主实验配置入口。

### 启动脚本
- `/nfs4/qhy/projects/threestudio/bashes/train_observed_supervise_only.sh`

当前训练启动脚本。

### 数据集
- 解压并放在`/nfs4/qhy/projects/threestudio/dataset`

当前训练启动脚本。

---

## 4. 数据假设

当前训练流程默认依赖以下输入：

- 一个已经插车后的 scene 3DGS 初始化
- 原始 COLMAP 相机文件
- 每个视角对应的 GT RGB
- 可选的车体 mask
- 静态 routed 训练所需的 point-tag 文件

典型 point-tag 文件形式为：

- `scene_plus_car_tagged.ply.tags.npz`

期望内容：

- key 为 `tags`
- shape 为 `[N]`

标签语义：

- `1 = car`
- `2 / 3 = background`

---

## 5. Teacher 模式

## 5.1 Fixed Teacher 模式

当前仓库主线支持 **fixed rewrite targets**。

使用流程：

1. 先为每个视角准备一张 teacher 图
2. 将 teacher 图保存到磁盘
3. 训练时根据 `view_id` 读取对应 teacher 图
4. 不再在训练过程中反复在线调用 Qwen

这个模式更适合：

- 稳定实验
- 快速迭代
- 排除 teacher 漂移
- 提高可解释性

### 5.2 为什么优先 fixed teacher

相比在线 teacher 刷新：

- loss 曲线更容易解释
- teacher 不会跟着 render 一起飘
- 训练更快
- 更适合验证“模型能否逼近固定目标”

---

## 6. 训练逻辑

对于每个训练 step（通常假设 `batch_size = 1`）：

1. 渲染当前视角
2. 读取 GT RGB
3. 读取 mask
4. 根据当前 `view_id` 读取 fixed teacher 图
5. 计算：
   - mask 内 rewrite loss
   - mask 外 rewrite loss
   - mask 内高频损失
6. 执行 **两次 backward**
7. 按 point-tag 裁剪梯度
8. 合并两组梯度
9. 执行 optimizer step

### 梯度路由规则

**分支 A：car branch**
- mask 内 rewrite
- mask 内 HF
- 只保留 car points 的梯度

**分支 B：background branch**
- mask 外 rewrite
- 只保留 background points 的梯度

这实现的是**按点路由**，而不是只在图像空间里乘一个 mask。

---

## 7. 无 Mask 样本处理策略

当前 routed 版本要求 mask 存在。

如果某个 batch 没有 mask：

- 该 step 会被直接跳过
- 不调用 teacher
- 不做 backward
- 不执行 optimizer step

这样做的目的是：

- 避免 point-routed 逻辑在无效样本上误运行
- 避免浪费渲染与 teacher 计算开销
- 保持训练行为可解释

---

## 8. 高频损失（HF Loss）

HF loss 用作 mask 内的额外锚点约束。

支持的响应模式包括：

- `laplacian`
- `sobel`

主要作用：

- 保护局部边缘 / 高频细节
- 减少 rewrite target 导致的过度平滑
- 在车区域内保留更稳定的局部结构信息

---

## 9. 重要说明

### 9.1 这不是一个通用 3DGS 训练器
本仓库当前实现是围绕以下目标定制的：

- observed-view rewrite training
- static point-tag routed training
- fixed teacher 实验

### 9.2 动态 routed 不是当前主线
之前尝试过按视角实时投影点到 mask 的动态 routed 方案，但投影统计不稳定，当前已不作为主线继续推进。  
当前推荐路线是：**静态 point-tag routed**。

### 9.3 不建议混入 shared/global losses
对于严格的 point-routed 训练，建议关闭所有 shared/global losses，例如：

- RGB supervised loss
- silhouette loss
- position regularizer
- opacity regularizer
- scale regularizer
- TV 类损失

否则这些全局项会和 routed 目标互相干扰。

### 9.4 学习率非常关键
如果 routed 版本只更新颜色参数，而 `feature_lr` 设得过小，会出现：

- loss 看起来存在
- 但 render 几乎不变化
- 肉眼观察像是“完全没学到”

因此在调试“rewrite 是否有效”时，通常需要显著提高颜色学习率。

---

## 10. 如何运行

一个典型启动方式如下：

```bash
python launch.py \
  --config /path/to/gaussian_splatting_static_routed_fixed_teacher_merged.yaml \
  --train --gpu 0