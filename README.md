🏆National Sencond Prize🏆
---


# 图像分类任务说明文档
## 🧩 文件结构说明

| 文件名 | 作用说明 |
|--------|-----------|
| `run_all.sh` | 主控制脚本，一键执行清洗、划分、训练与推理全流程。支持 400 类与 5000 类两种任务。 |
| `run_training.sh` | 400 类任务训练脚本，资源占用较低，适用于中小规模任务。 |
| `run_training_large.sh` | 5000 类任务训练脚本，针对大规模类别优化，包含更强模型与配置。 |
| `clip_clean_2.py` | 基于 CLIP 模型的粗清洗脚本，用于去除异常与跨类数据（仅 5000 类使用）。 |
| `snscl_cleaner2.py` | 类间样本相似度清洗（SNSCL），进一步过滤难分样本。 |
| `split_dataset.py` | 划分清洗后的样本至 `train/` 和 `val/`。 |
| `infer.py` | 推理脚本，根据已训练模型生成提交文件（`submission*.csv`）。 |

---

## 🚀 快速运行指南

### ▶️ 一键运行（推荐方式）
镜像评审请阅读并配置run_all.sh脚来本复现全流程：

```bash
bash run_all.sh
```

系统会自动运行 **5000 类任务流程**，包括：  
1. CLIP 清洗  
2. SNSCL 类间清洗  
3. 数据划分  
4. 模型训练  
5. 推理生成 `submission5000.csv`  


## ⚙️ 可配置环境变量

| 变量名 | 含义 | 默认值 | 示例 |
|--------|------|--------|------|
| `TASK` | 任务类型，支持 `400` 或 `5000` | `5000` | `TASK=400` |
| `TRAIN_DIR` | 原始训练集路径 | `train` | `/data/train_5000` |
| `DATA_PATH` | 清洗后数据输出目录 | `clean_out` | `/data/clean_out` |
| `TEST_DIR` | 测试集路径 | `test_B` | `/data/test_B` |

示例：
```bash
TRAIN_DIR=/data/train DATA_PATH=/data/clean_out TEST_DIR=/data/test_B TASK=5000 bash run_all.sh
```
## 🧱 训练脚本区别

| 脚本名 | 用途 | 特点 |
|--------|------|------|
| `run_training.sh` | 400 类任务 | 轻量配置，显存占用低 |
| `run_training_large.sh` | 5000 类任务 | 针对大规模任务优化，显存占用高 |
| `run_all.sh` | 主调度脚本 | 自动选择对应流程并执行所有步骤 |

---

## ⚠️ 注意事项

- 所有 Python 脚本均需在容器内运行。  
- 若出现 HuggingFace 权限或超时问题，请配置代理或本地缓存。  
- CLIP 清洗阶段可能退化到 CPU 执行，属于正常现象。  
- 模型训练阶段显存需求较高，请确保 GPU ≥ 24GB。  

---

## ✅ 输出结果

运行结束后生成：  
- `submission400.csv` 或 `submission5000.csv`（推理输出）  
- `model/` 目录下保存训练权重  
- `clean_out/` 目录下保存最终数据划分结果  

---

##  竞赛数据集 (๑•̀ㅂ•́)و✧

链接: https://pan.baidu.com/s/18mCgdptiqRjEh7Ys5DXMfw 提取码: pgd8 （这里面只有测试集 没有训练集咋办）
