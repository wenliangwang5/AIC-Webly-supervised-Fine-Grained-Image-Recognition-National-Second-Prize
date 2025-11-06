#!/bin/bash
# ============================================================
# run_all.sh
# 说明：
#   镜像内复现训练与推理结果的统一入口脚本。
#   包含：
#     - 5000 类任务完整流程（含 CLIP 清洗）
#     - 400  类任务完整流程（仅 SNSCL 清洗）
#
# 使用方式：
#   1）默认运行 5000 类流程：
#       bash run_all.sh
#
#   2）显式指定任务类型：
#       TASK=5000 bash run_all.sh    # 5000 类
#       TASK=400  bash run_all.sh    # 400 类
#
#   3）数据路径约定：
#       - 原始训练集：默认为当前目录下的 train/      (TRAIN_DIR)
#       - 清洗 + 划分后的数据：默认为 clean_out/     (DATA_PATH)
#       - 测试集目录：默认为 test_B/                 (TEST_DIR)
#
#      如需修改，可在运行前设置环境变量，例如：
#       TRAIN_DIR=/data/train_5000 \
#       DATA_PATH=/data/clean_out \
#       TEST_DIR=/data/test_B \
#       TASK=5000 bash run_all.sh
#
# 不强制要求传参控制，本脚本通过环境变量提供可选控制能力。
# 注意！！强烈建议您挂载训练数据集到当前目录的train文件夹下
# ============================================================

set -e  # 任一步骤出错则退出

#######################################
# 基本配置（可通过环境变量覆盖）
#######################################
TASK="${TASK:-5000}"          # 任务类型：5000 或 400
TRAIN_DIR="${TRAIN_DIR:-train}"  # 原始训练集根目录
DATA_PATH="${DATA_PATH:-clean_out}"  # 清洗 + 划分后的数据目录
TEST_DIR="${TEST_DIR:-test_B}"      # 测试集目录

echo "========================================"
echo "  任务类型 (TASK)   : ${TASK}"
echo "  训练集目录        : ${TRAIN_DIR}"
echo "  清洗后数据目录    : ${DATA_PATH}"
echo "  测试集目录        : ${TEST_DIR}"
echo "========================================"
echo

#######################################
# 5000 类任务完整流程
#######################################
run_5000() {
  echo ">>>> [5000 类] Step 1: CLIP 清洗（clip_clean_2.py）"
  #注意：CLIP兼容性非常差，我们的脚本优化不佳，在部分机器下会退化到CPU运行，较慢。但通常全程不会超过两小时
  python clip_clean_2.py \
    --train_dir "${TRAIN_DIR}" \
    --out_dir clip_ultimate \
    --skip_quality \
    --min_class_size 50
  #注意：在我们的服务器上，bs为50能流畅运行，若不适合您的机器，请您及时对参数做出调整。
  echo ">>>> [5000 类] Step 2: SNSCL 类间清洗（snscl_cleaner2.py）"
  python3 snscl_cleaner2.py \
    --data_path clip_ultimate/kept \
    --output_dir "${DATA_PATH}" \
    --save_cache \
    --batch_size 50 \
    --num_workers 10

  echo ">>>> [5000 类] Step 3: 划分训练集 / 验证集（split_dataset.py）"
  python split_dataset.py \
    --src "${DATA_PATH}/cleaned_all" \
    --dst "${DATA_PATH}"

  echo ">>>> [5000 类] Step 4: 训练模型（run_training_large.sh）"
  # 训练脚本内部会读取环境变量 DATA_PATH
  # 注意：训练较慢，且需要连接Huggingface下载模型，请提前配置可用的系统代理。由于时间紧张，我们没有花费过多时间优化性能，故此训练较消耗显存和算力，5090下的完整训练时间在48h左右，显存峰值31g。
  DATA_PATH="${DATA_PATH}" ./run_training_large.sh

  echo ">>>> [5000 类] Step 5: 推理生成提交文件（infer.py -> submission5000.csv）"
  # 下面的参数极大影响模型性能，请勿随意修改
  python infer.py \
    --input_dir "${TEST_DIR}" \
    --output_csv submission5000.csv \
    --net1 convnextv2_base \
    --net1_ckpt model/net1_step2_convnextv2_base_5000cls_best_1.pth \
    --img_size 384 \
    --batch_size 50 \
    --tta_hflip \
    --tta_scales 0.9,1.0,1.1

  echo ">>>> [5000 类] 完成，结果已输出到 submission5000.csv"
}

#######################################
# 400 类任务完整流程
#######################################
# 400 类的情况于5000类相似，其需求资源相对较少，故不再一一做注意事项说明
run_400() {
  echo ">>>> [400 类] Step 1: SNSCL 类间清洗（snscl_cleaner2.py，不做 CLIP 清洗）"
  python3 snscl_cleaner2.py \
    --data_path "${TRAIN_DIR}" \
    --output_dir "${DATA_PATH}" \
    --save_cache \
    --batch_size 50 \
    --num_workers 10

  echo ">>>> [400 类] Step 2: 划分训练集 / 验证集（split_dataset.py）"
  python split_dataset.py \
    --src "${DATA_PATH}/cleaned_all" \
    --dst "${DATA_PATH}"

  echo ">>>> [400 类] Step 3: 训练模型（run_training.sh 或 run_training_large.sh）"
  # 如果你有专门的 400 类训练脚本，用 run_training.sh；
  # 若统一用 run_training_large.sh，则改这里。
  DATA_PATH="${DATA_PATH}" ./run_training.sh


  echo ">>>> [400 类] Step 4: 推理生成提交文件（infer.py -> submission400.csv）"
  python infer.py \
    --input_dir "${TEST_DIR}" \
    --output_csv submission400.csv \
    --net1 convnextv2_base \
    --net1_ckpt model/net1_step2_convnextv2_base_400cls_best_1.pth \
    --img_size 384 \
    --batch_size 50 \
    --tta_hflip \
    --tta_scales 0.9,1.0,1.1

  echo ">>>> [400 类] 完成，结果已输出到 submission400.csv"
}

#######################################
# 主入口：根据 TASK 选择流程
#######################################
case "${TASK}" in
  5000)
    echo ">>>> 当前选择：5000 类任务流程"
    run_5000
    ;;
  400)
    echo ">>>> 当前选择：400 类任务流程"sss
    run_400
    ;;
  *)
    echo "!!! 错误：不支持的 TASK=${TASK}，请设置为 5000 或 400"
    exit 1
    ;;
esac

echo "==== 所有流程执行完毕 ===="
