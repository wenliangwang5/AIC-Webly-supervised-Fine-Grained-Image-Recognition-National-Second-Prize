#!/usr/bin/env bash
set -Eeuo pipefail

#############################################
# AIC Peer-Learning Training Launcher (优化版)
# - 自动为 ConvNeXt 启用 LLRD（其余骨干走判别式 LR）
# - 日志按时间戳切分，避免覆盖
# - 兼容 main.py 的 *_autocls_* 命名（n_classes=0）
#############################################

# =============== 基础环境 ===============
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# export TORCH_LOGS="+dynamo"  # 需要时打开

# 时间戳日志后缀
STAMP="$(date +%Y%m%d_%H%M%S)"

# =============== 数据与任务 ===============
DATA_PATH="${DATA_PATH:-aft}"        # 目录下需要有 train_split/ 和 val_split/
NUM_CLASSES="${NUM_CLASSES:-0}"                  # 0 = 让 main.py 自动检测（保存将使用 *_autocls_*）
 
if [[ ! -d "${DATA_PATH}/train_split" || ! -d "${DATA_PATH}/val_split" ]]; then
  echo "[ERROR] ${DATA_PATH}/train_split 或 ${DATA_PATH}/val_split 不存在" >&2
  exit 1
fi

# =============== 模型选择 ===============
# 默认为.fsmc/.fsmc._in1k
NET1_ARCH="${NET1_ARCH:-convnextv2_base}"
NET2_ARCH="${NET2_ARCH:-swin_base_patch4_window12_384}"

# =============== 训练超参 ===============
BATCH_SIZE="${BATCH_SIZE:-12}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-20}"             # 只训头，10~15 一般够
STAGE2_EPOCHS="${STAGE2_EPOCHS:-80}"             # 总计 ≈100，可按需改
BASE_LR_STAGE1="${BASE_LR_STAGE1:-0.001}"
BASE_LR_STAGE2="${BASE_LR_STAGE2:-0.00003}"

# =============== 算法 & 功能开关 =============== 
USE_BALANCED_SOFTMAX="${USE_BALANCED_SOFTMAX:-true}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.05}"       # 比 0.10 更稳
# LLRD 策略：auto = 只要有 ConvNeXt 就开；1=强制开；0=关（走判别式 LR）
USE_LLRD="${USE_LLRD:-auto}"
BB_LR_MULT="${BB_LR_MULT:-0.1}"                  # 不用 LLRD 时 backbone 的 LR 乘子

# 可选 TTA（会在每轮验证时触发，略慢；建议最终评测时打开）
USE_TTA="${USE_TTA:-0}"
TTA_SCALES="${TTA_SCALES:-1.0,1.15}"
TTA_HFLIP="${TTA_HFLIP:-1}"

# =============== 性能参数 ===============
NPROC=$(command -v nproc >/dev/null && nproc || echo 8)
DATALOADER_WORKERS="${DATALOADER_WORKERS:-$(( NPROC ))}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
USE_COMPILE="${USE_COMPILE:-1}"

# =============== 依赖检查 ===============
python - <<'PY'
try:
    import timm  # noqa
except Exception:
    import sys
    sys.stderr.write("WARNING: timm 未安装，将自动 pip install -U timm\n")
    sys.exit(1)
PY
if [[ $? -ne 0 ]]; then
  pip install -U timm
fi

# =============== 环境回显 ===============
echo "==== CONFIG @ ${STAMP} ===="
echo "DATA_PATH=${DATA_PATH}"
echo "NET1_ARCH=${NET1_ARCH}  NET2_ARCH=${NET2_ARCH}"
echo "BATCH_SIZE=${BATCH_SIZE}  EPOCHS: S1=${STAGE1_EPOCHS}, S2=${STAGE2_EPOCHS}"
echo "LR: S1=${BASE_LR_STAGE1}, S2=${BASE_LR_STAGE2}"
echo "BS=${USE_BALANCED_SOFTMAX}  LS=${LABEL_SMOOTHING}"
echo "LLRD=${USE_LLRD}  BB_LR_MULT=${BB_LR_MULT}"
echo "TTA=${USE_TTA} scales=${TTA_SCALES} hflip=${TTA_HFLIP}"
echo "WORKERS=${DATALOADER_WORKERS}  PREFETCH=${PREFETCH_FACTOR}  ACCUM=${GRAD_ACCUM}"
echo "COMPILE=${USE_COMPILE}"
echo "==========================="

# =============== 参数拼装（通用） ===============
EXTRA_ARGS=""
[[ "${USE_BALANCED_SOFTMAX}" == "true" ]] && EXTRA_ARGS+=" --use_balanced_softmax"
EXTRA_ARGS+=" --label_smoothing ${LABEL_SMOOTHING}"
EXTRA_ARGS+=" --workers ${DATALOADER_WORKERS} --prefetch_factor ${PREFETCH_FACTOR} --grad_accum_steps ${GRAD_ACCUM}"

if [[ "${USE_COMPILE}" == "1" ]]; then
  EXTRA_ARGS+=" --compile --compile_mode max-autotune"
else
  EXTRA_ARGS+=" --no_compile"
fi
# >>> 新增：根据架构决定输入分辨率（Swin-384 必须 384）
if [[ "${NET1_ARCH}" == *_384 || "${NET2_ARCH}" == *_384 ]]; then
  IMG_SIZE="${IMG_SIZE:-384}"
else
  IMG_SIZE="${IMG_SIZE:-224}"
fi
EXTRA_ARGS+=" --img_size ${IMG_SIZE}"

# 若已是 384，就避免“渐进放大”反复切分
if [[ "${IMG_SIZE}" -ge 384 ]]; then
  EXTRA_ARGS+=" --finetune_img_size ${IMG_SIZE} --resize_at_epoch 9999"
fi
# =============== Step 1：只训分类头 ===============
echo "=============== STARTING STEP 1: Training Classifier Head ==============="
STEP1_LOG="step1_${NET1_ARCH}_${NET2_ARCH}_${STAMP}.log"
python main.py \
  --dataset "${DATA_PATH}" \
  --n_classes ${NUM_CLASSES} \
  --net1 "${NET1_ARCH}" \
  --net2 "${NET2_ARCH}" \
  --batch_size ${BATCH_SIZE} \
  --epoch ${STAGE1_EPOCHS} \
  --step 1 \
  --base_lr ${BASE_LR_STAGE1} \
  ${EXTRA_ARGS} | tee "${STEP1_LOG}"

# 与保存命名规则一致（保持 *_1.pth 后缀）
MODEL_DIR="model"
# 当 NUM_CLASSES=0，main.py 会另存 *_autocls_*，下方的拼接会得到 autocls（即 'auto' + 'cls'）
MODEL1_PATH="${MODEL_DIR}/net1_step1_${NET1_ARCH}_$( [ ${NUM_CLASSES} -gt 0 ] && echo ${NUM_CLASSES} || echo auto )cls_best_1.pth"
MODEL2_PATH="${MODEL_DIR}/net2_step1_${NET2_ARCH}_$( [ ${NUM_CLASSES} -gt 0 ] && echo ${NUM_CLASSES} || echo auto )cls_best_1.pth"

if [[ ! -f "${MODEL1_PATH}" || ! -f "${MODEL2_PATH}" ]]; then
  echo "[WARN] Step 1 最优权重未找到："
  [[ ! -f "${MODEL1_PATH}" ]] && echo "  - 缺少 ${MODEL1_PATH}"
  [[ ! -f "${MODEL2_PATH}" ]] && echo "  - 缺少 ${MODEL2_PATH}"
  echo "      可能文件名里的类数与 --n_classes 不一致；Step 2 会由 main.py 继续尝试 --resume（含 *_autocls_* 兜底）。"
fi

echo "=============== WAITING 5 SECONDS BEFORE STEP 2 ==============="
sleep 5

# =============== Step 2：全量微调 ===============
echo "=============== STARTING STEP 2: Fine-tuning Whole Network ==============="

# 根据骨干类型自动决定是否启用 LLRD（仅对 ConvNeXt 有意义）
_autollrd=0
if [[ "${NET1_ARCH}" == convnext* || "${NET2_ARCH}" == convnext* ]]; then
  _autollrd=1
fi

_step2_llrd_flag=""
case "${USE_LLRD}" in
  1|true|TRUE|on|ON)     _step2_llrd_flag="--use_llrd" ;;
  0|false|FALSE|off|OFF) _step2_llrd_flag="--bb_lr_mult ${BB_LR_MULT}" ;;
  auto|AUTO)             [[ ${_autollrd} -eq 1 ]] && _step2_llrd_flag="--use_llrd" || _step2_llrd_flag="--bb_lr_mult ${BB_LR_MULT}" ;;
  *)                     [[ ${_autollrd} -eq 1 ]] && _step2_llrd_flag="--use_llrd" || _step2_llrd_flag="--bb_lr_mult ${BB_LR_MULT}" ;;
esac

STEP2_ARGS="${EXTRA_ARGS} ${_step2_llrd_flag}"

if [[ "${USE_TTA}" == "1" ]]; then
  STEP2_ARGS+=" --use_tta --tta_scales ${TTA_SCALES}"
  [[ "${TTA_HFLIP}" == "1" ]] && STEP2_ARGS+=" --tta_hflip"
fi

STEP2_LOG="step2_${NET1_ARCH}_${NET2_ARCH}_${STAMP}.log"
python main.py \
  --dataset "${DATA_PATH}" \
  --n_classes ${NUM_CLASSES} \
  --net1 "${NET1_ARCH}" \
  --net2 "${NET2_ARCH}" \
  --batch_size ${BATCH_SIZE} \
  --epoch ${STAGE2_EPOCHS} \
  --step 2 \
  --base_lr ${BASE_LR_STAGE2} \
  --resume \
  ${STEP2_ARGS} | tee "${STEP2_LOG}"

echo "=============== TRAINING FINISHED (${STAMP}) ==============="
echo "Logs:"
echo "  - ${STEP1_LOG}"
echo "  - ${STEP2_LOG}"
