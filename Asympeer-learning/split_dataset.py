#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import shutil
import random
import argparse
from pathlib import Path
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}

def _copy_or_link(src, dst, link_mode="hardlink"):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    if link_mode == "hardlink":
        try:
            os.link(src, dst); return
        except Exception:
            pass
    if link_mode == "symlink":
        try:
            os.symlink(os.path.abspath(src), dst); return
        except Exception:
            pass
    # fallback: copy
    shutil.copy2(src, dst)

def split_image_folder(
    original_train_dir: str,
    new_base_dir: str,
    split_ratio: float = 0.2,
    min_train_per_class: int = 1,
    min_val_per_class: int = 1,
    seed: int = 2025,
    link_mode: str = "hardlink",
    overwrite: bool = False,
    exts: set[str] | None = IMG_EXTS,
):
    """
    将 ImageFolder 结构（root/cls_x/xxx.jpg）按类随机划分为 train_split / val_split。
    - 对极少类友好：保证 train ≥ min_train_per_class，且（若可能）val ≥ min_val_per_class。
    - 硬链接优先，不占额外磁盘；跨盘或失败时回退复制。
    """
    original_train_dir = Path(original_train_dir)
    new_base_dir = Path(new_base_dir)
    new_train_dir = new_base_dir / "train_split"
    new_val_dir = new_base_dir / "val_split"

    if not original_train_dir.is_dir():
        raise FileNotFoundError(f"原始目录不存在: {original_train_dir}")

    # 处理已存在目录
    if (new_train_dir.exists() or new_val_dir.exists()):
        if not overwrite:
            raise FileExistsError(f"{new_train_dir} 或 {new_val_dir} 已存在；加 --overwrite 或先删除后再跑。")
        shutil.rmtree(new_train_dir, ignore_errors=True)
        shutil.rmtree(new_val_dir, ignore_errors=True)

    new_train_dir.mkdir(parents=True, exist_ok=True)
    new_val_dir.mkdir(parents=True, exist_ok=True)

    # 列出类别目录（按名字排序，保证可复现）
    class_dirs = sorted([p for p in original_train_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not class_dirs:
        raise RuntimeError(f"在 {original_train_dir} 下未找到任何类别文件夹。")

    rng = random.Random(seed)

    total_train = total_val = 0
    for cls_path in tqdm(class_dirs, desc="正在处理类别"):
        # 读取该类的文件（按名称排序，保证可复现）
        files = [f for f in cls_path.iterdir() if f.is_file() and (exts is None or f.suffix.lower() in exts)]
        files = sorted(files, key=lambda p: p.name)
        rng.shuffle(files)  # 固定 seed 的随机

        n = len(files)
        if n == 0:
            continue

        # 计算 val 数量（带护栏）
        if n <= min_train_per_class:
            val_count = 0
        elif n <= (min_train_per_class + min_val_per_class):
            # 可用的 val 不能让 train 变 0
            val_count = max(0, min(min_val_per_class, n - min_train_per_class))
        else:
            val_count = int(round(n * split_ratio))
            val_count = max(val_count, min_val_per_class)
            val_count = min(val_count, n - min_train_per_class)

        val_files = files[:val_count]
        train_files = files[val_count:]

        # 目标子目录
        dst_train_cls = new_train_dir / cls_path.name
        dst_val_cls = new_val_dir / cls_path.name
        dst_train_cls.mkdir(parents=True, exist_ok=True)
        dst_val_cls.mkdir(parents=True, exist_ok=True)

        for f in train_files:
            _copy_or_link(str(f), str(dst_train_cls / f.name), link_mode=link_mode)
        for f in val_files:
            _copy_or_link(str(f), str(dst_val_cls / f.name), link_mode=link_mode)

        total_train += len(train_files)
        total_val += len(val_files)

    print("\n数据集划分完成！")
    print(f"原始目录：{original_train_dir}")
    print(f"新的训练数据：{new_train_dir}")
    print(f"新的验证数据：{new_val_dir}")
    print(f"合计：train={total_train}  val={total_val}  (val_ratio≈{total_val / max(1, total_train+total_val):.3f})")

def parse_args():
    ap = argparse.ArgumentParser(description="将 ImageFolder 数据集按类划分为 train_split / val_split（长尾友好）")
    ap.add_argument("--src", type=str, default="clean_data_3/cleaned_all", help="源目录（各类子文件夹）")
    ap.add_argument("--dst", type=str, default="clean_data_3", help="输出根目录（会在此创建 train_split / val_split）")
    ap.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例（0~1）")
    ap.add_argument("--min-train", type=int, default=1, help="每类最少保留到训练集的样本数")
    ap.add_argument("--min-val", type=int, default=1, help="每类最少放入验证集的样本数（若可能）")
    ap.add_argument("--seed", type=int, default=2025, help="随机种子（保证可复现）")
    ap.add_argument("--link-mode", type=str, default="hardlink", choices=["hardlink", "symlink", "copy"], help="保存方式")
    ap.add_argument("--overwrite", action="store_true", help="目标目录已存在时覆盖")
    ap.add_argument("--all-files", action="store_true", help="不过滤扩展名（默认只统计常见图片后缀）")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    exts = None if args.all_files else IMG_EXTS
    split_image_folder(
        original_train_dir=args.src,
        new_base_dir=args.dst,
        split_ratio=args.val_ratio,
        min_train_per_class=args.min_train,
        min_val_per_class=args.min_val,
        seed=args.seed,
        link_mode=args.link_mode,
        overwrite=args.overwrite,
        exts=exts,
    )
