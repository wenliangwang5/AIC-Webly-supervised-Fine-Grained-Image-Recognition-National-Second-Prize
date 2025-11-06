#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
repeats.py
高吞吐 pHash（64-bit）计算 + “跨子目录”精确重复统计（PyTorch 版，GPU 做 2D-DCT）。

特点：
- CPU（多进程 DataLoader）解码 + EXIF 方向矫正 + 透明度处理 + 统一灰度 32×32
- GPU（PyTorch）批量 2D-DCT + 生成 64-bit pHash
- 两阶段：1) 边算边写 TSV；2) TSV 排序分组，仅输出“跨子目录”的重复
- 对坏图/异常图稳健跳过，不中断整体流程

依赖：
  pip install torch torchvision pillow tqdm numpy

推荐（5090 + 22 核 CPU）：
  OMP_NUM_THREADS=1 python repeats.py /your/huge/images \
    --device cuda:0 --batch-size 4096 --workers 16 --prefetch-factor 2 \
    --out-dir out --tmp-dir tmp
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# -----------------------------
# 基础配置
# -----------------------------
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in SUPPORTED_EXTS

# -----------------------------
# Dataset & Collate
# -----------------------------
class ImageFolderDataset(Dataset):
    """
    递归枚举根目录图片。
    在 __getitem__ 中完成：
      - EXIF 方向矫正
      - 透明度合成到白底
      - 统一为灰度 32×32（单通道）
    返回：tensor [1,32,32] (uint8), rel_dir, abs_path
    """
    def __init__(self, root: Path):
        self.root = root
        self.paths: List[Path] = [
            p for p in root.rglob("*") if p.is_file() and is_image_file(p)
        ]
        self.paths.sort()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        try:
            with Image.open(p) as im:
                # EXIF 方向
                im = ImageOps.exif_transpose(im)
                # 透明度/调色板处理
                if im.mode in ("RGBA", "LA") or ("transparency" in im.info):
                    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
                    im = Image.alpha_composite(bg, im.convert("RGBA")).convert("RGB")
                elif im.mode == "P":
                    im = im.convert("RGB")
                else:
                    im = im.convert("RGB")

                # 统一到灰度 32×32（CPU），确保 batch 可 stack
                im = im.resize((32, 32), Image.BILINEAR).convert("L")  # [32,32]
                arr = np.asarray(im, dtype=np.uint8)                    # H,W uint8
                ten = torch.from_numpy(arr).unsqueeze(0)                # [1,32,32] uint8
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            return None

        rel_dir = str(p.parent.relative_to(self.root)) if p.parent != self.root else "."
        return ten, rel_dir, str(p.resolve())

def collate_drop_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    imgs, rels, paths = zip(*batch)
    imgs = torch.stack(imgs, dim=0)  # [N,1,32,32] uint8
    return imgs, list(rels), list(paths)

# -----------------------------
# GPU: DCT & pHash
# -----------------------------
def dct_matrix(n: int, device=None, dtype=torch.float32):
    k = torch.arange(n, device=device, dtype=dtype).reshape(-1, 1)
    i = torch.arange(n, device=device, dtype=dtype).reshape(1, -1)
    C = torch.cos((torch.pi / n) * (i + 0.5) * k)
    C[0, :] *= 1.0 / torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
    C *= torch.sqrt(torch.tensor(2.0 / n, dtype=dtype, device=device))
    return C  # [n,n]

@torch.inference_mode()
def phash_batch_gray_u8_fixed32(
    imgs_u8_1x32x32: torch.Tensor,  # [N,1,32,32], uint8 (pinned/cpu)
    C32: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    x = imgs_u8_1x32x32.to(device, non_blocking=True).float() / 255.0  # [N,1,32,32]
    x = x.squeeze(1)  # [N,32,32]
    # 2D-DCT: Y = C @ x @ C^T
    y = torch.matmul(C32, torch.matmul(x, C32.t()))  # [N,32,32]
    low = y[:, :8, :8]
    bits = low.reshape(low.size(0), -1)  # [N,64]
    dc = bits[:, 0:1]
    payload = bits[:, 1:]                # [N,63]
    med = payload.median(dim=1, keepdim=True).values
    payload_bits = (payload > med).to(torch.uint8)
    dc_bit = (dc > med).to(torch.uint8)
    all_bits = torch.cat([dc_bit, payload_bits], dim=1).to(torch.uint64)  # [N,64]
    powers = (63 - torch.arange(64, device=device, dtype=torch.uint64)).unsqueeze(0)
    hashes = (all_bits << powers).sum(dim=1)  # [N] uint64
    return hashes

# -----------------------------
# Phase 1: 计算 hash -> TSV
# -----------------------------
def phase1_compute_hashes_to_tsv(
    root: Path,
    tmp_tsv: Path,
    batch_size: int,
    workers: int,
    device_str: str,
    prefetch_factor: int = 2,
    persistent_workers: Optional[bool] = None,
) -> int:
    dataset = ImageFolderDataset(root)
    if len(dataset) == 0:
        print("[INFO] 没有找到图片。支持的后缀：", ", ".join(sorted(SUPPORTED_EXTS)))
        return 0

    if persistent_workers is None:
        persistent_workers = workers > 0

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if workers > 0 else 2,
        collate_fn=collate_drop_none,
        drop_last=False,
    )

    device = torch.device(device_str)
    C32 = dct_matrix(32, device=device, dtype=torch.float32)

    count = 0
    tmp_tsv.parent.mkdir(parents=True, exist_ok=True)
    with tmp_tsv.open("w", encoding="utf-8") as f:
        for batch in tqdm(loader, desc="Phase 1/2: hashing", unit="batch"):
            if batch is None:
                continue
            imgs, rels, paths = batch
            if imgs.numel() == 0:
                continue
            hashes = phash_batch_gray_u8_fixed32(imgs, C32, device)  # GPU
            hashes_cpu = hashes.cpu().numpy()  # uint64
            for h, rel, p in zip(hashes_cpu, rels, paths):
                f.write(f"{int(h)}\t{rel}\t{p}\n")
                count += 1

    print(f"[INFO] Phase 1 完成：写入 {count} 条 -> {tmp_tsv}")
    return count

# -----------------------------
# Phase 2: 排序分组 -> 输出
# -----------------------------
def phase2_group_and_emit(
    tmp_tsv: Path,
    out_csv: Path,
    out_json: Path,
):
    rows: List[Tuple[int, str, str]] = []
    with tmp_tsv.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.rstrip("\n")
            if not s:
                continue
            try:
                h_str, rel, p = s.split("\t", 2)
                rows.append((int(h_str), rel, p))
            except ValueError:
                continue

    rows.sort(key=lambda x: x[0])

    duplicate_groups = []
    csv_rows = []

    i, n = 0, len(rows)
    while i < n:
        j = i + 1
        while j < n and rows[j][0] == rows[i][0]:
            j += 1
        group = rows[i:j]
        if len(group) >= 2:
            dirs = {g[1] for g in group}
            if len(dirs) >= 2:
                gid = len(duplicate_groups) + 1
                members = []
                for (_h, rel, p) in sorted(group, key=lambda x: (x[1], x[2])):
                    members.append({"rel_dir": rel, "path": p})
                    csv_rows.append(["exact", gid, "", rel, p])
                duplicate_groups.append({
                    "group_id": gid,
                    "phash_uint64": int(rows[i][0]),
                    "members": members
                })
        i = j

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump({
            "tmp_source": str(tmp_tsv),
            "total_groups": len(duplicate_groups),
            "groups": duplicate_groups
        }, f, ensure_ascii=False, indent=2)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group_type", "group_id", "distance_to_rep", "rel_dir", "path"])
        writer.writerows(csv_rows)

    print(f"[INFO] Phase 2 完成：跨目录重复组 {len(duplicate_groups)} 个")
    print(f"[DONE] 输出：\n- CSV:  {out_csv}\n- JSON: {out_json}")

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="pHash(64-bit) + 跨子目录精确重复统计（PyTorch/GPU）")
    ap.add_argument("root", type=str, help="根目录（包含大量子目录与图片）")
    ap.add_argument("--device", type=str, default="cuda:0", help="计算设备：如 cuda:0 / cpu")
    ap.add_argument("--batch-size", type=int, default=2048, help="GPU 批大小（显存足可加大）")
    ap.add_argument("--workers", type=int, default=8, help="DataLoader 进程数（建议略低于 CPU 核数）")
    ap.add_argument("--prefetch-factor", type=int, default=2, help="每个 worker 预取的 batch 数")
    ap.add_argument("--persistent-workers", action="store_true", help="启用持久化 worker（默认根据 workers>0 自动选择）")
    ap.add_argument("--out-dir", type=str, default="out", help="结果输出目录")
    ap.add_argument("--tmp-dir", type=str, default="tmp", help="临时目录（存 TSV）")
    ap.add_argument("--tmp-file", type=str, default="", help="自定义 TSV 文件名（默认 phash_<rootname>.tsv）")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"[ERROR] 根目录不存在或不是目录：{root}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(args.tmp_dir).resolve(); tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_file = args.tmp_file or f"phash_{root.name}.tsv"
    tmp_tsv = tmp_dir / tmp_file
    out_csv = out_dir / "duplicate_report.csv"
    out_json = out_dir / "duplicate_groups.json"

    # Phase 1
    total = phase1_compute_hashes_to_tsv(
        root=root,
        tmp_tsv=tmp_tsv,
        batch_size=args.batch_size,
        workers=args.workers,
        device_str=args.device,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers if args.persistent_workers else None,
    )
    if total == 0:
        return

    # Phase 2
    phase2_group_and_emit(tmp_tsv=tmp_tsv, out_csv=out_csv, out_json=out_json)

if __name__ == "__main__":
    # 防止 Pillow/BLAS 过度并行
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
