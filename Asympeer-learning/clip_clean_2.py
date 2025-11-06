#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU-accelerated surgical cleaner (Version 3).
- GOAL: Zero false positives. Aggressively preserve any potential organism photo.
- LOGIC: Multi-gate filtering with a "suspicious zone" and class size protection.
- Pass-1 (CPU): Basic quality heuristics + near-duplicate removal.
- Pass-2 (GPU): CLIP-based multi-gate decision logic.
"""

# ---------- 环境防护 ... (保持不变) ----------
import os
os.environ.setdefault("PILLOW_DISABLE_PLUGIN", "WebPImagePlugin")
os.environ.setdefault("OMP_NUM_THREADS", "1")
# ... (其余环境变量设置不变) ...
import sys, csv, argparse, shutil
# ... (其余 import 保持不变) ...
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from PIL import Image, UnidentifiedImageError, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import imagehash
import torch
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import open_clip
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from collections import Counter

# ---------------------- 阈值 (新版：三步决策阈值) ----------------------
# Pass-1 质量阈值 (保持不变)
MIN_SHORT_EDGE = 128
ASPECT_MIN, ASPECT_MAX = 1/3, 3.0
BLUR_VAR_THR = 25.0
ENTROPY_THR = 1.5

# Pass-2 CLIP 决策阈值
# <--- KEY CHANGE: 引入新的决策逻辑和阈值 ---
# 规则1: 硬性噪声移除 (必须同时满足才移除)
#   - 它与"负面类别"的相似度必须非常高
MUST_BE_NONPHOTO_NEG_SIM = 0.98
#   - 同时, 它与"正面类别"的相似度必须非常低
MUST_BE_NONPHOTO_POS_SIM = 0.05

# 规则2: 自然超类确认 (只要满足就保留)
#   - 只要它与任何自然超类的相似度高于这个极低阈值，就无条件保留
#   - 这个阈值极大地保护了所有看起来沾点边的图片
NATURE_SUPERCLS_MIN_SIM_FOR_KEEPING = 0.03

# ---------------------- 元类与模板 (保持不变) ----------------------
NATURE_SUPERCLASSES = [
    "Protozoa", "Chromista", "Fungi", "Plantae", "Mollusca", "Animalia",
    "Insecta", "Arachnida", "Aves", "Mammalia", "Actinopterygii", "Amphibia", "Reptilia"
]
PHOTO_POS_TEMPLATES = [
    "a natural photo of a living organism", "a photo of an animal, plant, or fungus", "a wildlife photograph",
    "a close-up photograph of an insect, flower, or animal", "a photo of a living creature in its natural habitat",
]
PHOTO_NEG_TEMPLATES = [
    "an illustration, drawing, painting, or cartoon", "a computer graphic, 3D render, CGI, or video game screenshot",
    "a screenshot of a user interface, app, or website", "a poster, meme, infographic, or logo",
    "a map, chart, graph, or diagram", "a photo of a book cover, magazine page, or text document",
    "a blurry, noisy, or low-quality image", "a photo of a pure landscape, scenery, or sky with no clear subject",
]
SUPERCLS_TEMPLATES = [
    "a natural photo of a {}", "a close-up photo of a {}",
    "a field observation photo of a {} in the wild", "a real camera photo of a {} in nature",
]

# ---------------------- 参数 (新增最小类保护参数) ----------------------
@dataclass
class Args:
    train_dir: str
    out_dir: str
    batch: int
    device: str
    amp: str
    hash_type: str
    skip_dedup: bool
    skip_quality: bool
    procs: int
    min_class_size: int # <--- NEW PARAMETER

def parse_args() -> Args:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True, help="train 根目录")
    ap.add_argument("--out_dir", required=True, help="输出根目录")
    # ... (其他参数不变) ...
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", choices=["bf16","fp16","off"], default="bf16")
    ap.add_argument("--hash_type", choices=["ahash","phash","dhash"], default="phash")
    ap.add_argument("--skip_dedup", action="store_true")
    ap.add_argument("--skip_quality", action="store_true")
    ap.add_argument("--procs", type=int, default=max(1, cpu_count() // 4)) #防止OOM
    # <--- NEW PARAMETER ---
    ap.add_argument("--min_class_size", type=int, default=50,
                        help="每个类别在清洗后至少保留的样本数。")
    return Args(**vars(ap.parse_args()))

# ---------------------- I/O 和图像读取 (保持不变) ----------------------
# ... (list_all_images, ensure_link, read_image_rgb_pil, etc. 保持不变) ...
def list_all_images(root: str) -> List[str]:
    exts = (".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff")
    out = []
    for dp,_,fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(exts):
                out.append(os.path.join(dp, fn))
    out.sort()
    return out

def ensure_link(dst_path, src_path):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    try:
        if os.path.exists(dst_path): os.remove(dst_path)
        os.link(src_path, dst_path)
    except OSError:
        shutil.copy2(src_path, dst_path)

def read_image_rgb(path: str) -> Optional[Image.Image]:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        try:
            data = np.fromfile(path, dtype=np.uint8)
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)) if bgr is not None else None
        except Exception:
            return None

# ---------------------- Pass-1 子进程任务 (保持不变) ----------------------
def _pass1_task(args_tuple):
    path, do_quality, hash_type = args_tuple
    img = read_image_rgb(path)
    if img is None: return (path, False, "corrupt_or_unsupported", 0.0, None)
    sharp = 0.0
    if do_quality:
        w, h = img.size
        if min(w, h) < MIN_SHORT_EDGE: return (path, False, f"short_edge<{MIN_SHORT_EDGE}", 0.0, None)
        ratio = w / (h + 1e-6)
        if not (ASPECT_MIN <= ratio <= ASPECT_MAX): return (path, False, "extreme_aspect", 0.0, None)
        gray = np.array(img.convert("L"))
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if sharp < BLUR_VAR_THR: return (path, False, f"too_blurry(var={sharp:.2f})", sharp, None)
        hist = np.asarray(img.convert('L').histogram(), dtype=np.float64)
        total = hist.sum()
        if total <= 0: return (path, False, "zero_histogram", sharp, None)
        p = hist / total
        entropy = float(-(p * np.log2(p + 1e-12)).sum())
        if entropy < ENTROPY_THR: return (path, False, f"low_entropy({entropy:.2f})", sharp, None)
    if hash_type == "ahash": h = imagehash.average_hash(img)
    elif hash_type == "dhash": h = imagehash.dhash(img)
    else: h = imagehash.phash(img)
    return (path, True, "ok_after_heuristics", sharp, str(h))

# ---------------------- CLIP scorer (保持不变) ----------------------
# ... (CLIPScorer class 保持不变) ...
class CLIPScorer:
    def __init__(self, device="cuda", amp="bf16", model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
        self.device, self.amp = device, amp
        torch.backends.cudnn.benchmark = True
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp) if torch.cuda.is_available() else None

        def encode_texts(txts: List[str]) -> torch.Tensor:
            tok = self.tokenizer(txts).to(self.device)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=self.autocast_dtype is not None):
                te = self.model.encode_text(tok).float()
            return te / te.norm(dim=-1, keepdim=True)
        
        self.photo_pos_te = encode_texts(PHOTO_POS_TEMPLATES)
        self.photo_neg_te = encode_texts(PHOTO_NEG_TEMPLATES)
        super_texts = [t.format(sc) for sc in NATURE_SUPERCLASSES for t in SUPERCLS_TEMPLATES]
        self.super_te = encode_texts(super_texts)
        self.super_slices = {sc: (i * len(SUPERCLS_TEMPLATES), (i + 1) * len(SUPERCLS_TEMPLATES)) for i, sc in enumerate(NATURE_SUPERCLASSES)}

    def img_emb(self, imgs: List[Image.Image]) -> torch.Tensor:
        batch = torch.stack([self.preprocess(im) for im in imgs]).to(self.device, non_blocking=True)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=self.autocast_dtype is not None):
            ie = self.model.encode_image(batch).float()
        return ie / ie.norm(dim=-1, keepdim=True)

    def score_natural_photo(self, img_emb: torch.Tensor):
        pos = (img_emb @ self.photo_pos_te.T).max(dim=1).values
        neg = (img_emb @ self.photo_neg_te.T).max(dim=1).values
        return pos, neg

    def score_superclasses(self, img_emb: torch.Tensor):
        sims = img_emb @ self.super_te.T
        per_class_max = torch.stack([sims[:, s:e].max(dim=1).values for sc, (s, e) in self.super_slices.items()], dim=1)
        sc_max, _ = per_class_max.max(dim=1)
        return sc_max

# ---------------------- 主流程 (核心逻辑修改) ----------------------
def main():
    args = parse_args()
    kept_root = os.path.join(args.out_dir, "kept")
    removed_root = os.path.join(args.out_dir, "removed")
    logs_dir = os.path.join(args.out_dir, "logs")
    os.makedirs(kept_root, exist_ok=True); os.makedirs(removed_root, exist_ok=True); os.makedirs(logs_dir, exist_ok=True)
    
    paths = list_all_images(args.train_dir)
    # <--- KEY CHANGE: 提前获取类别信息以用于最小类保护 ---
    path_to_class = {p: os.path.basename(os.path.dirname(p)) for p in paths}
    class_counts = Counter(path_to_class.values())
    
    print(f"Pass-1: Heuristics ({args.procs} procs)...")
    first_decision: Dict[str, Tuple[bool, str, float, Optional[str]]] = {}
    tasks = [(p, not args.skip_quality, args.hash_type) for p in paths]
    with Pool(processes=args.procs) as pool:
        for res in tqdm(pool.imap_unordered(_pass1_task, tasks, chunksize=256), total=len(tasks)):
            first_decision[res[0]] = res[1:]
    
    keep_after_pass1, remove_set = set(), set()
    if args.skip_dedup:
        for p, (ok, _, _, _) in first_decision.items():
            (keep_after_pass1 if ok else remove_set).add(p)
    else:
        buckets: Dict[str, Tuple[str, float]] = {}
        for p, (ok, _, sharp, h) in first_decision.items():
            if not ok or h is None: remove_set.add(p); continue
            if h in buckets:
                bp, bs = buckets[h]
                if sharp > bs: remove_set.add(bp); buckets[h] = (p, sharp)
                else: remove_set.add(p)
            else: buckets[h] = (p, sharp)
        for _, (p, _) in buckets.items(): keep_after_pass1.add(p)
    
    to_score = sorted(list(keep_after_pass1))
    if not to_score:
        print("No images left to score after Pass-1."); return

    print(f"Pass-2: CLIP scoring on {args.device} amp={args.amp}...")
    scorer = CLIPScorer(device=args.device, amp=args.amp)
    scores: Dict[str, Dict[str, float]] = {p: {} for p in to_score}
    
    B = args.batch
    for i in tqdm(range(0, len(to_score), B)):
        chunk = to_score[i:i+B]
        imgs, valid_paths = [], []
        for p in chunk:
            im = read_image_rgb(p)
            if im: imgs.append(im); valid_paths.append(p)
            else: remove_set.add(p)
        if not imgs: continue
        
        emb = scorer.img_emb(imgs)
        pos, neg = scorer.score_natural_photo(emb)
        scmax = scorer.score_superclasses(emb)
        for j, p in enumerate(valid_paths):
            scores[p] = {'pos': pos[j].item(), 'neg': neg[j].item(), 'sc_max': scmax[j].item()}
    
    print("Pass-3: Applying multi-gate decision logic...")
    kept_cnt, removed_cnt = 0, 0
    final_decisions: Dict[str, Tuple[str, str]] = {}

    # <--- KEY CHANGE: 新的三步决策逻辑 ---
    for p in paths:
        # Gate 0: 已经在 Pass-1 被移除了
        if p in remove_set:
            final_decisions[p] = ("REMOVE", first_decision[p][0])
            continue
        
        s = scores.get(p)
        if not s: # 图像读取失败等
             final_decisions[p] = ("REMOVE", "clip_load_failed")
             continue

        # Gate 1: 硬性噪声移除
        # 条件: (负面分很高 AND 正面分很低) -> 几乎肯定是海报/图表
        is_hard_noise = (s['neg'] > MUST_BE_NONPHOTO_NEG_SIM and s['pos'] < MUST_BE_NONPHOTO_POS_SIM)
        if is_hard_noise:
            final_decisions[p] = ("REMOVE", f"hard_noise(neg>{s['neg']:.2f},pos<{s['pos']:.2f})")
            continue
            
        # Gate 2: 自然超类确认
        # 条件: (与任何自然超类的相似度 > 极低阈值) -> 只要沾点边就保留
        looks_like_nature = s['sc_max'] > NATURE_SUPERCLS_MIN_SIM_FOR_KEEPING
        if looks_like_nature:
            final_decisions[p] = ("KEEP", f"looks_like_nature(sc_max>{s['sc_max']:.2f})")
            continue
            
        # Gate 3: 默认移除
        # 所有既不是硬噪声，又不像自然生物的，都归为"不确定/移除"
        final_decisions[p] = ("REMOVE", f"ambiguous(sc_max<{s['sc_max']:.2f})")

    # <--- KEY CHANGE: 应用最小类保护 ---
    print(f"Applying minimum class size protection (min_size={args.min_class_size})...")
    
    # 统计每个类别在初步决策后还剩多少图片
    current_kept_counts = Counter()
    for p, (decision, _) in final_decisions.items():
        if decision == "KEEP":
            current_kept_counts[path_to_class[p]] += 1
            
    # 对于那些即将低于下限的类别，从它们的"被移除"图片中救回一些
    paths_to_check_for_rescue = [p for p, (decision, _) in final_decisions.items() if decision == "REMOVE"]
    
    # 按分数高低排序，优先拯救质量相对较高的
    paths_to_check_for_rescue.sort(key=lambda p: scores.get(p, {}).get('sc_max', 0), reverse=True)

    for p in paths_to_check_for_rescue:
        class_name = path_to_class[p]
        if current_kept_counts[class_name] < args.min_class_size:
            # 拯救这张图片
            final_decisions[p] = ("KEEP", f"rescued_by_min_size_rule")
            current_kept_counts[class_name] += 1

    print("Writing outputs and logs...")
    csv_header = ["path", "bucket", "reason", "photo_pos_score", "photo_neg_score", "nature_superclass_max_sim", "sharpness"]
    with open(os.path.join(logs_dir, "decisions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(csv_header)
        for p in tqdm(paths):
            bucket, reason = final_decisions[p]
            s = scores.get(p, {})
            sharpness = first_decision.get(p, ["", "", 0.0])[2]
            row = [
                p, bucket, reason, 
                f"{s.get('pos', 0):.4f}", f"{s.get('neg', 0):.4f}", f"{s.get('sc_max', 0):.4f}",
                f"{sharpness:.2f}"
            ]
            
            if bucket == 'KEEP':
                ensure_link(os.path.join(kept_root, os.path.relpath(p, args.train_dir)), p)
                kept_cnt += 1
            else:
                ensure_link(os.path.join(removed_root, os.path.relpath(p, args.train_dir)), p)
                removed_cnt += 1
            w.writerow(row)

    print(f"\nDone. KEPT={kept_cnt} REMOVED={removed_cnt}")
    print(f"Final decisions logged in {logs_dir}")

if __name__ == "__main__":
    main()