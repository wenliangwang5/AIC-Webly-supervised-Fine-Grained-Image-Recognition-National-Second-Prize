# infer_folder.py
# -*- coding: utf-8 -*-
import os, csv, argparse, glob
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# timm 是你训练脚本依赖的
try:
    import timm
except ImportError:
    raise SystemExit("timm 未安装，请先: pip install -U timm")

# ====== 常量，与训练保持一致 ======
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)
IMG_EXTS = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}

def detect_num_classes_from_state_dict(sd: dict) -> int:
    """
    从 checkpoint 的最后分类层权重推断类别数：
    依次尝试几种常见命名（ConvNeXt/ViT/Swin 等 timm 模型）
    """
    candidate_keys = [
        'head.fc.weight',      # convnext
        'head.weight',         # swin/vit
        'classifier.weight',   # resnet/timm一些模型
        'fc.weight',           # torchvision resnet
    ]
    for k in candidate_keys:
        w = sd.get(k, None)
        if isinstance(w, torch.Tensor) and w.ndim >= 2:
            return w.shape[0]
    # 兜底：遍历找到任何2D权重名里带 head/cls/fc 的
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor) or v.ndim < 2:
            continue
        nk = k.lower()
        if any(t in nk for t in ['head', 'cls', 'classifier', 'fc']):
            return v.shape[0]
    raise RuntimeError("无法从 checkpoint 推断 num_classes，请手动指定 --num_classes")

def build_model(arch: str, ckpt_path: str, device: torch.device, channels_last: bool):
    sd = torch.load(ckpt_path, map_location='cpu')
    if "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]  # 兼容某些保存格式

    num_classes = detect_num_classes_from_state_dict(sd)
    print(f"[INFO] {arch} -> detected num_classes = {num_classes}")

    # 用指定类别数构建 timm 模型
    try:
        model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
    except TypeError:
        model = timm.create_model(arch, pretrained=False, num_classes=num_classes)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {len(missing)} (前若干) {missing[:8]}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)} (前若干) {unexpected[:8]}")

    model.eval()
    model.to(device)
    if channels_last:
        model.to(memory_format=torch.channels_last)
    return model

class ImageFolderFlat(torch.utils.data.Dataset):
    def __init__(self, root: str, img_size: int):
        self.paths = []
        for ext in IMG_EXTS:
            self.paths.extend(glob.glob(os.path.join(root, f'**/*{ext}'), recursive=True))
        self.paths.sort()
        if not self.paths:
            raise FileNotFoundError(f"在 {root} 下未找到图片")
        self.tf = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert('RGB')
        return self.tf(img), os.path.basename(p)

@torch.no_grad()
def run_infer(models, loader, device, tta_scales=(1.0,), tta_hflip=False, amp_dtype=None, channels_last=False):
    """
    返回 [(filename, pred_id_int), ...]
    """
    preds = []
    use_amp = (amp_dtype is not None and device.type == 'cuda')
    for imgs, names in loader:
        if channels_last:
            imgs = imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
        else:
            imgs = imgs.to(device, non_blocking=True)

        logits_sum = None
        for s in tta_scales:
            if s != 1.0:
                new_sz = int(imgs.shape[-1]*s)
                inp = F.interpolate(imgs, size=(new_sz,new_sz), mode='bilinear', align_corners=False)
            else:
                inp = imgs
            with torch.autocast('cuda', dtype=amp_dtype) if use_amp else torch.cuda.amp.autocast(enabled=False):
                out_sum_this_scale = None
                for m in models:
                    out = m(inp)
                    out_sum_this_scale = out if out_sum_this_scale is None else out_sum_this_scale + out
                if tta_hflip:
                    inp_flip = torch.flip(inp, dims=[-1])
                    for m in models:
                        out_flip = m(inp_flip)
                        out_sum_this_scale = out_sum_this_scale + out_flip
            logits_sum = out_sum_this_scale if logits_sum is None else logits_sum + out_sum_this_scale

        pred_ids = torch.argmax(logits_sum, dim=1).tolist()
        for n, pid in zip(names, pred_ids):
            preds.append((n, pid))
    return preds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', required=True, help='赛事方提供的初赛测试集文件夹')
    ap.add_argument('--output_csv', required=True, help='输出CSV路径（无表头），每行：filename, 4位类别码')
    ap.add_argument('--net1', required=True, help='如 convnext_base')
    ap.add_argument('--net1_ckpt', required=True, help='net1 的 .pth 权重')
    ap.add_argument('--net2', default='', help='可选：第二个模型，如 swin_base_patch4_window7_224')
    ap.add_argument('--net2_ckpt', default='', help='可选：第二个模型的 .pth 权重')
    ap.add_argument('--img_size', type=int, default=384, help='与训练/验证阶段一致（如 step2 用 384）')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--channels_last', action='store_true')
    ap.add_argument('--no_bf16', action='store_true')
    ap.add_argument('--tta_scales', type=str, default='1.0', help='如 1.0,1.15')
    ap.add_argument('--tta_hflip', action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_bf16 = (not args.no_bf16) and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    # 构建模型（支持集成多个模型求和）
    models = []
    models.append(build_model(args.net1, args.net1_ckpt, device, args.channels_last))
    if args.net2 and args.net2_ckpt:
        models.append(build_model(args.net2, args.net2_ckpt, device, args.channels_last))

    # Data
    ds = ImageFolderFlat(args.input_dir, args.img_size)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    scales = tuple(float(x) for x in args.tta_scales.split(',') if x.strip())
    preds = run_infer(
        models=models, loader=dl, device=device,
        tta_scales=scales, tta_hflip=args.tta_hflip,
        amp_dtype=amp_dtype, channels_last=args.channels_last
    )

    # 写 CSV：文件名, 四位类别码（示例： xxxxxxxxxxxx.jpg, 0000）
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, 'w', newline='') as f:
        # 如果竞赛方严格要求“逗号后有空格”，可以改为手工写行：f.write(f"{name}, {cls_code}\n")
        w = csv.writer(f)
        for name, pid in preds:
            cls_code = f"{int(pid):04d}"  # 不足4位左侧补0
            w.writerow([name, cls_code])

    print(f"[DONE] Wrote {len(preds)} lines to {args.output_csv}")

if __name__ == '__main__':
    main()
