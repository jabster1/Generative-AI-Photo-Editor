#AUTHOR: Jaden Barnwell, October 17th, 2025

#!/usr/bin/env python3
import os, sys, argparse, warnings, glob, random
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import torch
from PIL import Image

# Optional HEIC support (safe no-op if missing)
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

from transformers import DetrImageProcessor, DetrForSegmentation
from diffusers import StableDiffusionXLInpaintPipeline, EulerAncestralDiscreteScheduler
import torch.nn.functional as F

# -----------------------------
# Prompt parsing dictionaries
# -----------------------------
_COCO_SYNONYMS = {
    "person": ["person","people","man","woman","boy","girl","human"],
    "car": ["car","vehicle","sedan","coupe","convertible"],
    "truck": ["truck","pickup","lorry"],
    "bus": ["bus","coach"],
    "bicycle": ["bicycle","bike"],
    "motorcycle": ["motorcycle","motorbike"],
    "dog": ["dog","puppy"], "cat": ["cat","kitten"],
    "backpack": ["backpack","bag","handbag","purse"],
    "sports ball": ["ball","soccer ball","basketball","football","tennis ball","baseball"],
    "vehicle": [], "animal": []
}
_POS_WORDS = {"left":["left"],"right":["right"],"center":["center","middle"],"top":["top","upper"],"bottom":["bottom","lower"]}

def parse_prompt(prompt: str):
    p = prompt.lower()
    targets = set()
    for coco, kws in _COCO_SYNONYMS.items():
        if coco in p or any(k in p for k in kws):
            targets.add(coco)
    if any(k in p for k in ["all","everything","remove all"]):
        targets.add("ALL")
    positions = {pos for pos,kws in _POS_WORDS.items() if any(k in p for k in kws)}
    return targets, positions

# -----------------------------
# DETR panoptic
# -----------------------------
def load_detr(device: str = None):
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50-panoptic")
    model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return processor, model, device

@torch.no_grad()
def panoptic_segments(pil_img, processor, model, device, score_thresh=0.75, mask_thresh=0.5):
    W, H = pil_img.size
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    outputs = model(**inputs)

    # Try panoptic post-process first (API differences across versions)
    try:
        result = processor.post_process_panoptic(outputs, target_sizes=[(H, W)])[0]
    except TypeError:
        h, w = inputs["pixel_mask"].shape[-2], inputs["pixel_mask"].shape[-1]
        result = processor.post_process_panoptic(outputs, processed_sizes=[(h, w)])[0]
    except Exception:
        result = None

    segs_out = []
    if isinstance(result, dict) and "segments_info" in result and "segmentation" in result:
        panoptic_seg = np.array(result["segmentation"])
        for s in result["segments_info"]:
            if float(s.get("score", 1.0)) < score_thresh:
                continue
            sid = s["id"]
            mask = (panoptic_seg == sid).astype(np.uint8) * 255
            ys, xs = np.where(mask > 0)
            if len(xs) == 0: 
                continue
            x1, x2, y1, y2 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
            cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
            lbl = model.config.id2label[s["label_id"]].lower()
            segs_out.append({"label": lbl, "score": float(s.get("score",1.0)),
                             "mask": mask, "bbox": (x1,y1,x2,y2), "center": (cx,cy)})
        return segs_out, (W, H)

    # Fallback: instance masks
    probs = outputs.logits.softmax(-1)[0, :, :-1]
    scores, labels = probs.max(-1)
    keep = scores > score_thresh
    if keep.sum() == 0:
        return [], (W, H)

    masks = outputs.pred_masks[0, keep]  # (K,h,w)
    labels = labels[keep]; scores = scores[keep]
    masks_up = F.interpolate(masks.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)[0]
    masks_bin = (masks_up.sigmoid() > mask_thresh).cpu().numpy().astype(np.uint8) * 255

    for k in range(masks_bin.shape[0]):
        m = masks_bin[k]
        ys, xs = np.where(m > 0)
        if len(xs) == 0: 
            continue
        x1, x2, y1, y2 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
        lbl = model.config.id2label[int(labels[k])].lower()
        segs_out.append({"label": lbl, "score": float(scores[k]),
                         "mask": m, "bbox": (x1,y1,x2,y2), "center": (cx,cy)})
    return segs_out, (W, H)

# -----------------------------
# Select segments by type/position
# -----------------------------
def select_segments(segs, targets, positions, W, H, choose_top1_if_center=True):
    if not segs: 
        return []
    def type_ok(lbl):
        if "ALL" in targets: return True
        if lbl in targets: return True
        if "vehicle" in targets and lbl in {"car","truck","bus","motorcycle","bicycle"}: return True
        if "animal" in targets and lbl in {"dog","cat","bird"}: return True
        for k,syns in _COCO_SYNONYMS.items():
            if k in targets and (lbl==k or any(s in lbl for s in syns)): return True
        return False
    def pos_ok(center):
        if not positions: return True
        cx,cy = center; ok=False
        if "left" in positions and cx < W*0.5: ok=True
        if "right" in positions and cx >= W*0.5: ok=True
        if "center" in positions and abs(cx-W*0.5) < W*0.22: ok=True #tightened for middle
        if "top" in positions and cy < H*0.5: ok=True
        if "bottom" in positions and cy >= H*0.5: ok=True
        return ok
    return [s for s in segs if type_ok(s["label"]) and pos_ok(s["center"])]
   
 

# -----------------------------
# Mask refinement (GrabCut)
# -----------------------------
def refine_mask_grabcut(image_pil: Image.Image, mask_u8: np.ndarray,
                        dilate_fg=11, dilate_unknown=27, iters=6, max_area_ratio=0.2):
    img = np.array(image_pil.convert("RGB"))
    H, W = mask_u8.shape
    base = (mask_u8 > 0).astype(np.uint8)
    base = cv2.dilate(base, np.ones((dilate_fg,dilate_fg), np.uint8), iterations=1)

    unknown = cv2.dilate(base, np.ones((dilate_unknown,dilate_unknown), np.uint8), iterations=1)
    sure_fg = base
    sure_bg = (unknown == 0).astype(np.uint8)

    gc_mask = np.full((H, W), cv2.GC_PR_BGD, np.uint8)
    gc_mask[sure_bg == 1] = cv2.GC_BGD
    gc_mask[unknown == 1] = cv2.GC_PR_FGD
    gc_mask[sure_fg == 1] = cv2.GC_FGD

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, gc_mask, None, bgdModel, fgdModel, iters, cv2.GC_INIT_WITH_MASK)

    refined = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=1)
    refined = cv2.dilate(refined, np.ones((3,3), np.uint8), iterations=1)

    if refined.sum() / 255 > max_area_ratio * H * W:
        refined = (base*255).astype(np.uint8)
    return refined

# -----------------------------
# SDXL inpaint loader & call
# -----------------------------
def load_sdxl(device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if (device == "cuda") else torch.float32
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    if device == "cuda":
        pipe.to("cuda")
        pipe.enable_attention_slicing()
    return pipe

def sdxl_inpaint(pipe, image_pil, mask_u8, steps=32, strength=0.60, guidance=5.0, seed: int = None):
    if seed is None:
        seed = random.randint(1, 10_000_000)
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    mask_pil = Image.fromarray(mask_u8, mode="L")
    prompt = "clean seamless background continuation, natural textures and lighting, photorealistic"
    negative = "blurry, ghosting, leftover fabric, artifacts, watermark, repeating patterns, text"

    out = pipe(
        prompt=prompt, negative_prompt=negative,
        image=image_pil, mask_image=mask_pil,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        strength=float(strength),
        generator=generator
    )
    return out.images[0]

# -----------------------------
# High-level process
# -----------------------------
def process_image(
    image_path: str, prompt: str, out_path: str,
    score_thresh=0.72, mask_thresh=0.5, steps=40, strength=0.80,
    guidance=5.0, save_debug=False, force_cpu=False, seed=None
):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    device = "cuda" if (not force_cpu and torch.cuda.is_available()) else "cpu"

    targets, positions = parse_prompt(prompt)
    if not targets:
        targets = {"person"}  # safe default

    processor, detr, dev = load_detr(device)
    segs, (W,H) = panoptic_segments(img, processor, detr, dev, score_thresh=score_thresh, mask_thresh=mask_thresh)

    chosen = select_segments(segs, targets, positions, W, H)
    if not chosen:
        print(f"[{os.path.basename(image_path)}] No segments matched the prompt; nothing to remove.")
        img.save(out_path)
        return out_path

    # Merge base mask
    combined = np.zeros((H,W), np.uint8)
    for s in chosen:
        combined = np.maximum(combined, s["mask"])
    combined = cv2.dilate(combined, np.ones((5,5),np.uint8), iterations=1)
    mask_bin = (combined>0).astype(np.uint8)*255

    # Refine to include clothing/edges
    mask_refined = refine_mask_grabcut(img, mask_bin, dilate_fg=11, dilate_unknown=27, iters=6)

    if save_debug:
        # Create colorful overlays (all and selected)
        def overlay_from_segments(base_img, segs_list):
            base = np.array(base_img).copy()
            overlay = np.zeros_like(base)
            for s in segs_list:
                color = np.random.randint(0,255,3,dtype=np.uint8)
                overlay[s["mask"]>0] = color
            return cv2.addWeighted(base, 0.6, overlay, 0.4, 0)

        all_debug = overlay_from_segments(img, segs)
        sel_debug = overlay_from_segments(img, chosen)
        base, ext = os.path.splitext(out_path)
        Image.fromarray(all_debug).save(base + "_debug_all.png")
        Image.fromarray(sel_debug).save(base + "_debug_sel.png")
        Image.fromarray(mask_bin).save(base + "_mask_raw.png")
        Image.fromarray(mask_refined).save(base + "_mask_refined.png")

    pipe = load_sdxl(device)
    result = sdxl_inpaint(pipe, img, mask_refined, steps=steps, strength=strength, guidance=guidance, seed=seed)
    result.save(out_path)
    labels = sorted({s['label'] for s in chosen})
    print(f"[{os.path.basename(image_path)}] Removed {len(chosen)} segment(s): {labels} → {out_path}")
    return out_path

# -----------------------------
# CLI
# -----------------------------
def valid_image_paths(path: str):
    exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".heic",".heif",".webp"}
    if os.path.isdir(path):
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(path, f"*{ext}")))
        return sorted(paths)
    elif os.path.isfile(path):
        return [path]
    else:
        raise FileNotFoundError(f"Input path not found: {path}")

def main():
    p = argparse.ArgumentParser(description="Smart removal (DETR + GrabCut + SDXL Inpaint)")
    p.add_argument("--image", required=True, help="Image file OR directory to process")
    p.add_argument("--prompt", required=True, help='e.g. "remove the man on the bike in the center"')
    p.add_argument("--out", default=None, help="Output file (for single image) or output folder (for directory)")
    p.add_argument("--score", type=float, default=0.65, help="DETR score threshold")
    p.add_argument("--mask", type=float, default=0.5, help="Instance mask threshold")
    p.add_argument("--steps", type=int, default=40, help="SDXL steps")
    p.add_argument("--strength", type=float, default=0.80, help="SDXL strength")
    p.add_argument("--guidance", type=float, default=5.0, help="SDXL guidance scale")
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    p.add_argument("--save-debug", action="store_true", help="Save debug overlays & masks")
    p.add_argument("--seed", type=int, default=None, help="Optional seed for reproducibility")
    args = p.parse_args()

    in_paths = valid_image_paths(args.image)

    if len(in_paths) == 1:
        in_path = in_paths[0]
        out_path = args.out or os.path.join(os.path.dirname(in_path), "result.jpg")
        process_image(
            in_path, args.prompt, out_path,
            score_thresh=args.score, mask_thresh=args.mask,
            steps=args.steps, strength=args.strength, guidance=args.guidance,
            save_debug=args.save_debug, force_cpu=args.cpu, seed=args.seed
        )
    else:
        # Directory mode
        out_dir = args.out or os.path.join(args.image, "_results")
        os.makedirs(out_dir, exist_ok=True)
        for ip in in_paths:
            base = os.path.splitext(os.path.basename(ip))[0]
            out_path = os.path.join(out_dir, f"{base}_result.jpg")
            try:
                process_image(
                    ip, args.prompt, out_path,
                    score_thresh=args.score, mask_thresh=args.mask,
                    steps=args.steps, strength=args.strength, guidance=args.guidance,
                    save_debug=args.save_debug, force_cpu=args.cpu, seed=args.seed
                )
            except Exception as e:
                print(f"[ERROR] {ip}: {e}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
