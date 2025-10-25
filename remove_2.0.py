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

# >>> NEW (OWL-ViT)
from transformers import OwlViTProcessor, OwlViTForObjectDetection  # open-vocab text→box detector

# -----------------------------
# 0) Prompt parsing dictionaries
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

    # Only global ALL when no specific class is found (e.g., "remove everything")
    if ("remove all" in p or "remove everything" in p) and not targets:
        targets.add("ALL")

    positions = {pos for pos,kws in _POS_WORDS.items() if any(k in p for k in kws)}
    return targets, positions


# >>> NEW (OWL-ViT): simple extractor for open-vocab terms we want OWL-ViT to search
_COCO_KNOWN = {"person","car","truck","bus","bicycle","motorcycle","dog","cat","backpack","sports ball","vehicle","animal"}
_OV_CANDIDATES = ["rope","cord","rag", "wire","cable","leash","line","hose","chain","tape","string"]

ROPE_WORDS = {"rope","cord","cable","wire","string","line","clothesline","leash"}
# add nouns to search
_OV_CANDIDATES += ["tape","power cord","extension cord","duct tape","masking tape","gaffer tape"]

# color parsing
# ---- color helpers (put ABOVE ov_expansions) ----
COLOR_WORDS = {
    "yellow": ("yellow",),
    "orange": ("orange",),
    "red":    ("red",),
    "green":  ("green",),
    "blue":   ("blue",),
    "black":  ("black",),
    "white":  ("white",),
}

def get_color_word(prompt: str):
    p = prompt.lower()
    for cname, alts in COLOR_WORDS.items():
        if any(w in p for w in alts):
            return cname
    return None

def colorize_variants(variants, color_word):
    """If a color was requested, prepend/append it to each noun."""
    if not color_word:
        return variants
    out = []
    for v in variants:
        out.append(v)
        out.append(f"{color_word} {v}")
        out.append(f"{v} {color_word}")
    # dedupe while preserving order
    seen = set(); dedup = []
    for x in out:
        if x not in seen:
            seen.add(x); dedup.append(x)
    return dedup


def rope_prompt_expansions(user_prompt: str):
    p = user_prompt.lower()
    locs = []
    if "bottom" in p or "lower" in p: locs.append("bottom-right")
    if "right"  in p: locs.append("right")
    if "top"    in p or "upper" in p: locs.append("top-right")
    if "left"   in p: locs.append("left")
    if not locs: locs = ["bottom-right","right","lower right"]

    variants = [
        "rope","thin rope","coiled rope",
        "cord","nylon cord","string","line",
        "cable","wire","clothesline", "tape"
    ]
    phrases = []
    for v in variants:
        phrases.append(v)
        for L in locs:
            phrases.append(f"{v} {L}")
            phrases.append(f"{v} near the ground {L}")
            phrases.append(f"{v} along the floor {L}")
    return phrases

THIN_THINGS = [
    "rope","thin rope","coiled rope",
    "cord","power cord","extension cord",
    "string","line","cable","wire",
    "tape","duct tape","masking tape","gaffer tape",
    "rag","cloth","cleaning cloth","shop towel","microfiber cloth",
]

def ov_expansions(user_prompt: str):
    p = user_prompt.lower()

    # locations to bias the detector
    locs = []
    if "right" in p:  locs.append("right")
    if "left"  in p:  locs.append("left")
    if "bottom" in p or "lower" in p: locs.append("bottom")
    if "top" in p or "upper" in p:    locs.append("top")
    if "center" in p or "middle" in p: locs.append("center")
    if not locs:
        locs = ["center","right","bottom-right"]

    color = get_color_word(user_prompt)
    variants = colorize_variants(THIN_THINGS, color)

    phrases = []
    for v in variants:
        phrases.append(v)
        for L in locs:
            phrases.append(f"{v} {L}")
        phrases += [f"{v} on the table", f"{v} on desk", f"{v} on countertop"]

    return phrases, color



def extract_open_vocab_terms(prompt: str):
    p = prompt.lower()
    terms = [t for t in _OV_CANDIDATES if t in p]
    # optional: guess a noun after 'remove' if nothing matched and not a known COCO class
    if not terms and "remove" in p:
        try:
            after = p.split("remove", 1)[1].strip().split()
            if after:
                guess = after[0]
                if guess and guess not in _COCO_KNOWN:
                    terms.append(guess)
        except Exception:
            pass
    return terms

import cv2, numpy as np

# HSV hue ranges in OpenCV scale (H: 0..179)
HSV_RANGES = {
    "yellow": [(20, 50, 50,  40, 255, 255)],  # (Hmin,Smin,Vmin,Hmax,Smax,Vmax)
    "orange": [(10, 50, 50,  25, 255, 255)],
    # add more if you like
}

def _crop_hsv(pil_img, box):
    x1,y1,x2,y2 = box
    crop = np.array(pil_img.convert("RGB"))[y1:y2, x1:x2]
    if crop.size == 0: 
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    return hsv

def color_match_score(hsv_crop, color_word):
    if hsv_crop is None or color_word not in HSV_RANGES:
        return 0.0
    masks = []
    for (h1,s1,v1,h2,s2,v2) in HSV_RANGES[color_word]:
        lo = np.array([h1,s1,v1], dtype=np.uint8)
        hi = np.array([h2,s2,v2], dtype=np.uint8)
        masks.append(cv2.inRange(hsv_crop, lo, hi))
    mask = masks[0] if len(masks)==1 else np.maximum.reduce(masks)
    return float(mask.mean()) / 255.0  # 0..1 fraction of pixels in-range

def filter_boxes_by_color(pil_img, boxes, color_word, thresh=0.05):
    if not color_word: 
        return boxes
    kept = []
    for b in boxes:
        hsv = _crop_hsv(pil_img, b["box"])
        score = color_match_score(hsv, color_word)
        b["color_score"] = score
        if score >= thresh:  # at least ~5% of pixels match the requested color
            kept.append(b)
    # prefer higher color match
    kept.sort(key=lambda x: (x.get("color_score",0.0), x.get("score",0.0)), reverse=True)
    return kept

# -----------------------------
# 1) DETR panoptic
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
# 2) Select segments by type/position
# -----------------------------
def select_segments(segs, targets, positions, W, H):
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
        if "center" in positions and abs(cx-W*0.5) < W*0.22: ok=True #tighten
        if "top" in positions and cy < H*0.5: ok=True
        if "bottom" in positions and cy >= H*0.5: ok=True
        return ok
    return [s for s in segs if type_ok(s["label"]) and pos_ok(s["center"])]
    

# -----------------------------
# 3) Mask refinement (GrabCut)
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

def _offload_and_clear(*models):
    for m in models:
        try: m.to("cpu")
        except Exception: pass
    torch.cuda.empty_cache()


# -----------------------------
# 4) SDXL inpaint loader & call
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
        pipe.enable_vae_tiling()  # add this
    return pipe

def _to8(n):  # SDXL likes multiples of 8
    return max(8, (n // 8) * 8)

def sdxl_inpaint(pipe, image_pil, mask_u8, steps=32, strength=0.70, guidance=5.0, seed=None, max_side=2048):
    import random
    if seed is None:
        seed = random.randint(1, 10_000_000)
    gen = torch.Generator(device=pipe.device).manual_seed(seed)

    W, H = image_pil.size
    # scale image+mask down so the longest side is <= max_side (e.g., 2048)
    scale = min(1.0, float(max_side) / max(W, H))
    if scale < 1.0:
        newW, newH = _to8(int(W * scale)), _to8(int(H * scale))
        img_in  = image_pil.resize((newW, newH), Image.LANCZOS)
        mask_in = Image.fromarray(mask_u8, "L").resize((newW, newH), Image.NEAREST)
    else:
        newW, newH = _to8(W), _to8(H)
        img_in  = image_pil
        mask_in = Image.fromarray(mask_u8, "L")

    out = pipe(
        prompt="clean seamless background continuation, natural textures and lighting, photorealistic",
        negative_prompt="blurry, ghosting, leftover fabric, artifacts, watermark, repeating patterns, text",
        image=img_in,
        mask_image=mask_in,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        strength=float(strength),
        width=newW, height=newH,
        generator=gen,
    ).images[0]

    # Upscale back to original size if we downscaled
    if (newW, newH) != (W, H):
        out = out.resize((W, H), Image.BICUBIC)
    return out



# >>> NEW (OWL-ViT): loader + phrase detection + box→refined mask
_owl_cache = {"proc": None, "model": None}

def load_owlvit(device: str):
    if _owl_cache["proc"] is not None:
        return _owl_cache["proc"], _owl_cache["model"]
    proc = OwlViTProcessor.from_pretrained("google/owlvit-base-patch16")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch16")
    model.to(device).eval()
    _owl_cache["proc"], _owl_cache["model"] = proc, model
    return proc, model




@torch.no_grad()
def find_phrase_boxes_owlvit(pil_img, phrases, device: str, score_thresh=0.18):
    proc, model = load_owlvit(device)
    inputs = proc(text=[phrases], images=pil_img, return_tensors="pt").to(device)
    outputs = model(**inputs)
    target_sizes = torch.tensor([pil_img.size[::-1]], device=device)  # (H,W)
    results = proc.post_process_object_detection(outputs=outputs, target_sizes=target_sizes)[0]
    out = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        s = float(score.item())
        if s < score_thresh: 
            continue
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        out.append({"box": (x1,y1,x2,y2), "score": s, "label": phrases})
    return out

def box_to_refined_mask(pil_img, box, pad=8):
    W, H = pil_img.size
    x1,y1,x2,y2 = box
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(W, x2 + pad); y2 = min(H, y2 + pad)
    base = np.zeros((H,W), np.uint8)
    base[y1:y2, x1:x2] = 255
    return refine_mask_grabcut(pil_img, base, dilate_fg=7, dilate_unknown=21, iters=5)

def filter_boxes(boxes, W, H, min_score=0.25, max_area_ratio=0.30):
    kept = []
    for b in boxes:
        (x1,y1,x2,y2), s = b["box"], b["score"]
        if s < min_score: 
            continue
        w, h = max(1, x2-x1), max(1, y2-y1)
        area = (w*h) / (W*H)
        if area > max_area_ratio:   # reject huge blobs
            continue
        kept.append(b)
    return kept

# -----------------------------
# 5) High-level process
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

    # >>> NEW (OWL-ViT): route to open-vocab when prompt mentions non-COCO things (e.g., rope)
    # --- OWL-ViT routing (richer phrases / rope-friendly) ---
    pl = prompt.lower()
    phrases = []
    color_word = None
    
    # decide phrases
    if any(w in pl for w in ["rope","cord","cable","wire","rag", "tape","clothesline","power cord","extension cord"]):
        phrases, color_word = ov_expansions(prompt)
    else:
        ov_terms = extract_open_vocab_terms(prompt)
        phrases = ov_terms or []
    
    if phrases:
        boxes = find_phrase_boxes_owlvit(img, phrases, device=device, score_thresh=0.12)
    
        # gentle location bias
        cx = lambda b: (b["box"][0]+b["box"][2]) / 2
        cy = lambda b: (b["box"][1]+b["box"][3]) / 2
        if "right"  in pl: boxes = [b for b in boxes if cx(b) > 0.55 * W] or boxes
        if "left"   in pl: boxes = [b for b in boxes if cx(b) < 0.45 * W] or boxes
        if "bottom" in pl: boxes = [b for b in boxes if cy(b) > 0.55 * H] or boxes
        if "top"    in pl: boxes = [b for b in boxes if cy(b) < 0.45 * H] or boxes
    
        # area sanity (keep skinny, drop huge)
        def _filter_area(boxes, W, H, min_score=0.12, max_area=0.50, min_area=0.0003):
            kept=[]
            for b in boxes:
                (x1,y1,x2,y2), s = b["box"], b["score"]
                if s < min_score: continue
                w,h = max(1,x2-x1), max(1,y2-y1)
                ar = (w*h)/(W*H)
                if ar>max_area or ar<min_area: continue
                kept.append(b)
            return kept
        boxes = _filter_area(boxes, W, H)
    
        # NEW: color gate if user asked for one
        boxes = filter_boxes_by_color(img, boxes, color_word, thresh=0.05)
    
        # build mask
        combined_ov = np.zeros((H,W), np.uint8)
        for b in boxes:
            m = box_to_refined_mask(img, b["box"], pad=3)  # tight pad for thin stuff
            combined_ov = np.maximum(combined_ov, m)
    
        if combined_ov.max() > 0:
            if save_debug:
                base,_ = os.path.splitext(out_path)
                Image.fromarray(combined_ov).save(base + "_mask_owlvit.png")
            pipe = load_sdxl(device)
            # consider ROI inpainting to save VRAM
            result = sdxl_inpaint(pipe, img, combined_ov, steps=steps, strength=strength, guidance=guidance, seed=seed)
            result.save(out_path)
            print(f"[{os.path.basename(image_path)}] Removed: {phrases[:4]}... color={color_word} → {out_path}")
            return out_path


        # if OWL-ViT didn’t find anything, we fall through to DETR path below

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
    # we are going to offload the segmenting models so the GPU has space to regenrate a large image
    try:
        _offload_and_clear(detr)          # DETR model
        _offload_and_clear(_owl_cache.get("model"))  # OWL-ViT model (if loaded)
    except NameError:
        pass
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
# 6) CLI
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
    p = argparse.ArgumentParser(description="Smart removal (DETR + OWL-ViT + GrabCut + SDXL Inpaint)")
    p.add_argument("--image", required=True, help="Image file OR directory to process")
    p.add_argument("--prompt", required=True, help='e.g. "remove the rope on the right" or "remove the man in the center"')
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
