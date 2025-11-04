#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Jaden Barnwell
November 4th, 2025

Main script for background blurring agent to use detr to segment the main individual in the image and blur the rest of the detected background around that main person.
"""


import os, io, math
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from PIL import Image, ImageOps
import cv2
import torch

# Optional HEIC support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

from transformers import pipeline as hf_pipeline


class BackgroundEditor:
    """
    BackgroundEditor: blur-only background editing.
    - Uses DETR panoptic to protect the subject (person).
    - OpenCV-based post enhancement and blur.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        max_edit_side: int = 1024,
        seg_model_id: str = "facebook/detr-resnet-50-panoptic",
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.max_edit_side = int(max_edit_side)

        # Helpful torch flags
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        # Pipelines
        self.seg_pipe = None

        # Load segmentation for person mask (CPU or GPU)
        try:
            seg_dev = 0 if (torch.cuda.is_available() and self.device == "cuda") else -1
            self.seg_pipe = hf_pipeline("image-segmentation", model=seg_model_id, device=seg_dev)
            print("✓ Segmentation model loaded")
        except Exception as e:
            self.seg_pipe = None
            print(f"✗ Segmentation model not available: {e}")

    # --------------------------- I/O & sizing ---------------------------

    @staticmethod
    def ensure_image(path_or_url: str) -> Image.Image:
        p = str(path_or_url)
        if p.startswith(("http://", "https://")):
            import requests
            r = requests.get(p, timeout=20)
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        # local path
        p = os.path.abspath(os.path.expanduser(p))
        return Image.open(p).convert("RGB")

    def _optimal_size(self, image: Image.Image) -> Tuple[int, int]:
        w, h = image.size
        if max(w, h) <= self.max_edit_side:
            new_w = max(8, (w // 8) * 8)
            new_h = max(8, (h // 8) * 8)
        else:
            r = self.max_edit_side / max(w, h)
            new_w = max(8, (int(w * r) // 8) * 8)
            new_h = max(8, (int(h * r) // 8) * 8)
        return new_w, new_h

    def _resize_to_optimal(self, image: Image.Image) -> Image.Image:
        new_w, new_h = self._optimal_size(image)
        if (new_w, new_h) == image.size:
            return image
        return image.resize((new_w, new_h), Image.LANCZOS)

    # --------------------------- enhancement ---------------------------

    @staticmethod
    def enhance(image: Image.Image) -> Image.Image:
        """
        Gentle sharpening + CLAHE + mild saturation. Safe, fast, dependency-light.
        """
        try:
            img = np.array(image)

            # 1) gentle sharpen
            kernel = np.array([[-0.1, -0.1, -0.1],
                               [-0.1,  1.8, -0.1],
                               [-0.1, -0.1, -0.1]], dtype=np.float32)
            sharp = cv2.filter2D(img, -1, kernel)

            # 2) CLAHE on L-channel
            lab = cv2.cvtColor(sharp, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            contrast = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

            # 3) mild saturation boost
            hsv = cv2.cvtColor(contrast, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 1] = np.clip(hsv[..., 1] * 1.15, 0, 255)
            out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            return Image.fromarray(out)
        except Exception:
            return image

    # ------------------------- masks (subject/bg) -----------------------

    def _subject_union_mask(self, image_rgb: Image.Image) -> Image.Image:
        """
        Returns 'L' mask where white=subject (person). If seg unavailable -> all black.
        """
        W, H = image_rgb.size
        if self.seg_pipe is None:
            return Image.new("L", (W, H), 0)

        out = self.seg_pipe(image_rgb)
        segments = out.get("segments_info") if isinstance(out, dict) else out
        m = np.zeros((H, W), dtype=np.uint8)

        for s in segments:
            label = (s.get("label") or "").lower()
            mask = s.get("mask")
            if label == "person" and mask is not None:
                m |= (np.array(mask.convert("L")) > 0).astype(np.uint8)

        return Image.fromarray(m * 255, mode="L")

    def _bg_edit_mask(self, image_rgb: Image.Image, grow_px: int = 6, feather_px: int = 10) -> Image.Image:
        """
        White=background (editable), black=subject (protected).
        """
        subj = self._subject_union_mask(image_rgb)  # white=subject
        subj_np = np.array(subj, dtype=np.uint8)

        if grow_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow_px + 1, 2 * grow_px + 1))
            subj_np = cv2.dilate(subj_np, k, iterations=1)

        if feather_px > 0:
            subj_np = cv2.GaussianBlur(subj_np, (feather_px * 2 + 1, feather_px * 2 + 1), 0)

        bg_np = 255 - subj_np
        return Image.fromarray(bg_np, mode="L")

    @staticmethod
    def _soften_mask(mask_L: Image.Image, feather_px: int = 10, erode_px: int = 2) -> Image.Image:
        m = np.array(mask_L.convert("L"), dtype=np.uint8)
        if erode_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
            m = cv2.erode(m, k, iterations=1)
        if feather_px > 0:
            m = cv2.GaussianBlur(m, (feather_px * 2 + 1, feather_px * 2 + 1), 0)
        return Image.fromarray(m, mode="L")

    # ------------------------------ blur-only ---------------------------

    def _blur_gaussian(self, image: Image.Image, bg_mask: Image.Image, radius: int = 19) -> Image.Image:
        r = max(3, radius | 1)  # force odd
        rgb = np.array(image.convert("RGB"))
        blurred = cv2.GaussianBlur(rgb, (r, r), 0)
        m = np.array(self._soften_mask(bg_mask, feather_px=10, erode_px=2), dtype=np.float32) / 255.0
        m = m[..., None]
        out = (rgb * (1 - m) + blurred * m).astype(np.uint8)
        return Image.fromarray(out)

    def _blur_bokeh(self, image: Image.Image, bg_mask: Image.Image, radius: int = 8) -> Image.Image:
        def disk_kernel(rad: int) -> np.ndarray:
            r = max(2, int(rad))
            d = 2 * r + 1
            y, x = np.ogrid[-r:r + 1, -r:r + 1]
            k = (x * x + y * y) <= r * r
            k = k.astype(np.float32)
            k /= k.sum()
            return k

        rgb = np.array(image.convert("RGB"))
        k = disk_kernel(radius)
        blurred = np.dstack([cv2.filter2D(rgb[..., c], -1, k) for c in range(3)])
        m = np.array(self._soften_mask(bg_mask, feather_px=12, erode_px=2), dtype=np.float32) / 255.0
        m = m[..., None]
        out = (rgb * (1 - m) + blurred * m).astype(np.uint8)
        return Image.fromarray(out)

    def _blur_dof(self, image: Image.Image, bg_mask: Image.Image, max_radius: int = 25) -> Image.Image:
        rgb = np.array(image.convert("RGB"))
        subj = 255 - np.array(bg_mask.convert("L"), dtype=np.uint8)  # subject white
        dist = cv2.distanceTransform(255 - subj, cv2.DIST_L2, 3)
        dist = (dist / (dist.max() + 1e-6)).clip(0, 1)

        light = cv2.GaussianBlur(rgb, (9, 9), 0)
        heavy = cv2.GaussianBlur(rgb, (max_radius | 1, max_radius | 1), 0)
        w = dist[..., None].astype(np.float32)
        blended_bg = (light * (1 - w) + heavy * w).astype(np.uint8)

        m = np.array(self._soften_mask(bg_mask, feather_px=10, erode_px=2), dtype=np.float32) / 255.0
        m = m[..., None]
        out = (rgb * (1 - m) + blended_bg * m).astype(np.uint8)
        return Image.fromarray(out)

    def blur_background_only(
        self,
        image: Image.Image,
        mode: str = "gaussian",
        amount: int = 21,
        grow_px: int = 6,
        feather_px: int = 10,
    ) -> Image.Image:
        """
        Blur only the background (no generation).
        mode: 'gaussian' | 'bokeh' | 'dof'
        """
        work = self._resize_to_optimal(image)
        bg_mask = self._bg_edit_mask(work, grow_px=grow_px, feather_px=feather_px)

        if mode == "gaussian":
            out = self._blur_gaussian(work, bg_mask, radius=amount)
        elif mode == "bokeh":
            out = self._blur_bokeh(work, bg_mask, radius=max(6, amount // 3))
        elif mode == "dof":
            out = self._blur_dof(work, bg_mask, max_radius=amount)
        else:
            raise ValueError("mode must be 'gaussian', 'bokeh', or 'dof'")

        out = out.resize(image.size, Image.LANCZOS)
        return self.enhance(out)

    # ----------------------------- unified API --------------------------

    def edit_background(
        self,
        image: Image.Image,
        mode: str = "blur",  # Only blur mode now
        *,
        blur_radius: int = 21,
        blur_style: str = "dof",  # 'gaussian' | 'bokeh' | 'dof'
        **kwargs  # Ignore other params
    ) -> Image.Image:
        """
        Only blur mode is supported now.
        """
        if mode == "blur":
            return self.blur_background_only(image, mode=blur_style, amount=blur_radius)
        else:
            raise ValueError("Only 'blur' mode is supported")