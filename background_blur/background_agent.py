# -*- coding: utf-8 -*-
"""
Author: Jaden Barnwell
November 4th, 2025

Thin wrapper exposing `main(img_or_path, instruction, **kwargs) -> PIL.Image`
so I can import `from background_agent import main as bg_main`
and call it 
"""

from typing import Union, Optional
from PIL import Image
from background_editor import BackgroundEditor  # the class I gave you earlier

# Keep a single editor around to avoid re-loading weights every call.
_EDITOR: Optional[BackgroundEditor] = None

def _get_editor() -> BackgroundEditor:
    global _EDITOR
    if _EDITOR is None:
        _EDITOR = BackgroundEditor(
            device=None,          # auto 'cuda' if available
            max_edit_side=1024,   # bump if you have VRAM
        )
    return _EDITOR

def _ensure_image(x: Union[str, Image.Image]) -> Image.Image:
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    ed = _get_editor()
    return ed.ensure_image(x).convert("RGB")

def main(
    img_or_path: Union[str, Image.Image],
    instruction: str = "blur background",  # Default to blur
    *,
    # Blur params
    blur_style: str = "dof",     # 'gaussian' | 'bokeh' | 'dof'
    blur_radius: int = 21,
    # Ignore inpaint params
    **kwargs
) -> Image.Image:
    """
    Only blur background is supported.
    Returns a PIL.Image (no saving).
    """
    img = _ensure_image(img_or_path)
    ed = _get_editor()

    # Always do blur regardless of instruction
    out = ed.edit_background(
        img,
        mode="blur",
        blur_style=blur_style,
        blur_radius=blur_radius,
    )
    return out
