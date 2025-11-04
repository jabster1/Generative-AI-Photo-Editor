"""
Author: Jaden Barnwell
November 4th, 2025

Script to initiate the two background blur setup and script. Input your image and prompt
"""

from PIL import Image
from background_agent import main as bg_main

def run_demo():
    #img = Image.open("horseandman.jpg").convert("RGB")
    img = Image.open("manboat.jpg").convert("RGB")
    prompt = "blur background"

    # Bokeh blur background
    out_bokeh = bg_main(img, prompt, blur_style="bokeh", blur_radius=15)
    out_bokeh.save("debug_bokeh.png")
    print("✓ Bokeh blur completed")

    # Depth of field blur
    out_dof = bg_main(img, prompt, blur_style="dof", blur_radius=30)
    out_dof.save("debug_dof.png")
    print("✓ DOF blur completed")

    #out_dof or out_bokeh are our output or returned images
    #less aggressive blur is dof

if __name__ == "__main__":
    run_demo()