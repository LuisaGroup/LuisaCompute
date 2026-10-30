#!/usr/bin/env python3
"""
generate_test_multimodal_data.py -- Generates lightweight, reproducible multimodal test assets.

Creates synthetic test images with high semantic contrast for automated cross-modal
retrieval (Image-to-Text and Text-to-Image) and numerical parity verification.
"""

import os
from PIL import Image, ImageDraw

def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    print(f"Generating test images in: {data_dir}")

    # 1. Forest / Green nature image
    img_forest = Image.new("RGB", (224, 224), color=(34, 139, 34)) # Forest green
    draw = ImageDraw.Draw(img_forest)
    # Add green foliage patterns and tree trunks
    draw.rectangle([90, 120, 134, 224], fill=(101, 67, 33)) # Tree trunk brown
    draw.ellipse([50, 40, 174, 160], fill=(0, 100, 0))      # Dark green canopy
    draw.ellipse([70, 20, 154, 110], fill=(50, 205, 50))    # Lime green highlights
    forest_path = os.path.join(data_dir, "forest.png")
    img_forest.save(forest_path)
    print(f"Saved: {forest_path}")

    # 2. Ocean / Blue marine image
    img_ocean = Image.new("RGB", (224, 224), color=(0, 105, 148)) # Ocean blue
    draw = ImageDraw.Draw(img_ocean)
    # Add wave patterns and deep sea gradient
    draw.rectangle([0, 120, 224, 224], fill=(0, 49, 83))     # Deep navy water
    draw.ellipse([20, 80, 204, 150], fill=(64, 224, 208))    # Cyan wave crest
    draw.ellipse([40, 130, 184, 180], fill=(0, 128, 255))   # Electric blue shimmer
    ocean_path = os.path.join(data_dir, "ocean.png")
    img_ocean.save(ocean_path)
    print(f"Saved: {ocean_path}")

    # 3. Sunset / Warm orange sky image
    img_sunset = Image.new("RGB", (224, 224), color=(255, 69, 0)) # Red-orange sunset
    draw = ImageDraw.Draw(img_sunset)
    draw.ellipse([70, 70, 154, 154], fill=(255, 215, 0))    # Golden sun
    draw.rectangle([0, 160, 224, 224], fill=(128, 0, 32))    # Dark burgundy horizon
    sunset_path = os.path.join(data_dir, "sunset.png")
    img_sunset.save(sunset_path)
    print(f"Saved: {sunset_path}")

    # 4. Code / Syntax terminal image
    img_code = Image.new("RGB", (224, 224), color=(30, 30, 30))  # Dark editor
    draw = ImageDraw.Draw(img_code)
    # Simulated lines of code in colorful syntax blocks
    colors = [(86, 156, 214), (206, 145, 120), (220, 220, 170), (106, 153, 85)]
    for row, y in enumerate(range(30, 200, 24)):
        c = colors[row % len(colors)]
        draw.rectangle([30, y, 110 + (row * 15) % 80, y + 10], fill=c)
        draw.rectangle([130 + (row * 10) % 40, y, 190, y + 10], fill=(150, 150, 150))
    # 5. Non-square rectangular image (Beach: 400x200 landscape)
    img_rect = Image.new("RGB", (400, 200), color=(135, 206, 235)) # Sky blue
    draw = ImageDraw.Draw(img_rect)
    draw.rectangle([0, 120, 400, 200], fill=(238, 214, 175))       # Sand yellow
    draw.ellipse([300, 20, 360, 80], fill=(255, 255, 0))            # Bright sun
    rect_path = os.path.join(data_dir, "rect.png")
    img_rect.save(rect_path)
    print(f"Saved: {rect_path}")

    print("All test images generated successfully.")

if __name__ == "__main__":
    main()
