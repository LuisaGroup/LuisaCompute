#!/usr/bin/env python3
"""
build_video_dataset.py -- Generates synthetic benchmark video fixtures for EmbeddingGemma-2.

Creates 5 diverse short video samples:
  1. sample_01_nature_ocean: Rolling blue ocean waves with sunlight reflections.
  2. sample_02_human_walking: Silhouette human figure walking across a green park.
  3. sample_03_animated_star: Rotating and pulsating colorful geometric star.
  4. sample_04_static_slides: Presentation slides with bullet points appearing sequentially.
  5. sample_05_camera_zoom: Concentric geometric circles continuously zooming inward.

For each sample:
  - Generates PNG frame sequences in data/video/fixtures/<name>/frame_%04d.png
  - Generates standalone MP4 video in data/video/<name>.mp4
  - Saves bilingual annotations and search queries to data/video/video_dataset.json
"""

import os
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

FPS = 4.0
NUM_FRAMES = 16
WIDTH = 256
HEIGHT = 256

def render_ocean_frame(frame_idx: int, num_frames: int) -> Image.Image:
    """Frame of rolling ocean waves."""
    img = Image.new("RGB", (WIDTH, HEIGHT), color=(15, 30, 70))
    draw = ImageDraw.Draw(img)
    t = frame_idx / num_frames

    # Sky gradient
    for y in range(HEIGHT // 2):
        r = int(100 + 40 * (y / (HEIGHT // 2)))
        g = int(180 + 50 * (y / (HEIGHT // 2)))
        b = int(230 + 25 * (y / (HEIGHT // 2)))
        draw.line([(0, y), (WIDTH, y)], fill=(r, g, b))

    # Sun
    sun_x = WIDTH // 2 + int(30 * math.sin(t * 2 * math.pi))
    sun_y = 50
    draw.ellipse([sun_x - 18, sun_y - 18, sun_x + 18, sun_y + 18], fill=(255, 240, 160))

    # Sea horizon and waves
    sea_top = HEIGHT // 2
    for layer in range(5):
        y_base = sea_top + layer * 25
        wave_color = (10 + layer * 15, 45 + layer * 20, 120 + layer * 25)
        crest_color = (200, 230, 255)
        points = [(0, HEIGHT)]
        for x in range(0, WIDTH + 1, 4):
            freq = 0.04 + layer * 0.015
            phase = t * 2 * math.pi * 2.0 + layer * 1.2
            y_wave = y_base + 8 * math.sin(x * freq + phase)
            points.append((x, int(y_wave)))
        points.append((WIDTH, HEIGHT))
        draw.polygon(points, fill=wave_color)

        # Draw wave crest highlight
        for i in range(len(points) - 3):
            p1 = points[i + 1]
            p2 = points[i + 2]
            if p1[1] < y_base:
                draw.line([p1, p2], fill=crest_color, width=2)
    return img

def render_walking_frame(frame_idx: int, num_frames: int) -> Image.Image:
    """Frame of a human silhouette walking horizontally."""
    img = Image.new("RGB", (WIDTH, HEIGHT), color=(135, 206, 235))
    draw = ImageDraw.Draw(img)
    t = frame_idx / num_frames

    # Ground (green grass)
    draw.rectangle([0, 190, WIDTH, HEIGHT], fill=(50, 160, 60))
    # Path
    draw.rectangle([0, 215, WIDTH, HEIGHT], fill=(160, 130, 90))

    # Person horizontal progress
    px = int((frame_idx / (num_frames - 1)) * (WIDTH - 80) + 40)
    py = 140

    walk_phase = t * 4 * math.pi
    leg_spread = math.sin(walk_phase) * 16
    arm_spread = -math.sin(walk_phase) * 12

    # Head
    draw.ellipse([px - 10, py - 35, px + 10, py - 15], fill=(30, 30, 30))
    # Body
    draw.line([(px, py - 15), (px, py + 25)], fill=(30, 30, 30), width=6)
    # Left Leg & Right Leg
    draw.line([(px, py + 25), (px + int(leg_spread), py + 65)], fill=(30, 30, 30), width=5)
    draw.line([(px, py + 25), (px - int(leg_spread), py + 65)], fill=(30, 30, 30), width=5)
    # Arms
    draw.line([(px, py), (px + int(arm_spread), py + 30)], fill=(30, 30, 30), width=4)
    draw.line([(px, py), (px - int(arm_spread), py + 30)], fill=(30, 30, 30), width=4)

    return img

def render_star_frame(frame_idx: int, num_frames: int) -> Image.Image:
    """Frame of rotating and pulsating geometric star."""
    img = Image.new("RGB", (WIDTH, HEIGHT), color=(10, 10, 25))
    draw = ImageDraw.Draw(img)
    t = frame_idx / num_frames
    angle_offset = t * 2 * math.pi
    scale = 0.8 + 0.3 * math.sin(t * 2 * math.pi * 2.0)

    cx, cy = WIDTH // 2, HEIGHT // 2
    r_outer = 90 * scale
    r_inner = 40 * scale
    pts = []
    for i in range(10):
        angle = angle_offset + i * (math.pi / 5) - math.pi / 2
        r = r_outer if (i % 2 == 0) else r_inner
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        pts.append((x, y))

    # Glow layer
    draw.polygon(pts, fill=(255, 215, 0), outline=(255, 100, 0), width=3)
    # Inner star
    inner_pts = []
    for i in range(10):
        angle = -angle_offset + i * (math.pi / 5) - math.pi / 2
        r = (r_outer * 0.45) if (i % 2 == 0) else (r_inner * 0.45)
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        inner_pts.append((x, y))
    draw.polygon(inner_pts, fill=(255, 80, 80))
    return img

def render_slides_frame(frame_idx: int, num_frames: int) -> Image.Image:
    """Frame of presentation slide with progressive bullets."""
    img = Image.new("RGB", (WIDTH, HEIGHT), color=(245, 245, 250))
    draw = ImageDraw.Draw(img)

    # Slide Header
    draw.rectangle([15, 15, WIDTH - 15, 55], fill=(41, 128, 185))
    draw.text((25, 25), "EmbeddingGemma-2 Video", fill=(255, 255, 255))

    # Bullets appear at frames 0, 4, 8, 12
    bullets = [
        ("Frame 1: Zero-drift Video Features", (30, 30, 30)),
        ("Frame 2: Media Foundation Decoder", (30, 30, 30)),
        ("Frame 3: Temporal Uniform Sampling", (30, 30, 30)),
        ("Frame 4: Multimodal Sequence Alignment", (30, 30, 30)),
    ]

    visible_count = min(len(bullets), (frame_idx // 4) + 1)
    y_pos = 75
    for i in range(visible_count):
        text, color = bullets[i]
        # Bullet dot
        draw.ellipse([30, y_pos + 4, 38, y_pos + 12], fill=(230, 126, 34))
        draw.text((45, y_pos), text, fill=color)
        # Highlight box for newest bullet
        if i == visible_count - 1:
            draw.rectangle([25, y_pos - 2, WIDTH - 25, y_pos + 20], outline=(52, 152, 219), width=1)
        y_pos += 38

    return img

def render_zoom_frame(frame_idx: int, num_frames: int) -> Image.Image:
    """Frame of concentric circles zooming in."""
    img = Image.new("RGB", (WIDTH, HEIGHT), color=(20, 20, 30))
    draw = ImageDraw.Draw(img)
    t = frame_idx / num_frames

    cx, cy = WIDTH // 2, HEIGHT // 2
    colors = [
        (231, 76, 60),
        (241, 196, 15),
        (46, 204, 113),
        (52, 152, 219),
        (155, 89, 182),
        (26, 188, 156),
    ]

    base_r = 180
    num_circles = 8
    zoom_factor = 2.0 ** (t)

    for i in range(num_circles, 0, -1):
        r = (base_r / (2.0 ** (i * 0.4))) * zoom_factor
        color = colors[i % len(colors)]
        if r < 180:
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=4)

    # Center target dot
    draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=(255, 255, 255))
    return img

SAMPLES = [
    {
        "id": "sample_01_nature_ocean",
        "renderer": render_ocean_frame,
        "caption_en": "Calm ocean waves rolling gently across blue water under sunlight",
        "caption_zh": "在阳光下蔚蓝海面上轻轻翻滚的平静海浪",
        "detailed_en": "A wide angle view of an open blue sea with gentle rolling surface waves reflecting bright sunlight beneath a daytime sky.",
        "detailed_zh": "广阔碧蓝大海的广角画面，白天天空下平静的海面波浪在阳光照射下泛起波光。",
        "queries_en": [
            "blue ocean waves in daylight",
            "gentle sea waves rolling",
            "sunny ocean water surface",
        ],
        "queries_zh": [
            "白天蓝色的海洋波浪",
            "海面平静翻滚的波浪",
            "阳光下的碧蓝海水",
        ],
        "hard_negatives_en": [
            "dry arid desert sand dunes under hot sun",
            "stormy dark lightning clouds during night hurricane",
            "crowded metropolitan highway traffic jam",
        ],
        "hard_negatives_zh": [
            "烈日下干燥荒凉的沙漠沙丘",
            "夜间飓风暴雨中的黑色雷暴乌云",
            "拥挤繁华的都市高速公路交通拥堵",
        ],
        "video_prompt_en": "A video of ocean waves: <|video|>",
        "video_prompt_zh": "一段海洋波浪的视频：<|video|>",
    },
    {
        "id": "sample_02_human_walking",
        "renderer": render_walking_frame,
        "caption_en": "A person silhouette walking across a pathway in a park",
        "caption_zh": "一个在公园小径上行走的人影轮廓",
        "detailed_en": "A side view animation showing a human figure walking steadily from left to right along a ground path against a bright sky.",
        "detailed_zh": "一个侧面视角的动画，展现一个人形轮廓沿着地面小径在晴朗天空下稳健地从左向右行走。",
        "queries_en": [
            "a person walking outdoors",
            "silhouette figure walking across screen",
            "pedestrian stroll in park",
        ],
        "queries_zh": [
            "一个人在户外行走",
            "屏幕上行走的剪影人影",
            "在公园中漫步的行人",
        ],
        "hard_negatives_en": [
            "a parked stationary automobile in a garage",
            "empty classroom with wooden desks and chalkboard",
            "underwater coral reef with tropical fish swimming",
        ],
        "hard_negatives_zh": [
            "停在车库里静止不动的汽车",
            "摆放着木制桌椅和黑板的空旷教室",
            "热带鱼游弋其中的水下珊瑚礁",
        ],
        "video_prompt_en": "A video of a person walking: <|video|>",
        "video_prompt_zh": "一段人员行走的视频：<|video|>",
    },
    {
        "id": "sample_03_animated_star",
        "renderer": render_star_frame,
        "caption_en": "A glowing golden geometric star rotating and pulsating",
        "caption_zh": "一个发光的金色几何五角星旋转并跳动",
        "detailed_en": "A vibrant yellow and gold five-point geometric star spinning clockwise while smoothly expanding and contracting in size.",
        "detailed_zh": "一个鲜艳的黄色金色五角几何星沿顺时针方向旋转，同时平滑地放大与缩小跳动。",
        "queries_en": [
            "spinning gold star animation",
            "pulsating geometric star shape",
            "rotating glowing star graphic",
        ],
        "queries_zh": [
            "旋转的金色五角星动画",
            "有节奏跳动的几何五角星",
            "发光旋转的星星图案",
        ],
        "hard_negatives_en": [
            "greyscale medical chest x-ray scan",
            "plate of food on a wooden dining table",
            "printed legal contract document with signature lines",
        ],
        "hard_negatives_zh": [
            "灰度的医疗胸部X射线扫描图像",
            "摆放在木质餐桌上的一盘食物",
            "带有签字栏的打印法律合同文件",
        ],
        "video_prompt_en": "An animation of a rotating star: <|video|>",
        "video_prompt_zh": "一个旋转星星的动画：<|video|>",
    },
    {
        "id": "sample_04_static_slides",
        "renderer": render_slides_frame,
        "caption_en": "A business presentation slide with sequential bullet points",
        "caption_zh": "包含逐条出现的要点的商务演示幻灯片",
        "detailed_en": "A digital presentation slide layout with a blue top header bar displaying progressive bullet point text items appearing step by step.",
        "detailed_zh": "带有蓝色顶部标题栏的数字幻灯片版式，依次显示逐步出现的项目符号文字要点。",
        "queries_en": [
            "presentation slides with bullet points",
            "digital lecture slide deck animation",
            "business slides text list appearing",
        ],
        "queries_zh": [
            "带有项目符号的演讲演示幻灯片",
            "数字学术报告幻灯片动画",
            "逐步出现的商务列表幻灯片",
        ],
        "hard_negatives_en": [
            "wild lion running across African savannah",
            "snowy mountain peak surrounded by pine trees",
            "macro photography of dew drops on red flower petals",
        ],
        "hard_negatives_zh": [
            "在非洲稀树草原上奔跑的野生狮子",
            "被松树环绕的积雪山峰",
            "红色花瓣上露珠的微距摄影",
        ],
        "video_prompt_en": "A video of presentation slides: <|video|>",
        "video_prompt_zh": "演示幻灯片的视频：<|video|>",
    },
    {
        "id": "sample_05_camera_zoom",
        "renderer": render_zoom_frame,
        "caption_en": "Colorful concentric geometric circles zooming into center",
        "caption_zh": "多彩的同心几何圆形不断向中心缩放推进",
        "detailed_en": "Hypnotic motion graphics of multi-colored concentric circles continuously scaling and zooming toward a central focal point.",
        "detailed_zh": "迷幻动效图形，多色同心圆圈持续向中心对焦点缩放推进。",
        "queries_en": [
            "concentric circles zooming in",
            "colorful geometric optical zoom animation",
            "motion graphic rings expanding inward",
        ],
        "queries_zh": [
            "同心圆向内推镜头缩放",
            "多彩几何光学变焦动画",
            "向中心缩放的动态光圈环",
        ],
        "hard_negatives_en": [
            "handwritten recipe note on parchment paper",
            "rocky stone wall with green moss growth",
            "modern sleek passenger airplane landing on runway",
        ],
        "hard_negatives_zh": [
            "羊皮纸上的手写烹饪食谱",
            "长满绿苔藓的岩石石墙",
            "在跑道上降落的现代流线型客机",
        ],
        "video_prompt_en": "A zoom animation of concentric rings: <|video|>",
        "video_prompt_zh": "同心圆环的缩放动画：<|video|>",
    },
]

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_dir = os.path.join(base_dir, "data", "video")
    fixtures_dir = os.path.join(video_dir, "fixtures")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(fixtures_dir, exist_ok=True)

    metadata = {
        "dataset_name": "EmbeddingGemma-2 Video Benchmark Fixtures",
        "version": "1.0",
        "fps": FPS,
        "num_frames_per_video": NUM_FRAMES,
        "resolution": [WIDTH, HEIGHT],
        "samples": []
    }

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    for sample in SAMPLES:
        sample_id = sample["id"]
        renderer = sample["renderer"]
        sample_fixture_dir = os.path.join(fixtures_dir, sample_id)
        os.makedirs(sample_fixture_dir, exist_ok=True)

        mp4_path = os.path.join(video_dir, f"{sample_id}.mp4")
        video_writer = cv2.VideoWriter(mp4_path, fourcc, FPS, (WIDTH, HEIGHT))

        frame_paths = []
        for f in range(NUM_FRAMES):
            pil_img = renderer(f, NUM_FRAMES)
            png_name = f"frame_{f:04d}.png"
            png_path = os.path.join(sample_fixture_dir, png_name)
            pil_img.save(png_path)
            frame_paths.append(os.path.relpath(png_path, base_dir).replace("\\", "/"))

            # Convert RGB to BGR for cv2
            bgr_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            video_writer.write(bgr_cv)

        video_writer.release()
        print(f"[Generated] {sample_id}: {NUM_FRAMES} frames -> {mp4_path} ({os.path.getsize(mp4_path)} bytes)")

        sample_meta = {
            "id": sample_id,
            "mp4_file": os.path.relpath(mp4_path, base_dir).replace("\\", "/"),
            "frames_dir": os.path.relpath(sample_fixture_dir, base_dir).replace("\\", "/"),
            "num_frames": NUM_FRAMES,
            "fps": FPS,
            "duration_sec": NUM_FRAMES / FPS,
            "caption_en": sample["caption_en"],
            "caption_zh": sample["caption_zh"],
            "detailed_en": sample["detailed_en"],
            "detailed_zh": sample["detailed_zh"],
            "queries_en": sample["queries_en"],
            "queries_zh": sample["queries_zh"],
            "hard_negatives_en": sample["hard_negatives_en"],
            "hard_negatives_zh": sample["hard_negatives_zh"],
            "video_prompt_en": sample["video_prompt_en"],
            "video_prompt_zh": sample["video_prompt_zh"],
        }
        metadata["samples"].append(sample_meta)

    json_path = os.path.join(video_dir, "video_dataset.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"\nSaved benchmark dataset metadata to: {json_path}")

if __name__ == "__main__":
    main()
