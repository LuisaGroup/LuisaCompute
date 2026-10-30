#!/usr/bin/env python3
"""
dump_vision_golden.py -- Computes golden reference embeddings using PyTorch SentenceTransformers.

Generates ground-truth reference embeddings for the diverse multimodal benchmark dataset
and/or legacy synthetic unit test assets.

Features:
- Encodes all diverse benchmark images across MRL dimensions (768, 512, 256, 128).
- Encodes all bilingual annotations: short captions, detailed captions, hard negatives, and task prompts.
- Computes PyTorch cross-modal retrieval matrices, hard negative margins, and bilingual consistency.
- Dumps ground truth to data/diverse/golden_diverse_vision.json.
"""

import argparse
import json
import os
import sys
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer


def dump_legacy(model, base_data_dir: str):
    """Legacy 5-image dump for example_mm_embed_test.cpp compatibility."""
    out_json = os.path.join(base_data_dir, "golden_vision.json")
    print(f"\n[Legacy Mode] Dumping golden reference to: {out_json}")

    images = {
        "forest": os.path.join(base_data_dir, "forest.png"),
        "ocean": os.path.join(base_data_dir, "ocean.png"),
        "sunset": os.path.join(base_data_dir, "sunset.png"),
        "code": os.path.join(base_data_dir, "code.png"),
        "rect": os.path.join(base_data_dir, "rect.png"),
    }
    texts = {
        "forest": "A lush green forest with tall trees and green leaves",
        "ocean": "Deep blue ocean waters with waves in the sea",
        "sunset": "A warm sunset with an orange sun in the evening sky",
        "code": "Computer code in an editor syntax highlighting terminal",
        "rect": "A sunny sandy beach under a clear blue sky",
    }

    results = {"images": {}, "texts": {}, "similarities": {}}
    img_embs = {}
    for name, path in images.items():
        if not os.path.exists(path):
            print(f"  [WARN] Image {path} not found. Skipping legacy image.")
            continue
        img = Image.open(path).convert("RGB")
        emb = model.encode(img)
        img_embs[name] = emb
        results["images"][name] = {
            "768": emb.tolist(),
            "512": (emb[:512] / np.linalg.norm(emb[:512])).tolist(),
            "256": (emb[:256] / np.linalg.norm(emb[:256])).tolist(),
            "128": (emb[:128] / np.linalg.norm(emb[:128])).tolist(),
        }

    txt_embs = {}
    for name, text in texts.items():
        emb = model.encode(text)
        txt_embs[name] = emb
        results["texts"][name] = {"text": text, "768": emb.tolist()}

    for img_name in img_embs:
        results["similarities"][img_name] = {}
        for txt_name in txt_embs:
            sim = float(np.dot(img_embs[img_name], txt_embs[txt_name]))
            results["similarities"][img_name][txt_name] = sim

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Legacy golden reference saved to: {out_json}")


def dump_diverse(model, diverse_dir: str):
    """Computes comprehensive golden embeddings for diverse benchmark dataset."""
    dataset_json_path = os.path.join(diverse_dir, "diverse_dataset.json")
    out_json_path = os.path.join(diverse_dir, "golden_diverse_vision.json")

    if not os.path.exists(dataset_json_path):
        raise FileNotFoundError(f"Diverse dataset metadata not found at: {dataset_json_path}. Run build_diverse_dataset.py first.")

    with open(dataset_json_path, "r", encoding="utf-8") as f:
        dataset_meta = json.load(f)

    samples = dataset_meta.get("samples", [])
    print(f"\n============================================================")
    print(f" Computing Golden Reference for Diverse Benchmark ({len(samples)} samples)")
    print(f" Output Target: {out_json_path}")
    print(f"============================================================\n")

    results = {
        "version": "1.0",
        "num_samples": len(samples),
        "images": {},
        "texts": {},
        "cross_modal_matrix": {
            "short_en": {},
            "short_zh": {},
            "detailed_en": {},
            "detailed_zh": {},
        },
        "hard_negative_margins": {},
        "bilingual_consistency": {}
    }

    # 1. Encode all images
    print("[1/3] Encoding diverse benchmark images...")
    img_embs = {}
    for idx, item in enumerate(samples):
        item_id = item["id"]
        img_file = os.path.join(diverse_dir, item["image_filename"])
        if not os.path.exists(img_file):
            raise FileNotFoundError(f"Image file not found: {img_file}")

        img = Image.open(img_file).convert("RGB")
        emb = model.encode(img)  # 768-d unit-normalized
        emb_norm = np.linalg.norm(emb)
        img_embs[item_id] = emb

        emb_512 = emb[:512] / np.linalg.norm(emb[:512])
        emb_256 = emb[:256] / np.linalg.norm(emb[:256])
        emb_128 = emb[:128] / np.linalg.norm(emb[:128])

        results["images"][item_id] = {
            "filename": item["image_filename"],
            "category": item["category"],
            "aspect_ratio": item["aspect_ratio"],
            "width": item["width"],
            "height": item["height"],
            "norm": float(emb_norm),
            "768": emb.tolist(),
            "512": emb_512.tolist(),
            "256": emb_256.tolist(),
            "128": emb_128.tolist(),
        }
        print(f"  [{idx + 1:02d}/{len(samples)}] Image '{item_id}': shape={img.size}, norm={emb_norm:.4f}")

    # 2. Encode all text prompts (short, detailed, hard negatives, task prompts)
    print("\n[2/3] Encoding bilingual annotations (short, detailed, hard negatives, task prompts)...")
    txt_embs = {}

    for idx, item in enumerate(samples):
        item_id = item["id"]
        ann = item["annotations"]

        texts_to_encode = [
            (f"{item_id}_short_en", ann["short_caption"]["en"], "short_en"),
            (f"{item_id}_short_zh", ann["short_caption"]["zh"], "short_zh"),
            (f"{item_id}_detailed_en", ann["detailed_caption"]["en"], "detailed_en"),
            (f"{item_id}_detailed_zh", ann["detailed_caption"]["zh"], "detailed_zh"),
            (f"{item_id}_task_en", ann["task_prompt_retrieval"]["en"], "task_en"),
            (f"{item_id}_task_zh", ann["task_prompt_retrieval"]["zh"], "task_zh"),
        ]

        # Hard negatives
        for neg_i, neg in enumerate(ann["hard_negatives"]):
            texts_to_encode.append((f"{item_id}_neg_{neg_i}_en", neg["en"], f"neg_{neg_i}_en"))
            texts_to_encode.append((f"{item_id}_neg_{neg_i}_zh", neg["zh"], f"neg_{neg_i}_zh"))

        for key, text_val, text_type in texts_to_encode:
            emb = model.encode(text_val)
            txt_embs[key] = emb
            results["texts"][key] = {
                "sample_id": item_id,
                "type": text_type,
                "text": text_val,
                "768": emb.tolist(),
            }

        # Bilingual consistency check for this item
        cos_sim_short = float(np.dot(txt_embs[f"{item_id}_short_en"], txt_embs[f"{item_id}_short_zh"]))
        cos_sim_detailed = float(np.dot(txt_embs[f"{item_id}_detailed_en"], txt_embs[f"{item_id}_detailed_zh"]))
        results["bilingual_consistency"][item_id] = {
            "short_en_zh_sim": cos_sim_short,
            "detailed_en_zh_sim": cos_sim_detailed,
        }

    # 3. Compute Retrieval Metrics & Hard Negative Margins
    print("\n[3/3] Evaluating Golden Retrieval Metrics and Hard Negative Margins...")
    for q_type in ["short_en", "short_zh", "detailed_en", "detailed_zh"]:
        top1_hits = 0
        top3_hits = 0
        total = len(samples)

        for item_i in samples:
            id_i = item_i["id"]
            results["cross_modal_matrix"][q_type][id_i] = {}
            img_v = img_embs[id_i]

            sims = []
            for item_j in samples:
                id_j = item_j["id"]
                txt_v = txt_embs[f"{id_j}_{q_type}"]
                sim = float(np.dot(img_v, txt_v))
                results["cross_modal_matrix"][q_type][id_i][id_j] = sim
                sims.append((sim, id_j))

            # Sort descending
            sims.sort(key=lambda x: x[0], reverse=True)
            ranked_ids = [s[1] for s in sims]
            if ranked_ids[0] == id_i:
                top1_hits += 1
            if id_i in ranked_ids[:3]:
                top3_hits += 1

        top1_acc = (top1_hits / total) * 100.0
        top3_acc = (top3_hits / total) * 100.0
        print(f"  Retrieval [{q_type:11s}]: Top-1 = {top1_hits}/{total} ({top1_acc:.1f}%), Top-3 = {top3_hits}/{total} ({top3_acc:.1f}%)")

    # Hard-Negative Margins
    print("\nHard Negative Discrimination Margins:")
    margin_records = []
    for item in samples:
        item_id = item["id"]
        img_v = img_embs[item_id]
        pos_sim_en = float(np.dot(img_v, txt_embs[f"{item_id}_short_en"]))
        pos_sim_zh = float(np.dot(img_v, txt_embs[f"{item_id}_short_zh"]))

        # For all negatives of this item
        margins = []
        for neg_i, neg in enumerate(item["annotations"]["hard_negatives"]):
            neg_sim_en = float(np.dot(img_v, txt_embs[f"{item_id}_neg_{neg_i}_en"]))
            neg_sim_zh = float(np.dot(img_v, txt_embs[f"{item_id}_neg_{neg_i}_zh"]))
            margin_en = pos_sim_en - neg_sim_en
            margin_zh = pos_sim_zh - neg_sim_zh
            margins.append({
                "neg_index": neg_i,
                "margin_en": margin_en,
                "margin_zh": margin_zh,
                "pos_sim_en": pos_sim_en,
                "neg_sim_en": neg_sim_en,
                "pos_sim_zh": pos_sim_zh,
                "neg_sim_zh": neg_sim_zh,
            })
            margin_records.append(margin_en)
            margin_records.append(margin_zh)

        results["hard_negative_margins"][item_id] = margins
        min_m = min(m["margin_en"] for m in margins)
        print(f"  {item_id:32s} | Pos Sim: {pos_sim_en:.4f} | Min Neg Margin (EN): {min_m:+.4f}")

    avg_margin = float(np.mean(margin_records))
    min_margin = float(np.min(margin_records))
    print(f"\nOverall Hard-Negative Margin: Avg = {avg_margin:+.4f}, Min = {min_margin:+.4f}")

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved golden diverse vision reference to: {out_json_path} ({os.path.getsize(out_json_path):,} bytes)")


def main():
    parser = argparse.ArgumentParser(description="Dump golden vision embeddings using PyTorch SentenceTransformer.")
    parser.add_argument("--model-dir", type=str, default="models/embeddinggemma-2",
                        help="Path to EmbeddingGemma-2 model directory")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to data directory (defaults to examples/mm_embedding/data)")
    parser.add_argument("--legacy", action="store_true",
                        help="Dump legacy golden_vision.json for unit tests")
    parser.add_argument("--all", action="store_true", default=True,
                        help="Dump both diverse and legacy golden references (default)")
    args = parser.parse_args()

    if args.data_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.data_dir = os.path.join(os.path.dirname(script_dir), "data")

    diverse_dir = os.path.join(args.data_dir, "diverse")

    print(f"Loading SentenceTransformer model from '{args.model_dir}'...")
    model = SentenceTransformer(args.model_dir)

    if args.legacy:
        dump_legacy(model, args.data_dir)
    else:
        # Default: dump diverse benchmark golden
        dump_diverse(model, diverse_dir)
        # Also ensure legacy exists
        if not os.path.exists(os.path.join(args.data_dir, "golden_vision.json")):
            dump_legacy(model, args.data_dir)


if __name__ == "__main__":
    main()
