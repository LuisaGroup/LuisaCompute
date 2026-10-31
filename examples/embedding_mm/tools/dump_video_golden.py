#!/usr/bin/env python3
"""
dump_video_golden.py -- Extract fine-grained golden representations and embeddings for Video.

Features:
1. Loads Google EmbeddingGemma-2 via PyTorch Transformers / SentenceTransformers.
2. Extracts layer-by-layer activations for sample_01_nature_ocean:
   - Rescaled & patchified inputs
   - ViT Layer 0 activations
   - ViT Layer 15 (Vision Tower) activations
   - Spatial Pooling layer (3x3 average pool)
   - Vision/Video Connector projection (Soft Tokens: [num_frames * 16, 512])
3. Computes end-to-end golden video embeddings across MRL dimensions (768, 512, 256, 128).
4. Computes text embeddings for captions, queries, and hard negatives.
5. Computes cross-modal video-text similarity matrices.
6. Computes multimodal video + text prompt embeddings (interleaved <|video|> sequence).
7. Exports everything to data/golden_video.json and data/video/golden_video_dataset.json.
"""

import argparse
import glob
import json
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoTokenizer, EmbeddingGemma2Model
from transformers.video_utils import VideoMetadata
from sentence_transformers import SentenceTransformer

def main():
    parser = argparse.ArgumentParser(description="Dump golden video reference for EmbeddingGemma-2")
    parser.add_argument("--model-dir", default=os.environ.get("EMBEDDINGGEMMA_MODEL_DIR", "models/embeddinggemma-2"), help="Model directory")
    parser.add_argument("--data-dir", default=r"examples/embedding_mm/data", help="Output data directory")
    args = parser.parse_args()

    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        if os.path.exists("../models/embeddinggemma-2"):
            model_dir = "../models/embeddinggemma-2"
        elif os.path.exists("models/embeddinggemma-2"):
            model_dir = "models/embeddinggemma-2"

    video_dir = os.path.join(args.data_dir, "video")
    meta_json_path = os.path.join(video_dir, "video_dataset.json")
    if not os.path.exists(meta_json_path):
        raise FileNotFoundError(f"Video dataset metadata not found at {meta_json_path}. Run build_video_dataset.py first.")

    with open(meta_json_path, "r", encoding="utf-8") as f:
        dataset_meta = json.load(f)

    print(f"Loading EmbeddingGemma-2 models from '{model_dir}'...")
    processor = AutoProcessor.from_pretrained(model_dir)
    hf_model = EmbeddingGemma2Model.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
    st_model = SentenceTransformer(model_dir, model_kwargs={'torch_dtype': torch.float32})

    golden_output = {
        "version": "1.0",
        "model": "google/embeddinggemma-2",
        "layer_activations": {},
        "samples": {},
        "multimodal_prompts": {},
    }

    # 1. Fine-grained layer activations for sample_01_nature_ocean
    print("\n--- 1. Extracting Fine-Grained Layer Activations (sample_01_nature_ocean) ---")
    ocean_fixture_dir = os.path.join(video_dir, "fixtures", "sample_01_nature_ocean")
    all_frames = [Image.open(p).convert("RGB") for p in sorted(glob.glob(os.path.join(ocean_fixture_dir, "*.png")))]
    # 4 frames sampled uniformly at 1 fps
    sampled_indices = [0, 4, 8, 12]
    sampled_frames = [all_frames[i] for i in sampled_indices]

    proc_res = processor(videos=[sampled_frames], return_tensors="pt")
    pixel_values = proc_res["pixel_values_videos"].float() # [num_frames, max_patches, 768]
    pos_ids = proc_res["video_position_ids"].long() # [num_frames, max_patches, 2]
    print(f"Processor output: pixel_values={pixel_values.shape}, pos_ids={pos_ids.shape}")

    vt = hf_model.vision_tower
    connector = hf_model.embed_vision

    padding_positions = (pos_ids == -1).any(dim=-1)
    # Patch embedding + positional embedding
    h_patch = vt.patch_embedder(pixel_values, pos_ids, padding_positions=padding_positions)
    print(f"Patch embeddings: {h_patch.shape}")

    # Full Vision Tower forward with all hidden states
    vt_res = vt(pixel_values, pos_ids, output_hidden_states=True)
    h_layer0 = vt_res.hidden_states[1] # After 1st ViT layer
    vt_out = vt_res.last_hidden_state  # After 16 ViT layers & spatial pooling
    print(f"ViT Layer 0: {h_layer0.shape}")
    print(f"Vision Tower output (pooled): {vt_out.shape}")

    # Connector Soft Tokens
    connector_out = connector(inputs_embeds=vt_out) # [num_frames * 16, 512]
    print(f"Connector Soft Tokens: {connector_out.shape}")

    def to_slice_list(t, n=16):
        arr = t.detach().cpu().numpy().reshape(-1)
        return [float(x) for x in arr[:n]]

    golden_output["layer_activations"]["sample_01_nature_ocean"] = {
        "sampled_indices": sampled_indices,
        "num_frames": len(sampled_frames),
        "pixel_values_slice": to_slice_list(pixel_values, 16),
        "patch_embed_slice": to_slice_list(h_patch, 16),
        "vit_layer0_slice": to_slice_list(h_layer0, 16),
        "vit_layer15_slice": to_slice_list(vt_out, 16),
        "connector_soft_tokens_slice": to_slice_list(connector_out, 16),
        "connector_soft_tokens_mean": float(connector_out.detach().mean()),
        "connector_soft_tokens_std": float(connector_out.detach().std()),
    }

    # 2. Extract End-to-end Embeddings for all 5 samples
    print("\n--- 2. Computing End-to-End Golden Embeddings for Benchmark Samples ---")
    for sample in dataset_meta["samples"]:
        sample_id = sample["id"]
        print(f"Processing sample: {sample_id}...")
        sample_fixture = os.path.join(video_dir, "fixtures", sample_id)
        frames = [Image.open(p).convert("RGB") for p in sorted(glob.glob(os.path.join(sample_fixture, "*.png")))]
        if len(frames) == 0:
            raise FileNotFoundError(f"No frames found in {sample_fixture}")

        # Standalone video embedding (sampled 4 frames)
        s_frames = [frames[i] for i in sampled_indices]
        emb_768 = st_model.encode({"video": s_frames}, normalize_embeddings=True)
        emb_512 = emb_768[:512] / np.linalg.norm(emb_768[:512])
        emb_256 = emb_768[:256] / np.linalg.norm(emb_768[:256])
        emb_128 = emb_768[:128] / np.linalg.norm(emb_768[:128])

        # Also encode all 16 frames as full sequence
        emb_all_768 = st_model.encode({"video": frames}, normalize_embeddings=True)

        # Captions & queries embeddings
        cap_en_emb = st_model.encode(sample["caption_en"], normalize_embeddings=True)
        cap_zh_emb = st_model.encode(sample["caption_zh"], normalize_embeddings=True)
        queries_en_emb = [st_model.encode(q, normalize_embeddings=True) for q in sample["queries_en"]]
        queries_zh_emb = [st_model.encode(q, normalize_embeddings=True) for q in sample["queries_zh"]]
        neg_en_emb = [st_model.encode(n, normalize_embeddings=True) for n in sample["hard_negatives_en"]]
        neg_zh_emb = [st_model.encode(n, normalize_embeddings=True) for n in sample["hard_negatives_zh"]]

        # Similarities
        sim_cap_en = float(np.dot(emb_768, cap_en_emb))
        sim_cap_zh = float(np.dot(emb_768, cap_zh_emb))
        sim_q_en = [float(np.dot(emb_768, qe)) for qe in queries_en_emb]
        sim_q_zh = [float(np.dot(emb_768, qz)) for qz in queries_zh_emb]
        sim_neg_en = [float(np.dot(emb_768, ne)) for ne in neg_en_emb]
        sim_neg_zh = [float(np.dot(emb_768, nz)) for nz in neg_zh_emb]

        print(f"  Caption EN sim: {sim_cap_en:.4f}, ZH sim: {sim_cap_zh:.4f}")
        print(f"  Queries EN sims: {[round(s, 4) for s in sim_q_en]}")
        print(f"  Hard Neg EN sims: {[round(s, 4) for s in sim_neg_en]}")

        # Multimodal Video + Text Prompt Embedding
        prompt_en = sample["video_prompt_en"]
        prompt_zh = sample["video_prompt_zh"]
        mm_emb_en = st_model.encode({"text": prompt_en, "video": s_frames}, normalize_embeddings=True)
        mm_emb_zh = st_model.encode({"text": prompt_zh, "video": s_frames}, normalize_embeddings=True)

        golden_output["samples"][sample_id] = {
            "embeddings": {
                "768": emb_768.tolist(),
                "512": emb_512.tolist(),
                "256": emb_256.tolist(),
                "128": emb_128.tolist(),
                "all_frames_768": emb_all_768.tolist(),
            },
            "captions": {
                "en": sample["caption_en"],
                "zh": sample["caption_zh"],
                "emb_en": cap_en_emb.tolist(),
                "emb_zh": cap_zh_emb.tolist(),
                "similarity_en": sim_cap_en,
                "similarity_zh": sim_cap_zh,
            },
            "queries": {
                "en": sample["queries_en"],
                "zh": sample["queries_zh"],
                "similarities_en": sim_q_en,
                "similarities_zh": sim_q_zh,
            },
            "hard_negatives": {
                "en": sample["hard_negatives_en"],
                "zh": sample["hard_negatives_zh"],
                "similarities_en": sim_neg_en,
                "similarities_zh": sim_neg_zh,
            },
            "multimodal_prompts": {
                "en": {
                    "prompt": prompt_en,
                    "768": mm_emb_en.tolist(),
                },
                "zh": {
                    "prompt": prompt_zh,
                    "768": mm_emb_zh.tolist(),
                }
            }
        }

    out_file1 = os.path.join(args.data_dir, "golden_video.json")
    out_file2 = os.path.join(video_dir, "golden_video_dataset.json")

    with open(out_file1, "w", encoding="utf-8") as f:
        json.dump(golden_output, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] Golden Video Data saved to: {out_file1}")

    with open(out_file2, "w", encoding="utf-8") as f:
        json.dump(golden_output, f, indent=2, ensure_ascii=False)
    print(f"[Saved] Duplicate saved to: {out_file2}")

if __name__ == "__main__":
    main()
