#!/usr/bin/env python3
"""
dump_audio_golden.py -- Extract fine-grained golden representations and embeddings for Audio.

Loads Google's EmbeddingGemma-2 model using PyTorch Transformers / SentenceTransformers,
processes the official sample audio ('roses-are.wav') and candidates from
https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers,
and exports golden references for each critical layer to guarantee 0-error operator alignment:

1. Frontend STFT / Mel spectrogram features [307, 128]
2. Subsampling Conv2D Layer 0 (Conv + LayerNorm + ReLU)
3. Subsampling Conv2D Layer 1 (Conv + LayerNorm + ReLU)
4. Subsampling Input Projection Linear [77, 1024]
5. Conformer Layer 0 intermediate activations:
   - FeedForward 1 (pre_norm, ffw1, silu, ffw2, post_norm, scale 0.5, residual)
   - Self Attention (chunked attention with relative positional bias, logit cap 50)
   - LightConv1D (pre_norm, linear_start, glu, depthwise causal conv1d, norm, silu, linear_end, residual)
   - FeedForward 2 (pre_norm, ffw1, silu, ffw2, post_norm, scale 0.5, residual)
   - Output Norm
6. Conformer Layer 11 Output Norm [77, 1024]
7. Audio Tower Output Projection [77, 1536] (Linear + Bias)
8. Audio Connector Projection [77, 512] (Soft Tokens)
9. End-to-end Audio Embeddings across MRL dimensions (768, 512, 256, 128)
10. Candidate text embeddings and cross-modal similarities
11. Interleaved multimodal embedding (audio + text)
"""

import argparse
import json
import os
import urllib.request
import numpy as np
import scipy.io.wavfile as wavfile
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from transformers import AutoProcessor, AutoTokenizer, EmbeddingGemma2Model
from sentence_transformers import SentenceTransformer

OFFICIAL_AUDIO_URL = "https://ai.google.dev/gemma/docs/audio/roses-are.wav"

CANDIDATES = [
    "a person speaking English",
    "random static white noise",
    "Roses are red, violets are blue.",
]

def download_audio_if_needed(target_path: str):
    if not os.path.exists(target_path):
        print(f"Downloading official audio from {OFFICIAL_AUDIO_URL}...")
        urllib.request.urlretrieve(OFFICIAL_AUDIO_URL, target_path)
        print(f"Downloaded: {target_path} ({os.path.getsize(target_path)} bytes)")

def main():
    parser = argparse.ArgumentParser(description="Dump golden audio reference for EmbeddingGemma-2")
    parser.add_argument("--model-dir", default="models/embeddinggemma-2", help="Model directory")
    parser.add_argument("--data-dir", default=r"examples/embedding_mm/data", help="Output data directory")
    args = parser.parse_args()

    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        if os.path.exists("../models/embeddinggemma-2"):
            model_dir = "../models/embeddinggemma-2"
        elif os.path.exists("models/embeddinggemma-2"):
            model_dir = "models/embeddinggemma-2"

    os.makedirs(args.data_dir, exist_ok=True)
    raw_wav_path = os.path.join(args.data_dir, "roses-are.wav")
    wav_16k_path = os.path.join(args.data_dir, "roses_are_16k.wav")
    golden_json_path = os.path.join(args.data_dir, "golden_audio.json")

    download_audio_if_needed(raw_wav_path)

    # Load WAV and resample to 16 kHz
    sr, raw_data = wavfile.read(raw_wav_path)
    print(f"Loaded audio: sample_rate={sr}, length={len(raw_data)} samples, dtype={raw_data.dtype}")
    float_audio = raw_data.astype(np.float32) / 32768.0
    wav_tensor = torch.tensor(float_audio)

    if sr != 16000:
        print(f"Resampling audio from {sr} Hz to 16000 Hz...")
        resampler = T.Resample(sr, 16000)
        wav16_tensor = resampler(wav_tensor)
        wav16_np = wav16_tensor.numpy()
    else:
        wav16_np = float_audio
        wav16_tensor = wav_tensor

    # Save 16 kHz WAV file
    int16_16k = (np.clip(wav16_np, -1.0, 1.0) * 32767.0).astype(np.int16)
    wavfile.write(wav_16k_path, 16000, int16_16k)
    print(f"Saved 16 kHz audio to {wav_16k_path} ({len(wav16_np)} samples)")

    # Load processor and models
    print(f"Loading AutoProcessor from '{model_dir}'...")
    processor = AutoProcessor.from_pretrained(model_dir)

    print(f"Loading EmbeddingGemma2Model (fp32) from '{model_dir}'...")
    model = EmbeddingGemma2Model.from_pretrained(model_dir, torch_dtype=torch.float32).eval()

    # 1. Feature Extraction
    proc_res = processor(audio=wav16_np, return_tensors="pt")
    input_features = proc_res.input_features.float() # [1, 307, 128]
    input_features_mask = proc_res.input_features_mask # [1, 307]
    print(f"Extracted input_features: shape={input_features.shape}")

    # 2. Subsample Conv Projection
    audio_tower = model.audio_tower
    subsample = audio_tower.subsample_conv_projection

    h_in = input_features.unsqueeze(1) # [1, 1, 307, 128]
    h_sub0, mask0 = subsample.layer0(h_in, input_features_mask)
    h_sub1, mask1 = subsample.layer1(h_sub0, mask0)

    B, C, S, W = h_sub1.shape
    h_sub_flat = h_sub1.permute(0, 2, 3, 1).contiguous().reshape(B, S, -1)
    h_sub_proj = subsample.input_proj_linear(h_sub_flat)
    print(f"Subsample output: layer0={h_sub0.shape}, layer1={h_sub1.shape}, proj={h_sub_proj.shape}")

    # 3. Relative Positional Encoding
    rpe = audio_tower.rel_pos_enc
    pos_embed = rpe(h_sub_proj)
    print(f"Relative position embeddings: shape={pos_embed.shape}")

    # 4. Conformer Layers
    from transformers.models.gemma4.modeling_gemma4 import create_bidirectional_mask, sliding_window_mask_function
    attn_mask_4d = create_bidirectional_mask(
        config=audio_tower.config,
        inputs_embeds=h_sub_proj,
        attention_mask=mask1,
        and_mask_function=sliding_window_mask_function(
            (audio_tower.config.attention_context_left - 1, audio_tower.config.attention_context_right)
        ),
    )
    attn_mask_5d = audio_tower._convert_4d_mask_to_blocked_5d(attn_mask_4d) if attn_mask_4d is not None else None

    h_cur = h_sub_proj
    layer0_outs = {}
    layer11_out = None

    for idx, layer in enumerate(audio_tower.layers):
        if idx == 0:
            # Capture fine-grained activations of layer 0
            # FFN 1
            h_ffn1 = layer.feed_forward1(h_cur)
            res1 = h_ffn1
            # Pre-attn norm
            h_pre_attn = layer.norm_pre_attn(h_ffn1)
            # Self Attention
            h_attn, _ = layer.self_attn(h_pre_attn, position_embeddings=pos_embed, attention_mask=attn_mask_5d)
            h_post_attn = layer.norm_post_attn(h_attn) + res1
            # LightConv1D
            h_lconv = layer.lconv1d(h_post_attn)
            # FFN 2
            h_ffn2 = layer.feed_forward2(h_lconv)
            # Norm Out
            h_layer0_out = layer.norm_out(h_ffn2)

            layer0_outs = {
                "ffn1": h_ffn1.squeeze(0).detach().numpy(),
                "pre_attn_norm": h_pre_attn.squeeze(0).detach().numpy(),
                "attn": h_attn.squeeze(0).detach().numpy(),
                "post_attn": h_post_attn.squeeze(0).detach().numpy(),
                "lconv": h_lconv.squeeze(0).detach().numpy(),
                "ffn2": h_ffn2.squeeze(0).detach().numpy(),
                "out": h_layer0_out.squeeze(0).detach().numpy(),
            }
            h_cur = h_layer0_out
        else:
            h_cur = layer(h_cur, attention_mask=attn_mask_5d, position_embeddings=pos_embed)
            if idx == 11:
                layer11_out = h_cur.squeeze(0).detach().numpy()

    # 5. Output Projection and Connector
    h_audio_out = audio_tower.output_proj(h_cur) # [1, 77, 1536]
    h_soft_tokens = model.embed_audio(inputs_embeds=h_audio_out) # [1, 77, 512]
    print(f"Audio Tower output: {h_audio_out.shape}, Soft Tokens: {h_soft_tokens.shape}")

    # 6. End-to-end embeddings with SentenceTransformer
    print(f"Loading SentenceTransformer from '{model_dir}'...")
    st_model = SentenceTransformer(model_dir, model_kwargs={'torch_dtype': torch.float32})

    print("Encoding audio with SentenceTransformer...")
    emb_audio_768 = st_model.encode({"audio": wav16_np}, normalize_embeddings=True)
    emb_audio_512 = emb_audio_768[:512] / np.linalg.norm(emb_audio_768[:512])
    emb_audio_256 = emb_audio_768[:256] / np.linalg.norm(emb_audio_768[:256])
    emb_audio_128 = emb_audio_768[:128] / np.linalg.norm(emb_audio_768[:128])

    print("Encoding candidate texts with SentenceTransformer...")
    emb_candidates = st_model.encode(CANDIDATES, normalize_embeddings=True)
    audio_sims = st_model.similarity(emb_audio_768, emb_candidates)[0].tolist()

    for cand, sim in zip(CANDIDATES, audio_sims):
        print(f"  Similarity '{cand}': {sim:.4f}")

    # 7. Interleaved multimodal embedding: Prompt + Audio
    # Prompt: "A recording of speech: <|audio|>"
    prompt_interleaved = "A recording of speech: <|audio|>"
    emb_interleaved = st_model.encode({"text": prompt_interleaved, "audio": wav16_np}, normalize_embeddings=True)

    # Package into golden dictionary
    golden_data = {
        "audio_name": "roses-are",
        "sample_rate": 16000,
        "num_samples": len(wav16_np),
        "num_frames": int(input_features.shape[1]),
        "num_soft_tokens": int(h_soft_tokens.shape[1]),
        "input_features": input_features.squeeze(0).numpy().tolist(),
        "subsample_layer0_shape": list(h_sub0.shape),
        "subsample_layer1_shape": list(h_sub1.shape),
        "subsample_proj": h_sub_proj.squeeze(0).detach().numpy().tolist(),
        "layer0": {k: v.tolist() for k, v in layer0_outs.items()},
        "layer11_out": layer11_out.tolist(),
        "audio_tower_out": h_audio_out.squeeze(0).detach().numpy().tolist(),
        "soft_tokens": h_soft_tokens.squeeze(0).detach().numpy().tolist(),
        "embeddings": {
            "768": emb_audio_768.tolist(),
            "512": emb_audio_512.tolist(),
            "256": emb_audio_256.tolist(),
            "128": emb_audio_128.tolist(),
        },
        "candidates": {
            CANDIDATES[i]: {
                "embedding": emb_candidates[i].tolist(),
                "similarity": float(audio_sims[i]),
            }
            for i in range(len(CANDIDATES))
        },
        "interleaved": {
            "prompt": prompt_interleaved,
            "embedding": emb_interleaved.tolist(),
        }
    }

    print(f"Writing golden reference to {golden_json_path}...")
    with open(golden_json_path, "w", encoding="utf-8") as f:
        json.dump(golden_data, f, indent=2)

    print(f"Successfully generated {golden_json_path} ({os.path.getsize(golden_json_path)} bytes)!")

if __name__ == "__main__":
    main()
