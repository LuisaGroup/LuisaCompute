#!/usr/bin/env python3
"""
dump_golden.py -- Extract golden intermediate representations and embeddings from EmbeddingGemma-2.

Runs google/embeddinggemma-2 using SentenceTransformers or HuggingFace Transformers
on representative reference sentences and exports golden outputs for C++ parity tests:
- golden_tokens.json: Token IDs for each sentence
- golden_dense_768.bin: Native 768-dim normalized embedding vectors (fp32)
- golden_dense_512.bin: MRL 512-dim normalized embedding vectors (fp32)
- golden_dense_256.bin: MRL 256-dim normalized embedding vectors (fp32)
- golden_dense_128.bin: MRL 128-dim normalized embedding vectors (fp32)

Usage:
  python dump_golden.py --model google/embeddinggemma-2 --output_dir ./golden_data
"""

import argparse
import json
import os
import sys
import numpy as np

SAMPLE_TEXTS = [
    "Hello world, welcome to multimodal embeddings with Gemma!",
    "The quick brown fox jumps over the lazy dog.",
    "LuisaCompute is a high-performance, cross-platform rendering and compute framework.",
    "Artificial intelligence models map diverse modalities into a unified latent space.",
    "Matryoshka Representation Learning allows dynamic truncation of embedding dimensions.",
]

def main():
    parser = argparse.ArgumentParser(description="Generate golden test data for EmbeddingGemma-2")
    parser.add_argument("--model", default="google/embeddinggemma-2", help="HF model name or path")
    parser.add_argument("--output_dir", default="./golden_data", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    has_st = False
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        print(f"Loading model '{args.model}' with SentenceTransformer (fp32)...")
        model = SentenceTransformer(args.model, model_kwargs={'torch_dtype': torch.float32})
        has_st = True
    except Exception as e:
        print(f"SentenceTransformer not available ({e}). Using PyTorch reference forward pass...")

    if has_st:
        for dim in [768, 512, 256, 128]:
            print(f"Generating golden embeddings for dim={dim}...")
            embeddings = model.encode(SAMPLE_TEXTS, truncate_dim=dim, normalize_embeddings=True)
            emb_arr = np.array(embeddings, dtype=np.float32)
            out_file = os.path.join(args.output_dir, f"golden_dense_{dim}.bin")
            emb_arr.tofile(out_file)
            print(f"Saved: {out_file} (shape {emb_arr.shape})")
    else:
        # PyTorch reference implementation
        import torch
        import torch.nn.functional as F
        from safetensors import safe_open
        from transformers import AutoTokenizer

        st_path = os.path.join(args.model, "model.safetensors") if os.path.isdir(args.model) else args.model
        print(f"Loading safetensors from '{st_path}'...")
        weights = {}
        with safe_open(st_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                weights[k] = f.get_tensor(k).float()

        def get_w(name):
            if name in weights:
                return weights[name]
            for pfx in ["language_model.model.", "language_model.", "text_model.model.", "text_model.", "model.language_model.", "model.", ""]:
                cand = pfx + name
                if cand in weights:
                    return weights[cand]
            for k in weights:
                if name in k:
                    return weights[k]
            return None

        from tokenizers import Tokenizer
        tok_file = os.path.join(args.model, "tokenizer.json") if os.path.isdir(args.model) else "tokenizer.json"
        print(f"Loading tokenizer from '{tok_file}'...")
        hf_tok = Tokenizer.from_file(tok_file)
        
        encoded_list = []
        for text in SAMPLE_TEXTS:
            enc = hf_tok.encode(text)
            ids = enc.ids
            if not ids or ids[0] != 2:
                ids = [2] + ids
            encoded_list.append(ids)

        max_s = max(len(ids) for ids in encoded_list)
        B = len(SAMPLE_TEXTS)
        S = max_s
        input_ids = torch.zeros((B, S), dtype=torch.long)
        attention_mask = torch.zeros((B, S), dtype=torch.float32)
        for b, ids in enumerate(encoded_list):
            for s, tid in enumerate(ids):
                input_ids[b, s] = tid
                attention_mask[b, s] = 1.0
        H = 512
        I = 2048

        def rms_norm(x, w, eps=1e-6):
            return (x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)) * (1.0 + w)

        def rotate_half(x):
            half = x.shape[-1] // 2
            x1, x2 = x[..., :half], x[..., half:]
            return torch.cat((-x2, x1), dim=-1)

        def apply_rope(x, num_heads, head_dim, rope_theta):
            # x: [B, S, num_heads, head_dim]
            half = head_dim // 2
            freqs = 1.0 / (rope_theta ** (torch.arange(0, half, dtype=torch.float32) / half))
            pos = torch.arange(S, dtype=torch.float32)
            angles = pos[:, None] * freqs[None, :]  # [S, half]
            cos = torch.cat([angles.cos(), angles.cos()], dim=-1)[None, :, None, :]  # [1, S, 1, head_dim]
            sin = torch.cat([angles.sin(), angles.sin()], dim=-1)[None, :, None, :]
            return (x * cos) + (rotate_half(x) * sin)

        # 1. Embedding lookup
        embed_tokens = get_w("embed_tokens.weight")
        h0 = F.embedding(input_ids, embed_tokens) * (512.0 ** 0.5)

        # 2. 24 Layers
        for layer_idx in range(24):
            pfx = f"layers.{layer_idx}."
            is_full = (layer_idx in [5, 11, 17, 23])
            head_dim = 512 if is_full else 256
            num_q_heads = 4
            num_kv_heads = 1 if is_full else 2
            rope_theta = 1000000.0 if is_full else 10000.0
            is_sliding = not is_full

            # (a) Input RMSNorm
            h_norm = rms_norm(h0, get_w(pfx + "input_layernorm.weight"))

            # (b) Q, K, V projections
            q = F.linear(h_norm, get_w(pfx + "self_attn.q_proj.weight"))
            k = F.linear(h_norm, get_w(pfx + "self_attn.k_proj.weight"))
            v = F.linear(h_norm, get_w(pfx + "self_attn.v_proj.weight"))

            q = q.view(B, S, num_q_heads, head_dim)
            k = k.view(B, S, num_kv_heads, head_dim)
            v = v.view(B, S, num_kv_heads, head_dim)

            # (c) RoPE
            q = apply_rope(q, num_q_heads, head_dim, rope_theta)
            k = apply_rope(k, num_kv_heads, head_dim, rope_theta)

            # (d) GQA
            q = q.transpose(1, 2)  # [B, num_q_heads, S, head_dim]
            k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=2).transpose(1, 2)
            v = v.repeat_interleave(num_q_heads // num_kv_heads, dim=2).transpose(1, 2)

            scale = 1.0 / (head_dim ** 0.5)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, 4, S, S]

            # Masking
            pos_q = torch.arange(S)[:, None]
            pos_k = torch.arange(S)[None, :]
            pad_mask = (attention_mask[:, None, None, :] == 0)  # [B, 1, 1, S]
            scores = scores.masked_fill(pad_mask, -1e9)
            if is_sliding:
                dist_mask = torch.abs(pos_q - pos_k) > 512
                scores = scores.masked_fill(dist_mask[None, None, :, :], -1e9)

            attn_weights = F.softmax(scores, dim=-1)
            attn_ctx = torch.matmul(attn_weights, v)  # [B, 4, S, head_dim]
            attn_ctx = attn_ctx.transpose(1, 2).contiguous().view(B, S, num_q_heads * head_dim)

            # (e) Out proj + Post-attn RMSNorm + residual
            attn_out = F.linear(attn_ctx, get_w(pfx + "self_attn.o_proj.weight"))
            attn_norm = rms_norm(attn_out, get_w(pfx + "post_attention_layernorm.weight"))
            h0 = h0 + attn_norm

            # (f) Pre-FFN RMSNorm + GeGLU + Down proj + Post-FFN RMSNorm + residual
            ffn_norm = rms_norm(h0, get_w(pfx + "pre_feedforward_layernorm.weight"))
            gate = F.linear(ffn_norm, get_w(pfx + "mlp.gate_proj.weight"))
            up = F.linear(ffn_norm, get_w(pfx + "mlp.up_proj.weight"))
            mlp_act = F.gelu(gate, approximate="tanh") * up
            mlp_out = F.linear(mlp_act, get_w(pfx + "mlp.down_proj.weight"))
            mlp_norm = rms_norm(mlp_out, get_w(pfx + "post_feedforward_layernorm.weight"))
            h0 = h0 + mlp_norm

        # 3. Final RMSNorm
        h_final = rms_norm(h0, get_w("norm.weight"))

        # 4. Mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled = (h_final * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1.0)

        emb_proj = get_w("embedding_projection.weight")
        if emb_proj is None:
            emb_proj = get_w("embedding_proj.weight")
        if emb_proj is None:
            emb_proj = torch.eye(768, 512, dtype=torch.float32)
        proj_out = F.linear(pooled, emb_proj)

        # 6. MRL Truncation + L2Norm
        for dim in [768, 512, 256, 128]:
            truncated = proj_out[:, :dim]
            normalized = F.normalize(truncated, p=2, dim=-1)
            emb_arr = normalized.cpu().numpy().astype(np.float32)
            out_file = os.path.join(args.output_dir, f"golden_dense_{dim}.bin")
            emb_arr.tofile(out_file)
            print(f"Saved PyTorch golden: {out_file} (shape {emb_arr.shape})")

    # Save reference sentences and token IDs
    try:
        from tokenizers import Tokenizer
        tok_file = os.path.join(args.model, "tokenizer.json") if os.path.isdir(args.model) else "tokenizer.json"
        tok = Tokenizer.from_file(tok_file)
        tokens_record = []
        for text in SAMPLE_TEXTS:
            enc = tok.encode(text)
            ids = enc.ids
            if not ids or ids[0] != 2:
                ids = [2] + ids
            tokens_record.append({"text": text, "input_ids": ids})
        
        tokens_path = os.path.join(args.output_dir, "golden_tokens.json")
        with open(tokens_path, "w", encoding="utf-8") as f:
            json.dump(tokens_record, f, indent=2, ensure_ascii=False)
        print(f"Saved token IDs to {tokens_path}")
    except Exception as e:
        print(f"Warning: failed to extract tokenizer golden IDs: {e}")

    print("Golden data generation completed.")

if __name__ == "__main__":
    main()
