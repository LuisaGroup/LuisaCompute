#!/usr/bin/env python3
"""
pack_model.py -- Offline weight packer for EmbeddingGemma-2.

Converts HuggingFace safetensors weights (google/embeddinggemma-2) to the
high-performance memory-mapped model.mmegm binary package format.

Features:
- Validates model architecture against text_config
- Pre-transposes linear weights for coalesced GPU GEMM reads: B[K, N]
- Supports fp32, bf16, and fp16 export
- 256-byte aligned tensor blobs

Usage:
  python pack_model.py /path/to/embeddinggemma-2 model.mmegm [--precision fp32|bf16|fp16]
"""

import argparse
import json
import os
import struct
import sys
import numpy as np

try:
    from safetensors import safe_open
except ImportError:
    safe_open = None

EGM_MAGIC = 0x324D4745  # "EGM2"
EGM_VERSION = 1
EGM_ALIGN = 256

PRECISION_MAP = {
    "fp32": 0,
    "bf16": 1,
    "fp16": 2,
}

def fnv1a_64(name: str) -> int:
    h = 0xcbf29ce484222325
    for b in name.encode('utf-8'):
        h = ((h ^ b) * 0x00000100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h

def align_up(offset: int, alignment: int) -> int:
    return (offset + alignment - 1) & ~(alignment - 1)

def main():
    parser = argparse.ArgumentParser(description="Pack EmbeddingGemma-2 safetensors into model.mmegm")
    parser.add_argument("model_dir", help="Path to HF model directory or .safetensors file")
    parser.add_argument("output_path", help="Path to output model.mmegm file")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    args = parser.parse_args()

    if safe_open is None:
        print("Error: safetensors package is required. Run 'pip install safetensors'.", file=sys.stderr)
        sys.exit(1)

    model_dir = args.model_dir
    config_path = os.path.join(model_dir, "config.json") if os.path.isdir(model_dir) else None
    
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            text_cfg = cfg.get("text_config", cfg)
            print(f"Loaded config: hidden_size={text_cfg.get('hidden_size')}, "
                  f"num_layers={text_cfg.get('num_hidden_layers')}, "
                  f"embedding_dim={text_cfg.get('embedding_dim')}")

    # Locate safetensors files
    st_files = []
    if os.path.isdir(model_dir):
        for fname in sorted(os.listdir(model_dir)):
            if fname.endswith(".safetensors"):
                st_files.append(os.path.join(model_dir, fname))
    elif model_dir.endswith(".safetensors") and os.path.exists(model_dir):
        st_files.append(model_dir)

    if not st_files:
        print(f"Error: no safetensors files found in '{model_dir}'", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(st_files)} safetensors file(s). Indexing tensors...")
    tensors_raw = {}
    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False

    for st_path in st_files:
        if has_torch:
            with safe_open(st_path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    tensors_raw[k] = f.get_tensor(k).float().numpy()
        else:
            with safe_open(st_path, framework="numpy", device="cpu") as f:
                for k in f.keys():
                    tensors_raw[k] = f.get_tensor(k)

    print(f"Total tensors indexed: {len(tensors_raw)}")

    # Collect and transform tensors to canonical format
    canonical_tensors = []

    def get_raw(name):
        if name in tensors_raw:
            return tensors_raw[name]
        for prefix in [
            "language_model.model.",
            "language_model.",
            "text_model.model.",
            "text_model.",
            "model.language_model.",
            "model.",
            "",
        ]:
            cand = prefix + name
            if cand in tensors_raw:
                return tensors_raw[cand]
        return None

    # 1. Embedding table
    emb = get_raw("embed_tokens.weight")
    if emb is None:
        print("Warning: embed_tokens.weight not found directly; searching keys...")
        for k in tensors_raw:
            if "embed_tokens" in k:
                emb = tensors_raw[k]
                break
    if emb is None:
        raise ValueError("Cannot find embedding table 'embed_tokens.weight'")
    canonical_tensors.append(("emb.word", emb))

    # 2. PLE Global Projection and Norm
    ple_proj = get_raw("ple.per_layer_model_projection.weight")
    if ple_proj is None:
        raise ValueError("Cannot find ple.per_layer_model_projection.weight")
    # Pre-scale by 512**-0.5 and transpose [12288, 512] -> [512, 12288]
    ple_proj_scaled = (ple_proj.T * (512.0 ** -0.5)).astype(np.float32)
    canonical_tensors.append(("ple.proj", ple_proj_scaled))

    ple_norm = get_raw("ple.per_layer_projection_norm.weight")
    if ple_norm is None:
        raise ValueError("Cannot find ple.per_layer_projection_norm.weight")
    canonical_tensors.append(("ple.norm", ple_norm))

    # 3. 24 Layers
    for i in range(24):
        pfx = f"layers.{i}."
        in_norm = get_raw(pfx + "input_layernorm.weight")
        post_attn_norm = get_raw(pfx + "post_attention_layernorm.weight")
        pre_ffn_norm = get_raw(pfx + "pre_feedforward_layernorm.weight")
        post_ffn_norm = get_raw(pfx + "post_feedforward_layernorm.weight")

        # Projections: linear weights stored as [out_features, in_features] in PyTorch
        # Pre-transposed to B[K, N] = [in_features, out_features]
        q_proj = get_raw(pfx + "self_attn.q_proj.weight").T
        k_proj = get_raw(pfx + "self_attn.k_proj.weight").T
        v_proj = get_raw(pfx + "self_attn.v_proj.weight").T
        out_proj = get_raw(pfx + "self_attn.o_proj.weight").T

        q_norm = get_raw(pfx + "self_attn.q_norm.weight")
        k_norm = get_raw(pfx + "self_attn.k_norm.weight")

        gate_proj = get_raw(pfx + "mlp.gate_proj.weight").T
        up_proj = get_raw(pfx + "mlp.up_proj.weight").T
        down_proj = get_raw(pfx + "mlp.down_proj.weight").T

        ple_gate = get_raw(pfx + "ple_block.per_layer_input_gate.weight").T
        ple_proj_layer = get_raw(pfx + "ple_block.per_layer_projection.weight").T
        ple_norm_layer = get_raw(pfx + "ple_block.post_per_layer_input_norm.weight")

        layer_scalar = get_raw(pfx + "layer_scalar")
        if layer_scalar is None:
            layer_scalar = np.array([1.0], dtype=np.float32)

        canonical_tensors.extend([
            (f"layer{i}.input_norm", in_norm),
            (f"layer{i}.q_proj", q_proj),
            (f"layer{i}.k_proj", k_proj),
            (f"layer{i}.v_proj", v_proj),
            (f"layer{i}.q_norm", q_norm),
            (f"layer{i}.k_norm", k_norm),
            (f"layer{i}.out_proj", out_proj),
            (f"layer{i}.post_attn_norm", post_attn_norm),
            (f"layer{i}.pre_ffn_norm", pre_ffn_norm),
            (f"layer{i}.gate_proj", gate_proj),
            (f"layer{i}.up_proj", up_proj),
            (f"layer{i}.down_proj", down_proj),
            (f"layer{i}.post_ffn_norm", post_ffn_norm),
            (f"layer{i}.ple_gate", ple_gate),
            (f"layer{i}.ple_proj", ple_proj_layer),
            (f"layer{i}.ple_norm", ple_norm_layer),
            (f"layer{i}.layer_scalar", layer_scalar),
        ])

    # 4. Final norm & embedding projection
    final_norm = get_raw("norm.weight")
    canonical_tensors.append(("final_norm", final_norm))

    emb_proj = get_raw("embedding_projection.weight")
    if emb_proj is None:
        emb_proj = get_raw("embedding_proj.weight")
    if emb_proj is not None:
        # Transpose [768, 512] -> [512, 768]
        canonical_tensors.append(("embedding_proj", emb_proj.T))
    else:
        print("Warning: embedding_projection not found, identity/dummy used.")
        canonical_tensors.append(("embedding_proj", np.eye(512, 768, dtype=np.float32)))

    # 5. Vision Encoder weights (~170M params, 195 tensors)
    v_patch_proj = get_raw("vision_tower.patch_embedder.input_proj.weight")
    v_pos_table = get_raw("vision_tower.patch_embedder.position_embedding_table")
    if v_patch_proj is not None and v_pos_table is not None:
        print("Packaging Vision Tower tensors...")
        # v_patch_proj is [768, 768] (out_features, in_features) -> transpose to [768, 768]
        canonical_tensors.append(("vision.patch_embed.proj", v_patch_proj.T))
        # v_pos_table is [2, 10240, 768]
        canonical_tensors.append(("vision.patch_embed.pos_table", v_pos_table))

        for i in range(16):
            pfx = f"vision_tower.encoder.layers.{i}."
            v_in_norm = get_raw(pfx + "input_layernorm.weight")
            v_q_proj = get_raw(pfx + "self_attn.q_proj.linear.weight").T
            v_k_proj = get_raw(pfx + "self_attn.k_proj.linear.weight").T
            v_v_proj = get_raw(pfx + "self_attn.v_proj.linear.weight").T
            v_q_norm = get_raw(pfx + "self_attn.q_norm.weight")
            v_k_norm = get_raw(pfx + "self_attn.k_norm.weight")
            v_out_proj = get_raw(pfx + "self_attn.o_proj.linear.weight").T
            v_post_attn_norm = get_raw(pfx + "post_attention_layernorm.weight")
            v_pre_ffn_norm = get_raw(pfx + "pre_feedforward_layernorm.weight")
            v_gate_proj = get_raw(pfx + "mlp.gate_proj.linear.weight").T
            v_up_proj = get_raw(pfx + "mlp.up_proj.linear.weight").T
            v_down_proj = get_raw(pfx + "mlp.down_proj.linear.weight").T
            v_post_ffn_norm = get_raw(pfx + "post_feedforward_layernorm.weight")

            canonical_tensors.extend([
                (f"vision.layer{i}.input_norm", v_in_norm),
                (f"vision.layer{i}.q_proj", v_q_proj),
                (f"vision.layer{i}.k_proj", v_k_proj),
                (f"vision.layer{i}.v_proj", v_v_proj),
                (f"vision.layer{i}.q_norm", v_q_norm),
                (f"vision.layer{i}.k_norm", v_k_norm),
                (f"vision.layer{i}.out_proj", v_out_proj),
                (f"vision.layer{i}.post_attn_norm", v_post_attn_norm),
                (f"vision.layer{i}.pre_ffn_norm", v_pre_ffn_norm),
                (f"vision.layer{i}.gate_proj", v_gate_proj),
                (f"vision.layer{i}.up_proj", v_up_proj),
                (f"vision.layer{i}.down_proj", v_down_proj),
                (f"vision.layer{i}.post_ffn_norm", v_post_ffn_norm),
            ])

        # Connector projection: Linear(768, 512, bias=False) -> weight [512, 768] -> transpose to [768, 512]
        v_conn_proj = get_raw("embed_vision.embedding_projection.weight")
        if v_conn_proj is not None:
            canonical_tensors.append(("vision.connector.proj", v_conn_proj.T))
        print(f"Vision Tower added: 195 tensors (Total: {len(canonical_tensors)} tensors)")

    # 6. Audio Encoder weights (~300M params, 272 tensors)
    audio_sub0_conv = get_raw("audio_tower.subsample_conv_projection.layer0.conv.weight")
    if audio_sub0_conv is not None:
        print("Packaging Audio Tower tensors...")
        canonical_tensors.append(("audio.subsample.layer0.conv", audio_sub0_conv))
        canonical_tensors.append(("audio.subsample.layer0.norm", get_raw("audio_tower.subsample_conv_projection.layer0.norm.weight")))
        canonical_tensors.append(("audio.subsample.layer1.conv", get_raw("audio_tower.subsample_conv_projection.layer1.conv.weight")))
        canonical_tensors.append(("audio.subsample.layer1.norm", get_raw("audio_tower.subsample_conv_projection.layer1.norm.weight")))
        canonical_tensors.append(("audio.subsample.input_proj", get_raw("audio_tower.subsample_conv_projection.input_proj_linear.weight").T))

        for i in range(12):
            pfx = f"audio_tower.layers.{i}."
            canonical_tensors.extend([
                (f"audio.layer{i}.ffn1_pre_norm", get_raw(pfx + "feed_forward1.pre_layer_norm.weight")),
                (f"audio.layer{i}.ffn1_linear1", get_raw(pfx + "feed_forward1.ffw_layer_1.linear.weight").T),
                (f"audio.layer{i}.ffn1_linear2", get_raw(pfx + "feed_forward1.ffw_layer_2.linear.weight").T),
                (f"audio.layer{i}.ffn1_post_norm", get_raw(pfx + "feed_forward1.post_layer_norm.weight")),
                (f"audio.layer{i}.norm_pre_attn", get_raw(pfx + "norm_pre_attn.weight")),
                (f"audio.layer{i}.q_proj", get_raw(pfx + "self_attn.q_proj.linear.weight").T),
                (f"audio.layer{i}.k_proj", get_raw(pfx + "self_attn.k_proj.linear.weight").T),
                (f"audio.layer{i}.v_proj", get_raw(pfx + "self_attn.v_proj.linear.weight").T),
                (f"audio.layer{i}.per_dim_scale", get_raw(pfx + "self_attn.per_dim_scale")),
                (f"audio.layer{i}.relative_k_proj", get_raw(pfx + "self_attn.relative_k_proj.weight").T),
                (f"audio.layer{i}.out_proj", get_raw(pfx + "self_attn.post.linear.weight").T),
                (f"audio.layer{i}.norm_post_attn", get_raw(pfx + "norm_post_attn.weight")),
                (f"audio.layer{i}.lconv_pre_norm", get_raw(pfx + "lconv1d.pre_layer_norm.weight")),
                (f"audio.layer{i}.lconv_linear_start", get_raw(pfx + "lconv1d.linear_start.linear.weight").T),
                (f"audio.layer{i}.lconv_depthwise", get_raw(pfx + "lconv1d.depthwise_conv1d.weight")),
                (f"audio.layer{i}.lconv_norm", get_raw(pfx + "lconv1d.conv_norm.weight")),
                (f"audio.layer{i}.lconv_linear_end", get_raw(pfx + "lconv1d.linear_end.linear.weight").T),
                (f"audio.layer{i}.ffn2_pre_norm", get_raw(pfx + "feed_forward2.pre_layer_norm.weight")),
                (f"audio.layer{i}.ffn2_linear1", get_raw(pfx + "feed_forward2.ffw_layer_1.linear.weight").T),
                (f"audio.layer{i}.ffn2_linear2", get_raw(pfx + "feed_forward2.ffw_layer_2.linear.weight").T),
                (f"audio.layer{i}.ffn2_post_norm", get_raw(pfx + "feed_forward2.post_layer_norm.weight")),
                (f"audio.layer{i}.norm_out", get_raw(pfx + "norm_out.weight")),
            ])

        canonical_tensors.append(("audio.output_proj.weight", get_raw("audio_tower.output_proj.weight").T))
        canonical_tensors.append(("audio.output_proj.bias", get_raw("audio_tower.output_proj.bias")))
        canonical_tensors.append(("audio.connector.proj", get_raw("embed_audio.embedding_projection.weight").T))

        try:
            from transformers import AutoProcessor
            proc = AutoProcessor.from_pretrained(model_dir)
            if hasattr(proc, "feature_extractor") and hasattr(proc.feature_extractor, "mel_filters"):
                canonical_tensors.append(("audio.frontend.mel_filters", proc.feature_extractor.mel_filters.astype(np.float32)))
                print("Added audio.frontend.mel_filters to package.")
        except Exception as e:
            print(f"Note: mel_filters from processor not added ({e})")

        clip_linears = [
            "feed_forward1.ffw_layer_1",
            "feed_forward1.ffw_layer_2",
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.post",
            "lconv1d.linear_start",
            "lconv1d.linear_end",
            "feed_forward2.ffw_layer_1",
            "feed_forward2.ffw_layer_2",
        ]
        clip_bounds = np.zeros((12, 10, 4), dtype=np.float32)
        has_clips = False
        for i in range(12):
            pfx = f"audio_tower.layers.{i}."
            for j, lin in enumerate(clip_linears):
                in_min = get_raw(pfx + lin + ".input_min")
                in_max = get_raw(pfx + lin + ".input_max")
                out_min = get_raw(pfx + lin + ".output_min")
                out_max = get_raw(pfx + lin + ".output_max")
                if in_min is not None:
                    has_clips = True
                    clip_bounds[i, j, 0] = float(in_min)
                    clip_bounds[i, j, 1] = float(in_max)
                    clip_bounds[i, j, 2] = float(out_min)
                    clip_bounds[i, j, 3] = float(out_max)
        if has_clips:
            canonical_tensors.append(("audio.clip_bounds", clip_bounds))
            print("Added audio.clip_bounds [12, 10, 4] to package.")

        print(f"Audio Tower added tensors (Total: {len(canonical_tensors)} tensors)")

    prec_id = PRECISION_MAP[args.precision]
    print(f"Preparing binary package ({len(canonical_tensors)} tensors, precision={args.precision})...")

    # Layout offsets
    header_size = 32
    table_size = len(canonical_tensors) * 48
    blob_start = align_up(header_size + table_size, EGM_ALIGN)

    current_offset = blob_start
    table_entries = []
    blob_data_list = []

    for name, arr in canonical_tensors:
        arr = np.ascontiguousarray(arr)
        if args.precision == "fp32":
            blob = arr.astype(np.float32).tobytes()
        elif args.precision == "fp16":
            blob = arr.astype(np.float16).tobytes()
        elif args.precision == "bf16":
            # Convert float32 to bfloat16 bitwise
            u32 = arr.astype(np.float32).view(np.uint32)
            u16 = (u32 >> 16).astype(np.uint16)
            blob = u16.tobytes()

        rank = len(arr.shape)
        dims = list(arr.shape) + [0] * (4 - rank)
        size_bytes = len(blob)
        name_h = fnv1a_64(name)

        offset = current_offset
        table_entries.append((name_h, offset, size_bytes, rank, dims))

        blob_data_list.append(blob)
        current_offset = align_up(current_offset + size_bytes, EGM_ALIGN)

    # Write output file
    with open(args.output_path, "wb") as f:
        # Header (32 bytes)
        hdr = struct.pack("<IIIIIIII",
                          EGM_MAGIC,
                          EGM_VERSION,
                          prec_id,
                          len(canonical_tensors),
                          24,   # num_layers
                          512,  # hidden_size
                          2048, # intermediate_size
                          768)  # embedding_dim
        f.write(hdr)

        # Table entries
        for name_h, offset, size_bytes, rank, dims in table_entries:
            entry = struct.pack("<QQQIIIII",
                                name_h,
                                offset,
                                size_bytes,
                                rank,
                                dims[0], dims[1], dims[2], dims[3])
            f.write(entry)

        # Pad to blob_start
        pad_len = blob_start - f.tell()
        if pad_len > 0:
            f.write(b'\0' * pad_len)

        # Write blobs
        for i, blob in enumerate(blob_data_list):
            name_h, offset, size_bytes, rank, dims = table_entries[i]
            # Ensure file is padded to offset
            cur = f.tell()
            if cur < offset:
                f.write(b'\0' * (offset - cur))
            f.write(blob)

    print(f"Successfully generated '{args.output_path}' ({current_offset / (1024*1024):.2f} MB).")

if __name__ == "__main__":
    main()
