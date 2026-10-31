#!/usr/bin/env python3
"""
Official EmbeddingGemma-2 Multimodal Preprocessor & Feature Bundle Exporter.

Mirrors the official Google tutorial:
https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers

Uses HuggingFace `AutoProcessor` (`EmbeddingGemma2Processor`) to decode and preprocess
images, audio, video, and interleaved multimodal inputs into lightweight `.mmfeat` binary
bundles consumed directly by the LuisaCompute C++ GPU mega-kernel engine.

Usage:
  # Preprocess all official tutorial assets (downloads them into data/ if missing):
  python examples/embedding_mm/tools/preprocess.py --model-dir path/to/embeddinggemma-2

  # Preprocess and also export PyTorch golden reference for unit tests:
  python examples/embedding_mm/tools/preprocess.py --model-dir path/to/embeddinggemma-2 --export-golden

  # Preprocess a custom image, audio, or video file:
  python examples/embedding_mm/tools/preprocess.py --model-dir path/to/embeddinggemma-2 --image my_photo.jpg
  python examples/embedding_mm/tools/preprocess.py --model-dir path/to/embeddinggemma-2 --audio clip.wav
  python examples/embedding_mm/tools/preprocess.py --model-dir path/to/embeddinggemma-2 --video movie.mp4 --max-frames 16 --add-timestamps
"""

import argparse
import json
import math
import os
import struct
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import wavfile
from scipy.signal import resample_poly

MMFT_MAGIC = 0x54464D4D  # "MMFT"
MMFT_VERSION = 1

MODALITY_IMAGE = 1
MODALITY_AUDIO = 2
MODALITY_VIDEO = 3
MODALITY_MULTIMODAL = 4

OFFICIAL_ASSETS = {
    "GoldenGate.png": "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/apps/sample-data/GoldenGate.png",
    "kitchen_painting.jpg": "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/apps/sample-data/kitchen_painting.jpg",
    "roses-are.wav": "https://ai.google.dev/gemma/docs/audio/roses-are.wav",
    "cat_dozing.mp4": "https://github.com/bebechien/gemma/raw/refs/heads/main/videos/cat_dozing.mp4",
}


def configure_video_backend():
    """Ensure transformers video loader falls back cleanly to opencv if torchcodec/pyav are absent."""
    import transformers.video_processing_utils as vpu
    import transformers.video_utils as vu

    orig_load_video = vu.load_video

    def _load_video_with_opencv_fallback(
        video, num_frames=None, fps=None, backend="torchcodec", sample_indices_fn=None, **kwargs
    ):
        if backend in ("torchcodec", "torchvision"):
            try:
                import torchcodec  # noqa: F401
            except ImportError:
                try:
                    import av  # noqa: F401
                except ImportError:
                    backend = "opencv"
        return orig_load_video(
            video,
            num_frames=num_frames,
            fps=fps,
            backend=backend,
            sample_indices_fn=sample_indices_fn,
            **kwargs,
        )

    vu.load_video = _load_video_with_opencv_fallback
    vpu.load_video = _load_video_with_opencv_fallback


def pack_header(
    modality: int,
    num_frames: int,
    max_patches: int,
    num_patches_w: int,
    num_patches_h: int,
    num_soft_tokens: int,
    feat_dim: int,
    r0: int = 0,
) -> bytes:
    return struct.pack(
        "<16I",
        MMFT_MAGIC,
        MMFT_VERSION,
        modality,
        num_frames,
        max_patches,
        num_patches_w,
        num_patches_h,
        num_soft_tokens,
        feat_dim,
        r0,
        0, 0, 0, 0, 0, 0,
    )


def load_audio_16k(wav_path: Path) -> np.ndarray:
    sr, data = wavfile.read(str(wav_path))
    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        audio = (data.astype(np.float32) - 128.0) / 128.0
    else:
        audio = data.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        g = math.gcd(sr, 16000)
        audio = resample_poly(audio, 16000 // g, sr // g).astype(np.float32)
    return audio


def extract_image_record(px_row: np.ndarray, pos_row: np.ndarray) -> bytes:
    px = np.ascontiguousarray(px_row, dtype=np.float32)
    pos = np.ascontiguousarray(pos_row, dtype=np.int32)
    max_patches, feat_dim = px.shape
    valid = pos[:, 0] >= 0
    num_w = int(pos[valid, 0].max()) + 1
    num_h = int(pos[valid, 1].max()) + 1
    num_soft = int(valid.sum()) // 9
    hdr = pack_header(MODALITY_IMAGE, 1, max_patches, num_w, num_h, num_soft, feat_dim)
    return hdr + px.tobytes() + pos.tobytes()


def extract_audio_record(feat_row: np.ndarray, mask_row: np.ndarray) -> bytes:
    valid_len = int(mask_row.sum())
    feat = np.ascontiguousarray(feat_row[:valid_len], dtype=np.float32)
    num_frames, feat_dim = feat.shape
    h0 = (num_frames + 1) // 2
    num_soft = (h0 + 1) // 2
    hdr = pack_header(MODALITY_AUDIO, num_frames, 0, 0, 0, num_soft, feat_dim)
    return hdr + feat.tobytes()


def extract_video_record(
    px_vid: np.ndarray,
    pos_vid: np.ndarray,
    input_ids: np.ndarray | None = None,
) -> bytes:
    px = np.ascontiguousarray(px_vid, dtype=np.float32)
    pos = np.ascontiguousarray(pos_vid, dtype=np.int32)
    num_frames, max_patches, feat_dim = px.shape
    valid = pos[0, :, 0] >= 0
    num_w = int(pos[0, valid, 0].max()) + 1
    num_h = int(pos[0, valid, 1].max()) + 1
    num_soft = int(valid.sum()) // 9
    ids_bytes = b""
    seq_len = 0
    if input_ids is not None:
        ids_arr = np.ascontiguousarray(input_ids.reshape(-1), dtype=np.uint32)
        seq_len = int(ids_arr.shape[0])
        ids_bytes = ids_arr.tobytes()
    hdr = pack_header(
        MODALITY_VIDEO, num_frames, max_patches, num_w, num_h, num_soft, feat_dim, r0=seq_len
    )
    return hdr + px.tobytes() + pos.tobytes() + ids_bytes


def save_image_mmfeat(
    processor,
    image_path: Path,
    out_path: Path,
    max_soft_tokens: int = 280,
):
    img = Image.open(image_path).convert("RGB")
    batch = processor(
        text=["<|image|>"],
        images=[img],
        images_kwargs={"max_soft_tokens": max_soft_tokens},
        return_tensors="np",
    )
    rec = extract_image_record(batch["pixel_values"][0], batch["image_position_ids"][0])
    out_path.write_bytes(rec)
    valid = batch["image_position_ids"][0, :, 0] >= 0
    num_w = int(batch["image_position_ids"][0, valid, 0].max()) + 1
    num_h = int(batch["image_position_ids"][0, valid, 1].max()) + 1
    num_soft = int(valid.sum()) // 9
    print(
        f"  [Image] {out_path.name}: grid={num_w}x{num_h}, "
        f"max_patches={batch['pixel_values'].shape[1]}, soft_tokens={num_soft}"
    )


def save_audio_mmfeat(processor, audio_path: Path, out_path: Path):
    audio = load_audio_16k(audio_path)
    batch = processor(
        text=["<|audio|>"],
        audio=[audio],
        return_tensors="np",
    )
    rec = extract_audio_record(batch["input_features"][0], batch["input_features_mask"][0])
    out_path.write_bytes(rec)
    valid_len = int(batch["input_features_mask"][0].sum())
    num_soft = ((valid_len + 1) // 2 + 1) // 2
    print(f"  [Audio] {out_path.name}: frames={valid_len}, mel_bins=128, soft_tokens={num_soft}")


def save_video_mmfeat(
    processor,
    video_path: Path,
    out_path: Path,
    fps: float | None = None,
    max_frames: int | None = None,
    overflow_strategy: str | None = None,
    add_timestamps: bool | None = None,
    max_soft_tokens: int | None = None,
):
    videos_kwargs = {}
    if max_frames is not None:
        videos_kwargs["max_frames"] = max_frames
    if overflow_strategy is not None:
        videos_kwargs["overflow_strategy"] = overflow_strategy
    if add_timestamps is not None:
        videos_kwargs["add_timestamps"] = add_timestamps
        if add_timestamps and fps is None:
            videos_kwargs["fps"] = None
    elif fps is not None:
        videos_kwargs["fps"] = fps
    if max_soft_tokens is not None:
        videos_kwargs["max_soft_tokens"] = max_soft_tokens
    call_kwargs = {
        "text": ["<|video|>"],
        "videos": [str(video_path)],
        "return_tensors": "np",
    }
    if videos_kwargs:
        call_kwargs["videos_kwargs"] = videos_kwargs
    batch = processor(**call_kwargs)
    px_vid = batch["pixel_values_videos"]
    pos_vid = batch["video_position_ids"]
    rec = extract_video_record(px_vid, pos_vid, batch["input_ids"][0])
    out_path.write_bytes(rec)
    valid = pos_vid[0, :, 0] >= 0
    num_w = int(pos_vid[0, valid, 0].max()) + 1
    num_h = int(pos_vid[0, valid, 1].max()) + 1
    num_soft = int(valid.sum()) // 9
    print(
        f"  [Video] {out_path.name}: frames={px_vid.shape[0]}, grid={num_w}x{num_h}, "
        f"soft_tokens/frame={num_soft}, seq_len={batch['input_ids'].shape[1]}"
    )


def save_multimodal_mmfeat(
    processor,
    prompt_text: str,
    out_path: Path,
    images: list[Path] | None = None,
    audios: list[Path] | None = None,
    videos: list[Path] | None = None,
):
    pil_images = [Image.open(p).convert("RGB") for p in images] if images else None
    audio_arrays = [load_audio_16k(p) for p in audios] if audios else None
    video_paths = [str(p) for p in videos] if videos else None

    content = []
    if prompt_text:
        content.append({"type": "text", "text": prompt_text})
    if pil_images:
        for _ in pil_images:
            content.append({"type": "image"})
    if audio_arrays:
        for _ in audio_arrays:
            content.append({"type": "audio"})
    if video_paths:
        for _ in video_paths:
            content.append({"type": "video"})
    text = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False)

    kwargs = {"text": [text], "return_tensors": "np"}
    if pil_images:
        kwargs["images"] = [pil_images]
    if audio_arrays:
        kwargs["audio"] = [audio_arrays]
    if video_paths:
        kwargs["videos"] = [video_paths]

    batch = processor(**kwargs)
    input_ids = np.ascontiguousarray(batch["input_ids"][0], dtype=np.uint32)
    num_images = len(pil_images) if pil_images else 0
    num_audios = len(audio_arrays) if audio_arrays else 0
    num_videos = len(video_paths) if video_paths else 0

    hdr = pack_header(
        MODALITY_MULTIMODAL,
        int(input_ids.shape[0]),
        num_images,
        num_audios,
        num_videos,
        0,
        0,
    )
    chunks = [hdr, input_ids.tobytes()]

    if num_images > 0:
        px_all = batch["pixel_values"]
        pos_all = batch["image_position_ids"]
        for i in range(num_images):
            chunks.append(extract_image_record(px_all[i], pos_all[i]))

    if num_audios > 0:
        feat_all = batch["input_features"]
        mask_all = batch["input_features_mask"]
        for a in range(num_audios):
            chunks.append(extract_audio_record(feat_all[a], mask_all[a]))

    if num_videos > 0:
        px_vid = batch["pixel_values_videos"]
        pos_vid = batch["video_position_ids"]
        n_frames_per_vid = batch.get("num_frames_per_video", [px_vid.shape[0]])
        offset = 0
        for v in range(num_videos):
            nf = int(n_frames_per_vid[v])
            chunks.append(extract_video_record(px_vid[offset : offset + nf], pos_vid[offset : offset + nf]))
            offset += nf

    out_path.write_bytes(b"".join(chunks))
    print(
        f"  [Multimodal] {out_path.name}: seq_len={input_ids.shape[0]}, "
        f"images={num_images}, audios={num_audios}, videos={num_videos}"
    )


def ensure_official_assets(data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, url in OFFICIAL_ASSETS.items():
        dest = data_dir / name
        if not dest.exists():
            print(f"Downloading official tutorial asset '{name}'...")
            urllib.request.urlretrieve(url, str(dest))


def export_golden_reference(model_dir: str, data_dir: Path):
    import torch
    from sentence_transformers import SentenceTransformer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Golden Export] Loading SentenceTransformer model on '{dev}' (float32)...")
    st_model = SentenceTransformer(
        model_dir, device=dev, model_kwargs={"dtype": torch.float32}
    )
    st_model.eval()
    hf_model = st_model[0].auto_model
    processor = st_model[0].processor

    img1_path = data_dir / "GoldenGate.png"
    img2_path = data_dir / "kitchen_painting.jpg"
    audio_path = data_dir / "roses-are.wav"
    video_path = data_dir / "cat_dozing.mp4"

    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")
    audio_16k = load_audio_16k(audio_path)

    golden = {}

    def encode_mrl_dict(st_inst, inp, processing_kwargs=None, **kwargs):
        prev_pk = getattr(st_inst[0], "processing_kwargs", None)
        if processing_kwargs is not None:
            st_inst[0].processing_kwargs = processing_kwargs
        try:
            vec768 = st_inst.encode(inp, normalize_embeddings=True, **kwargs)
        finally:
            if processing_kwargs is not None:
                st_inst[0].processing_kwargs = prev_pk
        if isinstance(vec768, torch.Tensor):
            vec768 = vec768.detach().cpu().numpy()
        vec768 = np.asarray(vec768, dtype=np.float32).reshape(-1)
        out = {}
        for d in (768, 512, 256, 128):
            v = vec768[:d].copy()
            v /= np.linalg.norm(v) + 1e-12
            out[str(d)] = v.tolist()
        return out, vec768

    # 1. Text & MRL section (from official Quick Start)
    print("  [1/6] Computing Text & MRL golden embeddings...")
    query_text = "What causes the northern lights?"
    doc_text = "The northern lights are caused by charged particles from the sun."
    q_mrl, q_vec = encode_mrl_dict(st_model, query_text, prompt_name="SearchQuery")
    d_mrl, d_vec = encode_mrl_dict(st_model, doc_text, prompt_name="Document")
    golden["text"] = {
        "query": query_text,
        "document": doc_text,
        "query_embeddings": q_mrl,
        "document_embeddings": d_mrl,
        "similarity_768": float(np.dot(q_vec, d_vec)),
    }

    # 2. Image section (default 280 & custom 1120) + Vision intermediate layer activations
    print("  [2/6] Computing Vision intermediate activations & Image retrieval golden...")
    with torch.no_grad():
        v_batch = processor(
            text=["<|image|>"],
            images=[img1],
            return_tensors="pt",
        )
        px_t = v_batch["pixel_values"].to(dev, dtype=torch.float32)
        pos_t = v_batch["image_position_ids"].to(dev)
        v_tower = hf_model.vision_tower
        v_conn = hf_model.embed_vision

        padding_positions = (pos_t == -1).all(dim=-1)
        patch_embeds = v_tower.patch_embedder(px_t, pos_t, padding_positions)
        layer0_captured = []

        def _cap_v_l0(_m, _i, out):
            layer0_captured.append(out[0] if isinstance(out, tuple) else out)

        h_l0 = v_tower.encoder.layers[0].register_forward_hook(_cap_v_l0)
        v_out = v_tower(pixel_values=px_t, pixel_position_ids=pos_t)
        h_l0.remove()
        layer0_out = layer0_captured[0]
        pooled_out = v_out.last_hidden_state
        soft_tokens = v_conn(pooled_out)

    img1_mrl, img1_vec = encode_mrl_dict(st_model, {"image": img1})
    img2_mrl, img2_vec = encode_mrl_dict(st_model, {"image": img2})
    img_cand_texts = [
        "San Francisco",
        "a cute cat walking",
        "a friendly dog sitting in the grass",
    ]
    img_cand_vecs = st_model.encode(img_cand_texts, normalize_embeddings=True)
    img_sims_280 = [float(np.dot(img1_vec, img2_vec))] + [
        float(np.dot(img1_vec, c)) for c in img_cand_vecs
    ]

    img1_1120_mrl, img1_1120_vec = encode_mrl_dict(
        st_model,
        {"image": img1},
        processing_kwargs={"image": {"max_soft_tokens": 1120}},
    )
    img2_1120_mrl, img2_1120_vec = encode_mrl_dict(
        st_model,
        {"image": img2},
        processing_kwargs={"image": {"max_soft_tokens": 1120}},
    )
    img_sims_1120 = [float(np.dot(img1_1120_vec, img2_1120_vec))] + [
        float(np.dot(img1_1120_vec, c)) for c in img_cand_vecs
    ]

    golden["vision"] = {
        "layer_activations": {
            "patch_embed_slice": patch_embeds[0, :64, :16].detach().cpu().reshape(-1).tolist(),
            "vit_layer0_slice": layer0_out[0, :64, :16].detach().cpu().reshape(-1).tolist(),
            "pooled_slice": pooled_out[:32, :16].detach().cpu().reshape(-1).tolist(),
            "soft_tokens_slice": soft_tokens[:32, :16].detach().cpu().reshape(-1).tolist(),
        },
        "GoldenGate_280": img1_mrl,
        "kitchen_painting_280": img2_mrl,
        "GoldenGate_1120": img1_1120_mrl,
        "kitchen_painting_1120": img2_1120_mrl,
        "candidate_texts": img_cand_texts,
        "similarities_280": img_sims_280,
        "similarities_1120": img_sims_1120,
    }

    # 3. Audio section + Audio Conformer intermediate activations
    print("  [3/6] Computing Audio Conformer intermediate activations & Audio retrieval golden...")
    with torch.no_grad():
        a_batch = processor(text=["<|audio|>"], audio=[audio_16k], return_tensors="pt")
        a_feats = a_batch["input_features"].to(dev, dtype=torch.float32)
        a_mask = a_batch["input_features_mask"].to(dev)
        a_tower = hf_model.audio_tower
        cap = {}

        def _make_cap(k):
            def _hook(_m, _i, out):
                cap[k] = out[0] if isinstance(out, tuple) else out
            return _hook

        h1 = a_tower.subsample_conv_projection.register_forward_hook(_make_cap("sub"))
        h2 = a_tower.layers[0].register_forward_hook(_make_cap("l0"))
        h3 = a_tower.layers[11].register_forward_hook(_make_cap("l11"))
        h4 = a_tower.output_proj.register_forward_hook(_make_cap("tower"))
        aud_out = hf_model.get_audio_features(a_feats, a_mask, return_dict=True)
        for h in (h1, h2, h3, h4):
            h.remove()
        sub_out = cap["sub"]
        l0_out = cap["l0"]
        l11_out = cap["l11"]
        tower_out = cap["tower"]
        aud_soft = aud_out.pooler_output

    aud_mrl, aud_vec = encode_mrl_dict(st_model, {"audio": audio_16k})
    aud_cands = [
        "a person speaking English",
        "random static white noise",
        "Roses are red, violets are blue.",
    ]
    aud_cand_vecs = st_model.encode(aud_cands, normalize_embeddings=True)
    aud_sims = [float(np.dot(aud_vec, c)) for c in aud_cand_vecs]

    golden["audio"] = {
        "layer_activations": {
            "subsample_proj": sub_out[0].detach().cpu().tolist(),
            "layer0_out": l0_out[0].detach().cpu().tolist(),
            "layer11_out": l11_out[0].detach().cpu().tolist(),
            "audio_tower_out": tower_out[0].detach().cpu().tolist(),
            "soft_tokens": aud_soft[0].detach().cpu().tolist(),
        },
        "roses_are": aud_mrl,
        "candidates": aud_cands,
        "similarities": aud_sims,
    }

    # 4. Video section (default 1fps & custom 16 frames with timestamps)
    print("  [4/6] Computing Video retrieval golden (default & 16-frame timestamped)...")
    vid_mrl, vid_vec = encode_mrl_dict(st_model, {"video": str(video_path)})
    vid_cands = [
        "a cat dozing",
        "a cat playing with a small lizard",
        "a sports car racing on a highway",
    ]
    vid_cand_vecs = st_model.encode(vid_cands, normalize_embeddings=True)
    vid_sims_default = [float(np.dot(vid_vec, c)) for c in vid_cand_vecs]

    vid16_mrl, vid16_vec = encode_mrl_dict(
        st_model,
        {"video": str(video_path)},
        processing_kwargs={
            "video": {
                "max_frames": 16,
                "fps": None,
                "overflow_strategy": "uniform",
                "add_timestamps": True,
            }
        },
    )
    vid_sims_16f = [float(np.dot(vid16_vec, c)) for c in vid_cand_vecs]

    golden["video"] = {
        "cat_dozing_default": vid_mrl,
        "cat_dozing_16frames": vid16_mrl,
        "candidates": vid_cands,
        "similarities_default": vid_sims_default,
        "similarities_16frames": vid_sims_16f,
    }

    # 5. Interleaved Multimodal section (3 official queries)
    print("  [5/6] Computing Interleaved Multimodal golden embeddings...")
    q_mm = {
        "text": "Landmark and interior photos with spoken narration",
        "image": [img1, img2],
        "audio": audio_16k,
    }
    q_inter = {
        "text": "Compare the scene in <|video|> with <|image|> and <|image|>.",
        "image": [img1, img2],
        "video": str(video_path),
    }
    q_seq = {
        "text": "<|image|><|audio|><|image|>",
        "image": [img1, img2],
        "audio": audio_16k,
    }
    mm1_mrl, mm1_vec = encode_mrl_dict(st_model, q_mm)
    mm2_mrl, mm2_vec = encode_mrl_dict(st_model, q_inter)
    mm3_mrl, mm3_vec = encode_mrl_dict(st_model, q_seq)

    inter_cands = [
        "San Francisco Golden Gate Bridge and a kitchen painting with a poem about roses",
        "A sleepy cat compared with the Golden Gate Bridge and a kitchen painting",
        "A formula one race car speeding on a wet track",
    ]
    inter_cand_vecs = st_model.encode(inter_cands, normalize_embeddings=True)
    inter_sim_matrix = (
        np.matmul(np.stack([mm1_vec, mm2_vec, mm3_vec]), inter_cand_vecs.T).tolist()
    )

    golden["interleaved"] = {
        "query_multimodal": mm1_mrl,
        "query_interleaved": mm2_mrl,
        "query_media_sequence": mm3_mrl,
        "candidates": inter_cands,
        "similarity_matrix": inter_sim_matrix,
    }

    golden_path = data_dir / "golden_official.json"
    print(f"  [6/6] Saving '{golden_path}'...")
    with open(golden_path, "w", encoding="utf-8") as f:
        json.dump(golden, f, indent=2)
    print("[Golden Export] Complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess media files into .mmfeat tensor bundles using Google's official EmbeddingGemma-2 AutoProcessor."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("EMBEDDINGGEMMA_MODEL_DIR", "models/embeddinggemma-2"),
        help="Path or HuggingFace ID of google/embeddinggemma-2",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="examples/embedding_mm/data",
        help="Directory for official tutorial assets and .mmfeat outputs",
    )
    parser.add_argument("--image", type=str, nargs="*", help="Custom image file(s) to preprocess")
    parser.add_argument("--audio", type=str, nargs="*", help="Custom audio WAV file(s) to preprocess")
    parser.add_argument("--video", type=str, nargs="*", help="Custom video MP4 file(s) to preprocess")
    parser.add_argument("--text", type=str, default=None, help="Optional text prompt for interleaved multimodal bundle")
    parser.add_argument("--output", type=str, default=None, help="Output .mmfeat path for custom input")
    parser.add_argument("--max-soft-tokens", type=int, default=None, help="Vision max_soft_tokens budget (e.g. 280, 1120, 70)")
    parser.add_argument("--fps", type=float, default=1.0, help="Video sampling FPS (pass 0 for uniform max_frames sampling)")
    parser.add_argument("--max-frames", type=int, default=32, help="Video max frames budget")
    parser.add_argument("--overflow-strategy", type=str, default="truncate", choices=["truncate", "uniform"])
    parser.add_argument("--add-timestamps", action="store_true", help="Prepend mm:ss timestamps to video frames")
    parser.add_argument("--export-golden", action="store_true", help="Also export golden_official.json from PyTorch")
    args = parser.parse_args()

    configure_video_backend()

    from transformers import AutoProcessor

    print(f"Loading EmbeddingGemma2Processor from '{args.model_dir}'...")
    processor = AutoProcessor.from_pretrained(args.model_dir)

    # Custom user file preprocessing mode
    if args.image or args.audio or args.video:
        total_modalities = bool(args.image) + bool(args.audio) + bool(args.video)
        if args.text or total_modalities > 1 or (args.image and len(args.image) > 1):
            out = Path(args.output) if args.output else Path("custom_multimodal.mmfeat")
            save_multimodal_mmfeat(
                processor,
                args.text or "",
                out,
                images=[Path(p) for p in args.image] if args.image else None,
                audios=[Path(p) for p in args.audio] if args.audio else None,
                videos=[Path(p) for p in args.video] if args.video else None,
            )
        elif args.image:
            in_p = Path(args.image[0])
            out = Path(args.output) if args.output else Path(str(in_p) + ".mmfeat")
            save_image_mmfeat(processor, in_p, out, max_soft_tokens=args.max_soft_tokens or 280)
        elif args.audio:
            in_p = Path(args.audio[0])
            out = Path(args.output) if args.output else Path(str(in_p) + ".mmfeat")
            save_audio_mmfeat(processor, in_p, out)
        elif args.video:
            in_p = Path(args.video[0])
            out = Path(args.output) if args.output else Path(str(in_p) + ".mmfeat")
            save_video_mmfeat(
                processor,
                in_p,
                out,
                fps=None if args.fps <= 0 else args.fps,
                max_frames=args.max_frames,
                overflow_strategy=args.overflow_strategy,
                add_timestamps=args.add_timestamps,
                max_soft_tokens=args.max_soft_tokens or 70,
            )
        return

    # Default mode: preprocess all official tutorial assets
    data_dir = Path(args.data_dir)
    ensure_official_assets(data_dir)

    print(f"\nPreprocessing official tutorial assets in '{data_dir}'...")
    save_image_mmfeat(processor, data_dir / "GoldenGate.png", data_dir / "GoldenGate.png.mmfeat", max_soft_tokens=280)
    save_image_mmfeat(processor, data_dir / "GoldenGate.png", data_dir / "GoldenGate_1120.png.mmfeat", max_soft_tokens=1120)
    save_image_mmfeat(processor, data_dir / "kitchen_painting.jpg", data_dir / "kitchen_painting.jpg.mmfeat", max_soft_tokens=280)
    save_image_mmfeat(processor, data_dir / "kitchen_painting.jpg", data_dir / "kitchen_painting_1120.jpg.mmfeat", max_soft_tokens=1120)

    save_audio_mmfeat(processor, data_dir / "roses-are.wav", data_dir / "roses-are.wav.mmfeat")

    save_video_mmfeat(
        processor,
        data_dir / "cat_dozing.mp4",
        data_dir / "cat_dozing.mp4.mmfeat",
    )
    save_video_mmfeat(
        processor,
        data_dir / "cat_dozing.mp4",
        data_dir / "cat_dozing_16frames.mp4.mmfeat",
        fps=None,
        max_frames=16,
        overflow_strategy="uniform",
        add_timestamps=True,
    )

    save_multimodal_mmfeat(
        processor,
        "Landmark and interior photos with spoken narration",
        data_dir / "interleaved_multimodal.mmfeat",
        images=[data_dir / "GoldenGate.png", data_dir / "kitchen_painting.jpg"],
        audios=[data_dir / "roses-are.wav"],
    )
    save_multimodal_mmfeat(
        processor,
        "Compare the scene in <|video|> with <|image|> and <|image|>.",
        data_dir / "interleaved_prompt.mmfeat",
        images=[data_dir / "GoldenGate.png", data_dir / "kitchen_painting.jpg"],
        videos=[data_dir / "cat_dozing.mp4"],
    )
    save_multimodal_mmfeat(
        processor,
        "<|image|><|audio|><|image|>",
        data_dir / "interleaved_media_seq.mmfeat",
        images=[data_dir / "GoldenGate.png", data_dir / "kitchen_painting.jpg"],
        audios=[data_dir / "roses-are.wav"],
    )

    if args.export_golden:
        export_golden_reference(args.model_dir, data_dir)

    print("\nAll .mmfeat bundles generated successfully!")


if __name__ == "__main__":
    main()
