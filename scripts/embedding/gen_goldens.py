# M0 golden generator for the bge-m3 embedding engine (embedding_model_plan.md §6 M0).
# Dev-only script: produces the frozen corpus + golden token ids / dense / sparse /
# colbert / per-layer activations under src/tests/data/embedding/.
#
# Usage:  python scripts/embedding/gen_goldens.py
# Deps (dev env only): onnxruntime, onnx, tokenizers, torch (cpu), transformers, numpy.
import base64
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

MODEL_DIR = Path(r"D:/compute/bge-m3/bge-m3")
OUT_DIR = Path(r"D:/compute/src/tests/data/embedding")
MAX_SEQ_LEN = 512  # engine default knob; goldens are generated with this truncation
BATCH = 16
# subset of sentences that get token_embeddings / sparse / colbert / per-layer goldens
SUBSET_TE = list(range(0, 10))          # token_embeddings + sparse + colbert
SUBSET_LAYERS = [0, 3]                  # per-layer activations (D4 bisect harness)

# ---------------------------------------------------------------- corpus (frozen)
def _corpus() -> list[str]:
    en = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models transform natural language processing.",
        "How do I reverse a linked list in constant space?",
        "The capital of France is Paris, which sits on the Seine river.",
        "Quantum entanglement allows particles to share states instantaneously.",
        "She sells sea shells by the seashore on sunny Sundays.",
        "The recipe calls for two cups of flour and a pinch of salt.",
        "Climate change affects polar regions faster than temperate zones.",
        "In 1969, humans first walked on the surface of the Moon.",
        "The library opens at nine and closes at six in the evening.",
        "Water boils at one hundred degrees Celsius at sea level.",
        "My neighbor's dog barks at every delivery truck that passes by.",
        "The committee approved the budget after three hours of debate.",
        "Photosynthesis converts sunlight into chemical energy in plants.",
        "He finished the marathon in under four hours despite the heat.",
        "The novel explores themes of identity, exile, and belonging.",
        "Interest rates influence housing markets and consumer spending.",
        "The violin solo brought the entire audience to its feet.",
        "Please submit your report as a PDF before Friday noon.",
        "The storm knocked out power for thousands of homes overnight.",
        "A triangle has interior angles that sum to one hundred eighty degrees.",
        "The museum's new exhibition features Renaissance portraiture.",
        "Regular exercise improves cardiovascular health and sleep quality.",
        "The contract expires on the last day of December.",
        "Compilers translate high-level source code into machine instructions.",
        "The chef recommends the grilled salmon with lemon butter sauce.",
        "Volcanic ash can disrupt air travel across entire continents.",
        "The startup raised ten million dollars in its Series B round.",
        "Ancient Romans built aqueducts that still stand today in Spain.",
        "The children built a sandcastle with towers and a moat.",
        "Dental hygiene requires brushing twice daily and regular flossing.",
        "The satellite entered orbit after a flawless launch sequence.",
        "Inflation erodes purchasing power over long time horizons.",
        "The pianist practiced the sonata for six months before the recital.",
        "Renewable energy now supplies a third of the country's electricity.",
        "The detective examined the fingerprints on the broken window.",
        "Wheat futures rose after reports of drought in the plains.",
        "The telescope revealed a galaxy thirteen billion light years away.",
        "Kindness costs nothing but means everything to someone struggling.",
        "Our team shipped the release candidate after final regression tests.",
    ]
    zh = [
        "机器学习正在改变我们处理自然语言的方式。",
        "北京是中国的首都，有着三千多年的历史。",
        "请帮我写一首关于秋天的短诗。",
        "量子计算利用叠加态和纠缠态进行并行计算。",
        "这家餐厅的水饺和牛肉面都非常有名。",
        "长江是亚洲最长的河流，全长约六千三百公里。",
        "如何在Python中实现一个线程安全的队列？",
        "中秋节是家人团聚、赏月吃月饼的传统节日。",
        "人工智能在医疗影像诊断领域取得了显著进展。",
        "丽江古城以其独特的纳西族文化吸引着游客。",
        "股市的波动反映了投资者对经济前景的预期。",
        "他每天清晨沿着湖边跑步五公里。",
        "这本书讲述了一位科学家的奋斗与坚持。",
        "高铁极大地缩短了城市之间的旅行时间。",
        "垃圾分类是保护环境的有效措施之一。",
        "书法是中国传统艺术的重要组成部分。",
        "医生提醒：均衡饮食和充足睡眠对健康至关重要。",
        "5G网络的高速低延迟特性催生了新的应用场景。",
        "西湖的断桥残雪是杭州著名的景点。",
        "这项研究由清华大学和北京大学联合完成。",
        "咖啡豆的烘焙程度决定了咖啡的风味。",
        "快递公司承诺次日达，偏远地区除外。",
        "新能源汽车的销量在过去五年增长了三倍。",
        "他在面试中展示了扎实的算法功底和项目经验。",
        "长城始建于春秋战国时期，用于防御外敌。",
        "京剧的唱腔和脸谱艺术享誉世界。",
        "该公司计划在未来三年内上市。",
        "秦岭是大熊猫的重要栖息地。",
        "持续集成能够尽早发现软件缺陷。",
        "茶起源于中国，后来传播到世界各地。",
        "数学证明必须严谨，每一步都要有依据。",
        "她对古典音乐的热爱始于童年时的一台旧钢琴。",
        "台风\u201c海燕\u201d给沿海地区带来了强风暴雨。",  # curly quotes inside
        "区块链技术通过共识机制保证数据不可篡改。",
        "志愿者们在山区小学建立了十座图书室。",
        "气温骤降，请注意添衣保暖，预防感冒。",
        "敦煌莫高窟保存了大量精美的壁画和彩塑。",
        "他在论文中提出了一种新的图神经网络架构。",
        "早市的蔬菜新鲜又便宜，深受居民喜爱。",
        "丝绸之路促进了东西方文化的交流。",
    ]
    ja = [
        "機械学習は自然言語処理の分野を大きく変えています。",
        "桜の開花時期は毎年気象庁が発表します。",
        "京都には世界遺産に登録された寺院が多くあります。",
        "新幹線は正確な時刻運行で知られています。",
        "プログラミングの勉強は毎日の継続が大切です。",
        "寿司は日本を代表する伝統料理の一つです。",
        "太陽光発電の導入コストはここ十年で大幅に下がりました。",
        "漫画は日本のポップカルチャーの重要な一部です。",
        "富士山は標高三七七七メートルの成層火山です。",
        "リモートワークの普及で働き方が多様化しています。",
        "温泉は疲労回復に効果があるとされています。",
        "秋は読書の季節という言い方があります。",
        "ロボット技術は製造業の効率化に貢献しています。",
        "納豆は健康食品として国内外で人気です。",
        "野球は日本で最も人気のあるスポーツの一つです。",
        "雨の日は傘と長靴を忘れずに持って行きましょう。",
        "和食は二〇一三年にユネスコ無形文化遺産に登録されました。",
        "深夜のコンビニは都市の安全な灯りのような存在です。",
        "陶芸体験で自分だけの茶碗を作ることができます。",
        "相撲は日本の国技として親しまれています。",
    ]
    ko = [
        "인공지능은 우리 생활을 편리하게 만들어 주고 있습니다.",
        "서울은 대한민국의 수도이자 최대 도시입니다.",
        "김치는 한국 전통 음식으로 건강에 좋습니다.",
        "K-팝은 전 세계적으로 큰 인기를 얻고 있습니다.",
        "한글은 세종대왕이 창제한 과학적인 문자입니다.",
        "제주도는 한국 대표 휴양지로 유명합니다.",
        "매일 아침 조깅하는 것은 건강에 좋은 습관입니다.",
        "반도체 산업은 한국 경제의 핵심입니다.",
        "비빔밥은 다양한 나물을 비벼 먹는 음식입니다.",
        "봄에는 벚꽃 축제가 전국 각지에서 열립니다.",
        "독서는 마음의 양식이 된다고 합니다.",
        "태권도는 올림픽 정식 종목입니다.",
        "지하철은 대도시 교통의 핵심 인프라입니다.",
        "국립중앙박물관은 한국 최대 규모의 박물관입니다.",
        "겨울에는 스키장을 찾는 사람들이 많습니다.",
    ]
    mixed = [
        "The company 推出了新一代AI芯片 in Beijing last week.",
        "请 review 一下这个 pull request，谢谢！",
        "Transformer架构由Attention机制驱动，Attention is all you need.",
        "このpaperを読んで、要点をまとめてください。",
        "I love 北京烤鸭 and 小笼包, they are amazing.",
        "使用Docker容器化部署 microservices 可以简化运维。",
        "그는 machine learning 엔지니어로 일하고 있습니다.",
        "GPU算力是训练大模型的关键，this is well known.",
        "Café München에서 아침 커피를 마셨습니다.",
        "SQL查询优化是database tuning的重要一环。",
        "Der schnelle Braune Fuchs springt über den faulen Hund。",
        "La inteligencia artificial está transformando el mundo entero.",
        "東京タワーは Tokyo の象徴的なランドマークです。",
        "在Kubernetes集群上运行PyTorch分布式训练。",
        "Москва — крупнейший город России, with rich history.",
    ]
    code = [
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr",
        "for (int i = 0; i < n; ++i) { sum += a[i] * b[i]; }",
        "SELECT name, COUNT(*) FROM users GROUP BY name HAVING COUNT(*) > 5;",
        "class Attention(nn.Module):\n    def forward(self, q, k, v):\n        return softmax(q @ k.T / sqrt(d)) @ v",
        "git commit -m \"fix: resolve race condition in scheduler\"",
        "curl -X POST https://api.example.com/v1/embeddings -H \"Content-Type: application/json\"",
        "template<typename T> constexpr T clamp(T v, T lo, T hi) { return v < lo ? lo : (v > hi ? hi : v); }",
        "x = np.einsum('bsd,btd->bst', q, k) / np.sqrt(q.shape[-1])",
        "async fn fetch(url: &str) -> Result<String, Box<dyn Error>> { /* ... */ }",
        "<div class=\"container\"><p>Hello, world!</p></div>",
        "fn main() { println!(\"Hello, world!\"); }",
        "MOV AX, 0x4C00\nINT 21h",
        "println(json.dumps({\"status\": 200, \"ok\": True}, indent=2))",
        "const f = (x) => x * x + 2 * x + 1;",
        "if __name__ == '__main__': main()",
        "#!/usr/bin/env bash\nset -euo pipefail",
        "docker run --gpus all -v $(pwd):/work nvcr.io/nvidia/pytorch:24.01-py3",
        "enum class Color : uint8_t { Red, Green, Blue };",
        "expect(std::abs(actual - expected) < 1e-5f);",
        "CREATE INDEX idx_users_email ON users(email);",
    ]
    emoji_special = [
        "I love pizza 🍕 and sushi 🍣! 🎉🎉🎉",
        "🚀🚀🚀 To the moon!!! 💎🙌",
        "心情很好😊，今天天气不错☀️。",
        "Flags: 🇨🇳🇯🇵🇰🇷🇺🇸🇫🇷 — smileys: 😀😃😄😁",
        "Math symbols: ∑∏∫√∞≈≠≤≥±÷×",
        "Fullwidth chars：Ｈｅｌｌｏ　Ｗｏｒｌｄ！",
        "★☆♠♣♥♦ •‿• ¯\\_(ツ)_/¯",
        "Emoji with ZWJ: 👨‍👩‍👧‍👦 and flags 🇪🇺.",
        "Symbols: ™ © ® § ¶ † ‡ • …",
        "Mixed １２３４５ and 12345 and Ⅻ numbers.",
    ]
    whitespace_edge = [
        "   leading spaces",
        "trailing spaces   ",
        "multiple   internal    spaces",
        "\ttab at start",
        "tab\tin\tthe\tmiddle",
        "line with\nembedded newline char",
        "carriage\rreturn char",
        " ",
        "  \t ",
        "a",
        "word",  # duplicate content, different position (tests corpus integrity only)
        "The THE the tHe case variants",
    ]
    numbers_urls = [
        "Visit https://example.com/docs for more information.",
        "Contact me at user.name+tag@example-domain.org please.",
        "Call +1-800-555-0199 or 010-12345678 for support.",
        "Order #12345 shipped on 2024-03-15, total $199.99.",
        "π ≈ 3.14159265358979323846264338327950288",
        "The ratio 3:1 yields 75% success in 2^10 trials.",
        "IPv4 192.168.0.1 and IPv6 ::1 are loopback addresses.",
        "CVE-2024-1234 affects versions < 2.0.1, upgrade now!",
        "50% off until 12/31/2025 — don't miss it!",
        "Coordinates: 39.9042° N, 116.4074° E (Beijing).",
    ]
    long1 = ("Long context stress test. " * 120).strip() + " End of long document one."
    long2 = ("这是一段用于测试长文本编码的中文文档。" * 100).strip() + " 文档二结束。"
    corpus = (en + zh + ja + ko + mixed + code + emoji_special +
              whitespace_edge + numbers_urls + [long1, long2])
    return corpus


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    corpus = _corpus()
    assert len(corpus) == len(set(corpus)), "corpus must not contain duplicates"
    for s in corpus:
        if "\r" in s:
            assert s == "carriage\rreturn char", f"unexpected \\r in corpus sentence: {s!r}"
    n = len(corpus)
    print(f"corpus: {n} sentences")

    # ---- tokenize (HF fast tokenizer == engine's tokenizer.json ground truth) ----
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    tok.enable_truncation(max_length=MAX_SEQ_LEN)
    entries = []
    ids_all = []
    for i, s in enumerate(corpus):
        e = tok.encode(s)
        ids = e.ids
        entries.append({"i": i, "n": len(ids), "trunc": len(ids) >= MAX_SEQ_LEN, "ids": ids})
        ids_all.append(ids)
    n_trunc = sum(1 for e in entries if e["trunc"])
    print(f"tokenized: {n} sentences, {n_trunc} hit the {MAX_SEQ_LEN} truncation limit")

    # ---- normalizer cross-check: tokenizer.json charsmap == sentencepiece model blob ----
    tj = json.loads((MODEL_DIR / "tokenizer.json").read_text(encoding="utf-8"))
    b64_blob = None
    for step in tj["normalizer"]["normalizers"]:
        if step.get("type") == "Precompiled":
            b64_blob = step["precompiled_charsmap"]
    assert b64_blob is not None
    blob_json = base64.b64decode(b64_blob)
    # minimal protobuf wire scan of sentencepiece.bpe.model:
    # ModelProto.field 3 = normalizer_spec (trainer_spec is field 2);
    # NormalizerSpec.field 3 = precompiled_charsmap
    raw = (MODEL_DIR / "sentencepiece.bpe.model").read_bytes()

    def _fields(buf: bytes):
        pos = 0
        while pos < len(buf):
            key = 0
            shift = 0
            while True:
                b = buf[pos]; pos += 1
                key |= (b & 0x7F) << shift
                shift += 7
                if not b & 0x80:
                    break
            field, wire = key >> 3, key & 7
            if wire == 2:
                ln = 0; shift = 0
                while True:
                    b = buf[pos]; pos += 1
                    ln |= (b & 0x7F) << shift
                    shift += 7
                    if not b & 0x80:
                        break
                yield field, wire, buf[pos:pos + ln]
                pos += ln
            elif wire == 0:
                while buf[pos] & 0x80:
                    pos += 1
                pos += 1
            elif wire == 5:
                pos += 4
            elif wire == 1:
                pos += 8
            else:
                raise ValueError(f"unsupported wire type {wire}")

    norm_spec = next(v for f, w, v in _fields(raw) if f == 3 and w == 2)
    # NormalizerSpec.field 2 = precompiled_charsmap
    blob_sp = next(v for f, w, v in _fields(norm_spec) if f == 2 and w == 2)
    assert blob_json == blob_sp, "precompiled charsmap mismatch between tokenizer.json and .model"
    print(f"charsmap cross-check ok ({len(blob_json)} bytes, identical in both sources)")

    # ---- ONNX inference: dense + token_embeddings ----
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(MODEL_DIR / "onnx/model.onnx"), so,
                                providers=["CPUExecutionProvider"])

    def run_batch(batch_ids):
        maxlen = max(len(x) for x in batch_ids)
        padded = np.zeros((len(batch_ids), maxlen), dtype=np.int64)
        mask = np.zeros((len(batch_ids), maxlen), dtype=np.int64)
        for r, x in enumerate(batch_ids):
            padded[r, :len(x)] = x
            mask[r, :len(x)] = 1
        feeds = {"input_ids": padded, "attention_mask": mask}
        return sess.run(None, feeds)  # token_embeddings, sentence_embedding

    te_out = [None] * n          # per-sentence token_embeddings (unpadded)
    dense = np.zeros((n, 1024), dtype=np.float32)
    for b0 in range(0, n, BATCH):
        batch_ids = ids_all[b0:b0 + BATCH]
        te, se = run_batch(batch_ids)
        for r, x in enumerate(batch_ids):
            te_out[b0 + r] = te[r, :len(x), :].copy()
            dense[b0 + r] = se[r]
        print(f"onnx batch {b0}..{b0+len(batch_ids)-1} done")

    # sentence_embedding must equal L2-normalized CLS token (Pooling CLS + Normalize)
    for i in range(n):
        cls = te_out[i][0]
        ref = cls / max(np.linalg.norm(cls), 1e-12)
        c = float(np.dot(ref, dense[i]) /
                  (np.linalg.norm(ref) * np.linalg.norm(dense[i]) + 1e-12))
        if c < 0.99999:
            print(f"WARNING: dense/CLS mismatch at {i}: cos={c}")
    print("dense == normalized CLS check done")

    # ---- sparse + colbert heads (torch .pt checkpoints) ----
    import torch
    sp_ckpt = torch.load(MODEL_DIR / "sparse_linear.pt", map_location="cpu", weights_only=True)
    cb_ckpt = torch.load(MODEL_DIR / "colbert_linear.pt", map_location="cpu", weights_only=True)
    w_sparse = sp_ckpt["weight"].numpy().astype(np.float32)   # [1,1024]
    b_sparse = sp_ckpt["bias"].numpy().astype(np.float32)     # [1]
    w_colb = cb_ckpt["weight"].numpy().astype(np.float32)     # [1024,1024]
    b_colb = cb_ckpt["bias"].numpy().astype(np.float32)       # [1024]
    sparse_entries = []
    colbert_blobs = []
    for i in SUBSET_TE:
        x = te_out[i] # [L,1024]
        # sparse_linear maps 1024 -> 1: a scalar per TOKEN position [L,1].
        # bge-m3 lexical weights are per-token scalars keyed by the token's
        # own id; positions with id <= 2 (specials) are dropped.
        svec = np.maximum(x @ w_sparse.T + b_sparse, 0.0).reshape(-1) # relu, [L]
        ids_i = ids_all[i]
        nz = [(int(ids_i[t]), float(svec[t])) for t in range(svec.shape[0])
              if svec[t] > 0.0 and int(ids_i[t]) > 2]
        sparse_entries.append({"i": i, "pairs": nz})
        y = x @ w_colb.T + b_colb                              # [L,1024]
        ids_i = ids_all[i]
        norms = np.linalg.norm(y, axis=1, keepdims=True)
        y = y / np.maximum(norms, 1e-12)
        keep = [j for j, t in enumerate(ids_i) if t > 2]       # drop <s>, pad, </s>
        colbert_blobs.append((i, y[keep]))
    print(f"sparse/colbert goldens for {len(SUBSET_TE)} sentences")

    # ---- per-layer activations via transformers (D4 bisect oracle) ----
    from transformers import AutoModel
    model = AutoModel.from_pretrained(str(MODEL_DIR), torch_dtype=torch.float32)
    model.eval()
    layer_goldens = []  # (sentence_idx, list of 25 hidden states [L,1024])
    for i in SUBSET_LAYERS:
        ids_i = ids_all[i]
        inp = torch.tensor([ids_i], dtype=torch.long)
        att = torch.ones_like(inp)
        with torch.no_grad():
            out = model(input_ids=inp, attention_mask=att, output_hidden_states=True)
        hs = [h[0].numpy().astype(np.float32) for h in out.hidden_states]  # 25 states
        # cross-check last hidden state against ONNX token_embeddings
        diff = float(np.max(np.abs(hs[-1] - te_out[i])))
        print(f"layers[{i}]: 25 states, max|hf-onnx| last state = {diff:.3e}")
        layer_goldens.append((i, hs))

    # ---- write outputs ----
    # one sentence per line; the two corpus sentences that intentionally contain
    # control chars are escaped (\\n \\r) and must be unescaped by the reader.
    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r")
    (OUT_DIR / "corpus.txt").write_text("".join(esc(s) + "\n" for s in corpus), encoding="utf-8")
    (OUT_DIR / "golden_tokens.json").write_text(json.dumps({
        "max_seq_len": MAX_SEQ_LEN,
        "count": n,
        "entries": entries}), encoding="utf-8")
    dense.astype("<f4").tofile(OUT_DIR / "golden_dense.bin")

    meta = {
        "model": "bge-m3 (XLM-RoBERTa, 24L/1024H/16heads)",
        "max_seq_len": MAX_SEQ_LEN,
        "count": n,
        "corpus_escapes": "corpus.txt lines escape backslash, \\n and \\r; readers must unescape",
        "onnx": {"ir_version": 6, "opset": 11, "nodes": 2495, "initializers": 389},
        "dense": {"file": "golden_dense.bin", "rows": n, "dim": 1024, "dtype": "float32-le",
                  "definition": "ONNX sentence_embedding output (CLS pool + L2 norm)"},
        "sha256": {},
        "truncated_sentences": [e["i"] for e in entries if e["trunc"]],
    }
    for f in ["onnx/model.onnx", "onnx/model.onnx_data", "tokenizer.json",
              "sparse_linear.pt", "colbert_linear.pt", "pytorch_model.bin"]:
        h = hashlib.sha256()
        with open(MODEL_DIR / f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        meta["sha256"][f] = h.hexdigest()
        print("sha256", f, h.hexdigest()[:16] + "...")

    # token_embeddings subset
    off = 0
    te_meta = []
    with open(OUT_DIR / "golden_te.bin", "wb") as f:
        for i in SUBSET_TE:
            x = te_out[i].astype("<f4")
            f.write(x.tobytes())
            te_meta.append({"i": i, "rows": int(x.shape[0]), "dim": 1024,
                            "offset_bytes": off})
            off += x.nbytes
    meta["token_embeddings"] = {"file": "golden_te.bin", "dtype": "float32-le",
                                "entries": te_meta}
    (OUT_DIR / "golden_sparse.json").write_text(json.dumps({
        "definition": "relu(token_embeddings @ sparse_w.T + b), kept per token with id>2",
        "entries": sparse_entries}), encoding="utf-8")
    off = 0
    cb_meta = []
    with open(OUT_DIR / "golden_colbert.bin", "wb") as f:
        for i, y in colbert_blobs:
            f.write(y.astype("<f4").tobytes())
            cb_meta.append({"i": i, "rows": int(y.shape[0]), "dim": 1024,
                            "offset_bytes": off})
            off += y.nbytes
    meta["colbert"] = {"file": "golden_colbert.bin", "dtype": "float32-le",
                       "definition": "per-token L2-normalized colbert_linear output, special tokens dropped",
                       "entries": cb_meta}
    off = 0
    ly_meta = []
    with open(OUT_DIR / "golden_layers.bin", "wb") as f:
        for i, hs in layer_goldens:
            for l, h in enumerate(hs):
                f.write(h.astype("<f4").tobytes())
                ly_meta.append({"i": i, "layer": l, "rows": int(h.shape[0]),
                                "dim": 1024, "offset_bytes": off})
                off += h.nbytes
    meta["layers"] = {"file": "golden_layers.bin", "dtype": "float32-le",
                      "note": "layer 0 = embedding output, layer k = encoder layer k-1 output (post-LN), layer 24 = final LN output",
                      "entries": ly_meta}
    (OUT_DIR / "golden_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote goldens to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
