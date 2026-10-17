#!/usr/bin/env python3
"""L3 retrieval fixture generator for the bge-m3 embedding engine (dev-only Python).

Produces the frozen `test_embed_retrieval` fixtures of
embedding_model_test_plan.md §4.3 (generation rules from §1 / §4.1):

    src/tests/data/embedding/retrieval/
        corpus.jsonl                 5000 passages   {"id","text"}
        queries.jsonl                 200 queries    {"id","text"}
        golden_dense_scores.fp32.bin  float32-le row-major [200][5000]
        golden_sparse_scores.fp32.bin float32-le row-major [200][5000]
        qrels.json                    golden ordered top-10 (dense + sparse), tie groups,
                                      human gold_ids, manifest meta (shapes, pins, hashes,
                                      sanity stats)
        README.md                     provenance + sha256 manifest + stats + how to rerun

Nothing outside `retrieval/` is written. The parent sentence-level goldens
(src/tests/data/embedding/{corpus.txt,golden_*.bin/golden_*.json,golden_meta.json,
MANIFEST.json}) are frozen (plan §1.1 / §8(a)); this script only *reads* the model
files and never regenerates them.

Reference = PyTorch bge-m3 on CPU, fp32 (never the engine):
  dense : last_hidden_state -> CLS row -> L2 normalize (the model's 1_Pooling(cls) +
          2_Normalize modules; cross-checked against sentence-transformers
          encode(normalize_embeddings=True) at run time).
  sparse: lexical weight per token = relu(h @ W_sparse.T + b_sparse), W_sparse/b_sparse
          from sparse_linear.pt (fp16 values, fp32 compute), kept for token ids > 2
          (same convention as the frozen ../golden_sparse.json), zero weights dropped,
          max-pooled per token id within a document.
          score(q,p) = sum over token ids shared by q and p of w_q * w_p, accumulated in
          ascending token-id order (deterministic summation order).
  truncation: 512 tokens == golden_meta.json["max_seq_len"] (the engine's default knob).

Corpus/query provenance (public eval sets, Hugging Face datasets-server /rows API with
`revision` pinned; page counts bounded, responses cached on disk):
  EN   mteb/nfcorpus              CORDIS biomedical passages + human qrels + queries
  ZH   Hieuman/zh_MIRACL          MIRACL-zh style triples (query / positives / negatives)
  JA   crystina-z/mmarco-passage  mMARCO-japanese triples (query / positives / negatives)
  KO   Hieuman/ko_MIRACL          MIRACL-ko triples
  **   miracl/miracl-corpus       cross-domain Wikipedia filler passages, sampled at
                                  deterministic offsets (en/zh/ja/ko)
Each selected query contributes 1..MAX_GOLDS_PER_QUERY of its human-judged relevant
passages to the corpus, so every query (except the 2 synthetic gibberish negative
controls) has 1-5 clearly relevant passages. 3 exact duplicate passages are appended as
the dedup/adjacency negative control of plan §4.3.

Determinism / re-runnability:
  * SHA-256 counter PRNG (independent of Python's RNG implementation), fixed quotas,
    sorted/deterministic iteration everywhere, json dumps with sort_keys=True;
  * row counts of every remote table are asserted against EXPECTED_ROWS, so a moved
    upstream revision aborts instead of silently resampling;
  * HTTP responses cached (default $EMBED_FIXTURE_CACHE or <tempdir>/
    bge_m3_retrieval_fixture_cache) => `--offline` replays them with no network;
  * fp32 CPU torch with a fixed intra-op thread count and fixed batch order.
  Byte-identity is guaranteed on the same machine/torch build; elsewhere the last fp32
  bits may differ, which is why the committed sha256 values (README/qrels meta) are the
  authority and ranking parity uses the §4.3 tie rule rather than bitwise equality.

Usage (dev env: torch(cpu), transformers, sentence-transformers, tokenizers, numpy):

    python scripts/embedding/gen_retrieval_fixtures.py                  # full run
    python scripts/embedding/gen_retrieval_fixtures.py --stage data     # jsonl only
    python scripts/embedding/gen_retrieval_fixtures.py --offline        # replay cache
    python scripts/embedding/gen_retrieval_fixtures.py --verify-only    # re-hash + diff
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- constants
REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO / "bge-m3" / "bge-m3"
OUT_DIR = REPO / "src" / "tests" / "data" / "embedding" / "retrieval"

SEED = 20261017            # fixed sampling seed
MAX_SEQ_LEN = 512          # token truncation limit
BATCH = 64                 # encode batch size
TORCH_THREADS = 24         # fixed intra-op thread count (fp32 CPU determinism)
TIE_EPS = 1e-5             # tie window, plan §4.3
TOP_K = 10
CORPUS_N = 5000
QUERY_N = 200
MAX_GOLDS_PER_QUERY = 5
N_DUP_ENTRIES = 3          # exact duplicate corpus rows (negative control)
MIN_QUERY_CHARS = 4
MIN_PASSAGE_CHARS = 16
MAX_PASSAGE_CHARS = 6000   # raw-char guard; the 512-token limit does the real truncation

# language quotas; blocks + duplicates must land exactly on CORPUS_N
CORPUS_QUOTA = {"en": 1299, "zh": 1499, "ja": 999, "ko": 1200}   # sum 4997 + 3 dups
QUERY_QUOTA = {"en": 55, "zh": 50, "ja": 45, "ko": 48}           # sum 198 + 2 gibberish
SAME_ROW_NEGATIVES = 2       # hard distractors taken from an anchored query's own row
# reserved cross-domain budget inside each language block (MIRACL Wikipedia filler)
MIRACL_FILLER_QUOTA = {"en": 300, "zh": 350, "ja": 250, "ko": 300}
MIRACL_LANGS = ("en", "zh", "ja", "ko")
# Gibberish negative controls (plan §4.3: "a query of gibberish must return low scores
# everywhere — max score below the median golden score"). A candidate pool is encoded,
# then the GIBBERISH_N candidates with the *lowest* top dense score are frozen into
# queries.jsonl, so the fixture provably satisfies its own negative control.
GIBBERISH_N = 2
GIBBERISH_CANDIDATES = [
    "asdf qwerty zxcvb jkl mnbvc xz q fjdk lfjdf lkjfdjg",
    "zzzz xqqvk fjord 0110 0111 1000 xqz qqq wvvv ppppp",
    "!!! ??? ### $$$ %%% &&& ((( ))) ~~~ ^^^ *** +++ ---",
    "7419 3382 5077 1600 8842 9135 2760 4418 6027 5391",
    "bbb dd kk zz nn rr tt qq ss ll mm pp cc vv xx",
    "xq. vzb. kkp. rw. tq. fj. lmn. zq. wx. cd. vb.",
    "zxcv bnm, asdf qwer tyui op[ 3132 4354 6576 8798",
    "zh zh zh zh zh zh zh zh zh zh zh zh zh zh zh zh",
]

SERVER = "https://datasets-server.huggingface.co"
PAGE = 100                                   # /rows hard cap
NFC_QRELS_PAGES = 40                         # only complete query-id groups are used
NFC_ALL_PAGES = None                         # None = read the whole corpus/queries table
MMARCO_JA_PAGES = 6                          # ~600 ja triples are plenty for 45 queries

REVISIONS = {
    "mteb/nfcorpus": "52ac3f19d3449632d9f00aab0ad34a110fc03816",
    "Hieuman/zh_MIRACL": "1ec5797bced7557cf0b37e4f7a260e34f92f7bcc",
    "Hieuman/ko_MIRACL": "6fc42e5d99ae42c62af578998294d664d1c79212",
    "crystina-z/mmarco-passage": "2cacf5e3d2474f6320df95c5e23253ecd390a071",
    "miracl/miracl-corpus": "d921ec7e349ce0d28daf30b2da9da5ee698bef0d",
}
EXPECTED_ROWS = {
    ("mteb/nfcorpus", "corpus", "corpus"): 3633,
    ("mteb/nfcorpus", "queries", "queries"): 3237,
    ("mteb/nfcorpus", "default", "dev"): 11385,
    ("Hieuman/zh_MIRACL", "default", "train"): 1699,
    ("Hieuman/ko_MIRACL", "default", "train"): 799,
    ("crystina-z/mmarco-passage", "japanese", "train"): 352000,
    ("miracl/miracl-corpus", "en", "train"): 11254000,
    ("miracl/miracl-corpus", "zh", "train"): 4934368,
    ("miracl/miracl-corpus", "ja", "train"): 6953614,
    ("miracl/miracl-corpus", "ko", "train"): 1486752,
}
SRC_TAG = {
    "en": "nfcorpus",
    "zh": "miracl-zh-triples",
    "ja": "mmarco-ja-triples",
    "ko": "miracl-ko-triples",
}
DATA_FILES = ["corpus.jsonl", "queries.jsonl", "golden_dense_scores.fp32.bin",
              "golden_sparse_scores.fp32.bin", "qrels.json"]
MODEL_FILES = ["tokenizer.json", "sparse_linear.pt", "colbert_linear.pt", "pytorch_model.bin"]


def log(msg: str) -> None:
    print(msg, flush=True)


# ------------------------------------------------------------------------------ helpers
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


@dataclass
class DetRand:
    """SHA-256 counter PRNG: stable across Python versions and platforms."""

    key: bytes
    counter: int = 0

    def u64(self) -> int:
        self.counter += 1
        raw = hashlib.sha256(self.key + self.counter.to_bytes(8, "big")).digest()
        return int.from_bytes(raw[:8], "big")

    def below(self, n: int) -> int:
        return self.u64() % n

    def shuffled(self, seq: list) -> list:
        """Deterministic Fisher-Yates permutation (returns a new list)."""
        items = list(seq)
        for i in range(len(items) - 1, 0, -1):
            j = self.below(i + 1)
            items[i], items[j] = items[j], items[i]
        return items


def norm_ws(s: str) -> str:
    """Collapse whitespace runs to single spaces, drop control chars, strip."""
    s = "".join(" " if (ord(c) < 0x20 and c != " ") or ord(c) == 0xFEFF else c for c in s)
    return re.sub(r" +", " ", s).strip()


def join_title(title: str, text: str) -> str:
    t, x = norm_ws(title or ""), norm_ws(text or "")
    if not t or x.startswith(t):
        return x or t
    if not x:
        return t
    return f"{t} {x}"


def unwrap_literal(s: str) -> str:
    """Some sources ship `\"['text']\"` (stringified one-element lists): unwrap them."""
    s = s.strip()
    if len(s) > 3 and s[0] == "[" and s[-1] == "]":
        inner = s[1:-1].strip()
        for quote in ("'", '"'):
            if len(inner) > 1 and inner.startswith(quote) and inner.endswith(quote):
                cand = inner[1:-1]
                if quote not in cand and "\\" not in cand:
                    return cand
    return s


def uniq_keep_order(xs: list[str]) -> list[str]:
    s: set[str] = set()
    return [x for x in xs if x and not (x in s or s.add(x))]


# ------------------------------------------------------------------------------ http
@dataclass
class Http:
    """GET with an on-disk response cache and datasets-server-friendly pacing.

    The HF datasets-server rate-limits aggressively (HTTP 429), so requests are spaced
    by `min_interval` seconds and 429/5xx honour Retry-After with exponential backoff.
    """

    cache_dir: Path
    offline: bool
    timeout: int = 90
    retries: int = 9
    min_interval: float = 0.35
    calls: int = 0
    hits: int = 0
    _last: float = 0.0

    def bytes_for(self, url: str) -> bytes:
        path = self.cache_dir / (hashlib.sha256(url.encode("utf-8")).hexdigest() + ".json")
        if path.exists():
            self.hits += 1
            return path.read_bytes()
        if self.offline:
            raise RuntimeError(f"--offline and cache miss for {url}")
        delay, last = 2.0, None
        for _ in range(self.retries):
            gap = time.time() - self._last
            if gap < self.min_interval:
                time.sleep(self.min_interval - gap)
            self._last = time.time()
            self.calls += 1
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "bge-m3-fixture-gen"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    body = r.read()
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                tmp.write_bytes(body)
                tmp.replace(path)
                return body
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as e:
                last = e
                self._last = time.time()
                code = getattr(e, "code", None)
                if isinstance(code, int) and 400 <= code < 500 and code != 429:
                    break
                retry_after = None
                hdr = getattr(getattr(e, "headers", None), "get", lambda _k: None)("Retry-After")
                try:
                    retry_after = float(hdr) if hdr else None
                except ValueError:
                    retry_after = None
                time.sleep(retry_after if retry_after is not None else delay)
                delay = min(max(delay, retry_after or 0.0) * 2, 90.0)
        raise RuntimeError(f"GET failed for {url}: {last}")

    def json_for(self, url: str):
        return json.loads(self.bytes_for(url).decode("utf-8"))


def rows_page(http: Http, dataset: str, config: str, split: str, offset: int,
              length: int = PAGE) -> tuple[list[dict], int]:
    url = (f"{SERVER}/rows?dataset={urllib.parse.quote(dataset)}"
           f"&config={urllib.parse.quote(config)}&split={urllib.parse.quote(split)}"
           f"&offset={offset}&length={length}&revision={REVISIONS[dataset]}"
           f"&partial=false&new=true")
    d = http.json_for(url)
    total = int(d.get("num_rows_total", -1))
    want = EXPECTED_ROWS.get((dataset, config, split))
    if want is not None and total != want:
        raise RuntimeError(
            f"{dataset}[{config}/{split}] reports {total} rows, pinned expectation is "
            f"{want}: upstream moved or revision pin ignored — aborting (plan §1.1)")
    return [r["row"] for r in d.get("rows", [])], total


def page_table(http: Http, dataset: str, config: str, split: str,
               max_pages: int | None = None) -> list[dict]:
    out: list[dict] = []
    off = 0
    _, total = rows_page(http, dataset, config, split, 0)
    limit = total if max_pages is None else min(total, max_pages * PAGE)
    while off < limit:
        got, _ = rows_page(http, dataset, config, split, off)
        if not got:
            break
        out.extend(got)
        off += len(got)
    return out


# ------------------------------------------------------------------------ fixture records
@dataclass
class Passage:
    text: str
    lang: str
    src: str
    gold_for: list[int] = field(default_factory=list)
    dup_of: int | None = None


@dataclass
class Query:
    text: str
    lang: str
    src: str
    golds: list[int] = field(default_factory=list)


@dataclass
class Fixture:
    corpus: list[Passage]
    queries: list[Query]
    provenance: list[dict]
    notes: list[str]
    dup_pairs: list[tuple[int, int]]


# --------------------------------------------------------------------------- stage: data
def _load_en(http: Http):
    """EN candidates (qtexts, gold texts) + the full EN passage pool."""
    qrels = page_table(http, "mteb/nfcorpus", "default", "dev", NFC_QRELS_PAGES)
    groups: dict[str, list[str]] = {}
    for r in qrels:
        if float(r["score"]) > 0:
            groups.setdefault(str(r["query-id"]), []).append(str(r["corpus-id"]))
    last_qid = str(qrels[-1]["query-id"]) if qrels else None
    corpus_rows = page_table(http, "mteb/nfcorpus", "corpus", "corpus", NFC_ALL_PAGES)
    texts = {str(r["_id"]): join_title(r.get("title", ""), r.get("text", ""))
             for r in corpus_rows}
    query_rows = page_table(http, "mteb/nfcorpus", "queries", "queries", NFC_ALL_PAGES)
    qtexts = {str(r["_id"]): norm_ws(r.get("text", "")) for r in query_rows}
    # nfcorpus judgments are heavy-tailed (1..170 golds/query). Prefer the queries with
    # the *fewest* judged-relevant passages so that the anchored subset covers their
    # judgment set completely, and never anchor more than MAX_GOLDS_PER_QUERY passages.
    cand = []
    complete = 0
    for qid in sorted(groups):
        if qid == last_qid:                 # group may be cut by the page window
            continue
        golds = uniq_keep_order([texts[p] for p in groups[qid] if p in texts])
        golds = [g for g in golds if len(g) >= MIN_PASSAGE_CHARS]
        q = qtexts.get(qid, "")
        if len(golds) >= 1 and len(q) >= MIN_QUERY_CHARS:
            cand.append((q, golds[:MAX_GOLDS_PER_QUERY], len(golds)))
            if len(golds) <= MAX_GOLDS_PER_QUERY:
                complete += 1
    cand.sort(key=lambda c: (c[2], c[0]))   # fewest judgments first
    cand = cand[: 3 * QUERY_QUOTA["en"]]   # sampling pool biased to fully covered queries
    pool = [t for _, t in sorted(texts.items()) if len(t) >= MIN_PASSAGE_CHARS]
    return [(q, g) for (q, g, _n) in cand], pool, len(cand), complete


def _load_triples(http: Http, dataset: str, config: str, split: str,
                 pos_key: str, neg_key: str, pos_text_key: str | None,
                 max_pages: int | None) -> list[tuple[str, list[str], list[str]]]:
    """query/(positives/negatives) triple tables in two shapes:
    Hieuman MIRACL lists of strings, mMARCO lists of {docid,text} dicts."""
    rows = page_table(http, dataset, config, split, max_pages)
    out = []
    for r in rows:
        q = unwrap_literal(norm_ws(str(r.get("query", ""))))

        def _texts(field):
            vals = []
            for x in (r.get(field) or []):
                t = norm_ws(str(x if pos_text_key is None else x.get(pos_text_key, "")))
                vals.append(t)
            return uniq_keep_order([t for t in vals if len(t) >= MIN_PASSAGE_CHARS])

        pos, neg = _texts(pos_key), _texts(neg_key)
        if len(q) >= MIN_QUERY_CHARS and 1 <= len(pos) <= MAX_GOLDS_PER_QUERY:
            out.append((q, pos, neg))
    return out


def _miracl_windows(http: Http, lang: str, n_pages: int, rng: DetRand) -> list[str]:
    """n_pages random 100-row windows of the MIRACL v1.0 corpus for `lang`."""
    total = EXPECTED_ROWS[("miracl/miracl-corpus", lang, "train")]
    offsets: list[int] = []
    guard = 0
    while len(offsets) < n_pages and guard < 4000:
        guard += 1
        o = rng.below(max(1, total - PAGE))
        if all(abs(o - x) >= PAGE for x in offsets):
            offsets.append(o)
    out: list[str] = []
    for o in sorted(offsets):
        got, _ = rows_page(http, "miracl/miracl-corpus", lang, "train", o)
        for r in got:
            t = join_title(r.get("title", ""), r.get("text", ""))[:MAX_PASSAGE_CHARS]
            if len(t) >= MIN_PASSAGE_CHARS:
                out.append(t)
    return out


def build_fixture(http: Http) -> Fixture:
    """Assemble the corpus and query tables deterministically (SEED-driven SHA-256 PRNG).

    Passes: A anchor the human-relevant golds of every selected query,
    B same-row negatives (hard distractors), C same-source distractors,
    D cross-domain MIRACL filler (fixed per-language budget), then flatten into
    language blocks, E append exact duplicates and F the gibberish queries.
    """
    rng = DetRand(hashlib.sha256(f"retrieval-fixture/{SEED}".encode("utf-8")).digest())
    notes: list[str] = []

    en_cand, en_pool, en_total, en_complete = _load_en(http)
    zh_pool = _load_triples(http, "Hieuman/zh_MIRACL", "default", "train",
                            "positive", "negative", None, None)
    ko_pool = _load_triples(http, "Hieuman/ko_MIRACL", "default", "train",
                            "positive", "negative", None, None)
    ja_pool = _load_triples(http, "crystina-z/mmarco-passage", "japanese", "train",
                            "positive_passages", "negative_passages", "text", MMARCO_JA_PAGES)
    # unified record shape: (query text, [gold passages], [negative passages])
    pools = {"en": [(q, g, []) for (q, g) in en_cand],
             "zh": zh_pool, "ja": ja_pool, "ko": ko_pool}
    log(f"pools: nfcorpus {len(pools['en'])} queries / {len(en_pool)} passages, "
        f"zh {len(zh_pool)}, ja {len(ja_pool)}, ko {len(ko_pool)}")

    selected: dict[str, list[tuple[str, list[str], list[str]]]] = {}
    for lang in MIRACL_LANGS:
        pool = pools[lang]
        if len(pool) < QUERY_QUOTA[lang]:
            raise RuntimeError(f"not enough {lang} query candidates: {len(pool)}")
        idx = sorted(rng.shuffled(list(range(len(pool))))[: QUERY_QUOTA[lang]])
        rows = sorted(((pool[i][0], pool[i][1], list(pool[i][2])) for i in idx),
                      key=lambda r: r[0])
        selected[lang] = rows
        note_extra = ""
        if lang == "en":
            note_extra = (f"; {en_complete} of {en_total} candidates have at most "
                          f"{MAX_GOLDS_PER_QUERY} judgments (fully covered), the rest are "
                          f"capped to their first {MAX_GOLDS_PER_QUERY} judged passages")
        notes.append(f"{lang}: {len(rows)} of {len(pool)} candidate queries selected; "
                     f"each carries 1..{MAX_GOLDS_PER_QUERY} human-relevant passages"
                     f"{note_extra}")

    blocks: dict[str, list[Passage]] = {lang: [] for lang in MIRACL_LANGS}
    by_text: dict[str, Passage] = {}

    def add(text: str, lang: str, src: str) -> Passage | None:
        text = text[:MAX_PASSAGE_CHARS]
        if len(text) < MIN_PASSAGE_CHARS:
            return None
        old = by_text.get(text)
        if old is not None:
            return old                       # global dedup: same text -> same corpus row
        if len(blocks[lang]) >= CORPUS_QUOTA[lang]:
            return None
        p = Passage(text, lang, src)
        by_text[text] = p
        blocks[lang].append(p)
        return p

    # A) anchored golds
    for lang in MIRACL_LANGS:
        for (qtext, golds, _neg) in selected[lang]:
            placed = sum(1 for g in golds if add(g, lang, SRC_TAG[lang]) is not None)
            if placed < 1:
                raise RuntimeError(f"query lost all gold passages: {qtext[:60]!r}")

    # B+C) distractors from the same domain, up to quota minus the cross-domain budget
    for lang in MIRACL_LANGS:
        cap = CORPUS_QUOTA[lang] - MIRACL_FILLER_QUOTA[lang]
        for (_q, _g, negs) in selected[lang]:
            for n in negs[:SAME_ROW_NEGATIVES]:
                if len(blocks[lang]) >= cap:
                    break
                add(n, lang, SRC_TAG[lang])
        extras: list[str] = list(en_pool) if lang == "en" else []
        if lang != "en":
            chosen = {r[0] for r in selected[lang]}
            for (q, pos, neg) in pools[lang]:
                if q in chosen:
                    continue
                extras.extend(pos)
                extras.extend(neg)
        for t in rng.shuffled(extras):
            if len(blocks[lang]) >= cap:
                break
            add(t, lang, SRC_TAG[lang])

    # D) cross-domain filler: random MIRACL v1.0 Wikipedia windows
    for lang in MIRACL_LANGS:
        need = CORPUS_QUOTA[lang] - len(blocks[lang])
        pages = max(1, need // PAGE + 1)
        rounds = 0
        while len(blocks[lang]) < CORPUS_QUOTA[lang] and rounds < 12:
            for t in _miracl_windows(http, lang, pages, rng):
                if len(blocks[lang]) >= CORPUS_QUOTA[lang]:
                    break
                add(t, lang, f"miracl-corpus-{lang}")
            rounds += 1
        if len(blocks[lang]) != CORPUS_QUOTA[lang]:
            raise RuntimeError(f"{lang} block stuck at {len(blocks[lang])}/"
                               f"{CORPUS_QUOTA[lang]} after {rounds} filler rounds")

    # flatten into language blocks (stable id ranges) and resolve gold indices
    corpus: list[Passage] = []
    prov: list[dict] = []
    for lang in MIRACL_LANGS:
        start = len(corpus)
        corpus.extend(blocks[lang])
        prov.append({"kind": "lang-block", "lang": lang, "id_start": f"p{start:04d}",
                     "id_end": f"p{len(corpus) - 1:04d}", "count": len(blocks[lang]),
                     "note": f"quota {CORPUS_QUOTA[lang]} incl. "
                             f"{MIRACL_FILLER_QUOTA[lang]} cross-domain filler"})
    index_of = {p.text: i for i, p in enumerate(corpus)}

    queries: list[Query] = []
    for lang in MIRACL_LANGS:
        for (qtext, golds, _neg) in selected[lang]:
            refs = sorted({index_of[g] for g in golds if g in index_of})
            if not refs:
                raise RuntimeError(f"no resolvable gold for {qtext[:60]!r}")
            queries.append(Query(qtext, lang, SRC_TAG[lang], refs))
            for gi in refs:
                corpus[gi].gold_for.append(len(queries) - 1)

    # E) exact duplicates (negative control: engine must not silently drop them)
    anchored = [i for i, p in enumerate(corpus) if p.gold_for]
    dup_pairs: list[tuple[int, int]] = []
    for src_i in sorted(rng.shuffled(anchored)[:N_DUP_ENTRIES]):
        idx = len(corpus)
        corpus.append(Passage(corpus[src_i].text, corpus[src_i].lang, "duplicate",
                              dup_of=src_i))
        dup_pairs.append((src_i, idx))
        prov.append({"kind": "duplicate", "lang": corpus[-1].lang,
                     "id_start": f"p{idx:04d}", "id_end": f"p{idx:04d}", "count": 1,
                     "note": f"exact copy of p{src_i:04d} (plan §4.3 negative control)"})
    if len(corpus) != CORPUS_N:
        raise RuntimeError(f"corpus {len(corpus)} != {CORPUS_N}")

    # F) gibberish candidate pool (negative control); GIBBERISH_N of them are frozen
    #    after scoring — see select_gibberish().
    for g in GIBBERISH_CANDIDATES:
        queries.append(Query(g, "xx", "gibberish-candidate"))
    expect_n = QUERY_N - GIBBERISH_N + len(GIBBERISH_CANDIDATES)
    if len(queries) != expect_n:
        raise RuntimeError(f"queries {len(queries)} != {expect_n}")

    src_counts: dict[str, list[int]] = {}
    for i, p in enumerate(corpus):
        src_counts.setdefault(p.src, []).append(i)
    for src, ids in sorted(src_counts.items()):
        prov.append({"kind": "source", "lang": "-", "id_start": f"p{min(ids):04d}",
                     "id_end": f"p{max(ids):04d}", "count": len(ids),
                     "note": f"source {src}"})
    notes.append("duplicate pairs (gold -> copy): "
                 + ", ".join(f"p{a:04d} -> p{b:04d}" for a, b in dup_pairs))
    notes.append(f"gibberish negative controls: {len(GIBBERISH_CANDIDATES)} candidates "
                 f"encoded, the {GIBBERISH_N} with the lowest top dense score are frozen "
                 f"as q{QUERY_N - GIBBERISH_N:03d}..q{QUERY_N - 1:03d}")
    notes.append(f"gold coverage: {len(anchored)} distinct corpus rows are marked "
                 f"relevant by {sum(len(q.golds) for q in queries)} (query, gold) pairs")
    return Fixture(corpus, queries, prov, notes, dup_pairs)


# ------------------------------------------------------------------------ stage: encode
@dataclass
class Encoded:
    dense_q: np.ndarray
    dense_c: np.ndarray
    sparse_q: list[dict[int, float]]
    sparse_c: list[dict[int, float]]
    tok_len_q: list[int]
    tok_len_c: list[int]
    st_check: dict


def encode_texts(texts: list[str], model, tok, w_sp: np.ndarray, b_sp: np.ndarray,
                 tag: str, batch: int):
    """fp32 CPU bge-m3 forward: dense (normalized CLS) + per-token lexical weights.

    Texts are length-bucketed before batching (padding waste is the dominant CPU cost)
    and results are scattered back to the original row order, so the batch composition
    is a fixed function of the input list => deterministic across runs.
    """
    import torch
    n = len(texts)
    dense = np.zeros((n, 1024), dtype=np.float32)
    sparse: list[dict[int, float]] = [{} for _ in range(n)]
    tlens: list[int] = [0] * n

    t0 = time.time()
    lengths: list[int] = []
    for b0 in range(0, n, 512):
        enc = tok(texts[b0:b0 + 512], truncation=True, max_length=MAX_SEQ_LEN)
        lengths.extend(len(ids) for ids in enc["input_ids"])
    total_tokens = int(sum(lengths))
    order = sorted(range(n), key=lambda i: (lengths[i], i))
    log(f"  {tag}: {n} texts, {total_tokens:,} tokens "
        f"(mean {total_tokens / n:.0f}, max {max(lengths)}, "
        f"at limit {sum(1 for l in lengths if l >= MAX_SEQ_LEN)}), "
        f"tokenize {time.time() - t0:.1f}s")

    t1 = time.time()
    pad_tokens = 0
    real_tokens = 0
    done = 0
    for b0 in range(0, n, batch):
        idxs = order[b0:b0 + batch]
        enc = tok([texts[i] for i in idxs], padding=True, truncation=True,
                  max_length=MAX_SEQ_LEN, return_tensors="pt")
        ids = enc["input_ids"].to(torch.int64)
        att = enc["attention_mask"].to(torch.int64)
        with torch.no_grad():
            h = model(input_ids=ids, attention_mask=att).last_hidden_state
        h = h.to(torch.float32).numpy()
        ids_np, att_np = ids.numpy(), att.numpy()
        pad_tokens += int(ids.shape[1]) * len(idxs)
        real_tokens += sum(lengths[i] for i in idxs)
        cls = h[:, 0, :]
        norms = np.maximum(np.linalg.norm(cls, axis=1, keepdims=True), 1e-12)
        unit = (cls / norms).astype(np.float32)
        lex = np.maximum(h @ w_sp.T + b_sp, 0.0).reshape(h.shape[0], -1)
        for r, i in enumerate(idxs):
            dense[i] = unit[r]
            ln = int(att_np[r].sum())
            tlens[i] = ln
            d = sparse[i]
            for t_id, w in zip(ids_np[r][:ln].tolist(), lex[r][:ln].tolist()):
                if t_id <= 2 or w <= 0.0:
                    continue                 # frozen convention: token id > 2
                if w > d.get(t_id, 0.0):
                    d[t_id] = float(w)
        done += len(idxs)
        if (b0 // batch) % 8 == 0 or done == n:
            el = time.time() - t1
            log(f"    {tag}: {done}/{n} in {el:7.1f}s "
                f"({done / max(el, 1e-6):5.1f} texts/s, "
                f"pad waste {100.0 * (1.0 - real_tokens / max(pad_tokens, 1)):.1f}%)")
    return dense, sparse, tlens

def sparse_dot_matrix(sparse_q: list[dict[int, float]], sparse_c: list[dict[int, float]]
                      ) -> np.ndarray:
    """dot over shared token ids, accumulated in ascending token-id order.

    Both sides are flattened and then sorted globally by (token id, doc index), so the
    merge is a two-pointer walk and every (query, passage) cell receives its
    contributions in ascending token-id order: a fixed summation order, hence bitwise
    reproducible.
    """
    nq, nc = len(sparse_q), len(sparse_c)

    def flatten(maps):
        ids: list[int] = []
        rows: list[int] = []
        vals: list[float] = []
        for i, d in enumerate(maps):
            for t in sorted(d):
                ids.append(t)
                rows.append(i)
                vals.append(float(d[t]))
        return (np.asarray(ids, dtype=np.int64), np.asarray(rows, dtype=np.int64),
                np.asarray(vals, dtype=np.float32))

    d_ids, d_rows, d_vals = flatten(sparse_q)
    p_ids, p_rows, p_vals = flatten(sparse_c)
    qo = np.lexsort((d_rows, d_ids))
    co = np.lexsort((p_rows, p_ids))
    qi, qr, qv = d_ids[qo], d_rows[qo], d_vals[qo]
    ci, cr, cv = p_ids[co], p_rows[co], p_vals[co]
    qg = np.flatnonzero(np.concatenate(([True], qi[1:] != qi[:-1])))
    cg = np.flatnonzero(np.concatenate(([True], ci[1:] != ci[:-1])))
    qe = np.append(qg[1:], len(qi))
    ce = np.append(cg[1:], len(ci))
    flat = np.zeros(nq * nc, dtype=np.float32)
    a = b = shared = 0
    while a < len(qg) and b < len(cg):
        t1, t2 = qi[qg[a]], ci[cg[b]]
        if t1 < t2:
            a += 1
        elif t2 < t1:
            b += 1
        else:
            qs, qend = qg[a], qe[a]
            cs, cend = cg[b], ce[b]
            idx = (qr[qs:qend][:, None] * nc + cr[cs:cend][None, :]).ravel()
            addv = (qv[qs:qend][:, None] * cv[cs:cend][None, :]).ravel()
            np.add.at(flat, idx, addv)
            shared += 1
            a += 1
            b += 1
    log(f"  sparse: query entries {len(qi)}, passage entries {len(ci)}, shared token ids "
        f"{shared}, non-zero cells {int((flat != 0.0).sum())} of {nq * nc}")
    return flat.reshape(nq, nc)


def sparse_brute_check(sparse_q, sparse_c, sparse: np.ndarray,
                       n_pairs: int = 4000) -> tuple[float, float, int]:
    """Independent brute-force recompute of a sample of sparse scores.

    Accumulates in float32 and in ascending token-id order like the vectorized path, so
    the comparison is exact-arithmetic-equivalent; the residual is fp32 rounding only
    (the two implementations still differ in how partial sums are grouped).
    """
    rng = DetRand(hashlib.sha256(b"sparse-check").digest())
    nq, nc = len(sparse_q), len(sparse_c)
    worst_abs = 0.0
    worst_rel = 0.0
    n_fail = 0
    for _ in range(n_pairs):
        i, j = rng.below(nq), rng.below(nc)
        a, b = sparse_q[i], sparse_c[j]
        s = np.float32(0.0)
        for t, w in sorted(a.items()):
            v = b.get(t)
            if v is not None:
                s = np.float32(s + np.float32(w) * np.float32(v))
        ref = float(sparse[i, j])
        d = abs(float(s) - ref)
        if d > worst_abs:
            worst_abs = d
        if abs(ref) > 1.0:
            worst_rel = max(worst_rel, d / abs(ref))
        if d > 1e-5 + 1e-4 * abs(ref):       # fp32-appropriate mixed tolerance
            n_fail += 1
    return worst_abs, worst_rel, n_fail

def cross_check_st(ct: list[str], qt: list[str], dense_c, dense_q, tok_lens_c, tok_lens_q,
                   batch: int) -> dict:
    """sentence-transformers is the reference implementation of the dense head."""
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(str(MODEL_DIR), device="cpu")
    try:
        st[0].max_seq_length = MAX_SEQ_LEN
    except Exception as e:                                   # pragma: no cover
        log(f"  WARN: could not pin ST max_seq_length: {e}")
    res: dict = {"performed": True, "note": "subset with token length < 512 (no truncation)"}
    sub_c = [i for i, t in enumerate(tok_lens_c) if t < MAX_SEQ_LEN][:32]
    sub_q = [i for i, t in enumerate(tok_lens_q) if t < MAX_SEQ_LEN][:32]
    for name, idxs, mine in (("corpus", sub_c, dense_c), ("queries", sub_q, dense_q)):
        texts = [(ct if name == "corpus" else qt)[i] for i in idxs]
        ref = st.encode(texts, batch_size=min(8, batch), normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        got = mine[idxs]
        diff = float(np.max(np.abs(ref - got)))
        cos = float(np.mean([float(np.dot(x, y)) for x, y in zip(ref, got)]))
        res[name] = {"n": len(idxs), "max_abs_diff": diff, "mean_cosine": cos}
        log(f"  ST cross-check {name}: n={len(idxs)} max|diff|={diff:.3e} mean cos={cos:.9f}")
    return res


def onnx_cross_check(fx: Fixture, enc: Encoded, n_corpus: int = 8,
                     n_query: int = 6) -> dict:
    """Cross-check the PyTorch oracle against the ONNX export (plan §4.1 item 3).

    A deterministic sample of corpus rows and query rows is re-encoded with
    onnxruntime (onnx/model.onnx: token_embeddings + sentence_embedding) and the lexical
    head is rebuilt from the ONNX token embeddings exactly as in ../gen_goldens.py.
    Dense vectors, sparse vectors and the sampled (query, passage) scores are compared
    against the PyTorch values that become the frozen matrices.
    """
    import onnxruntime as ort
    import torch
    from tokenizers import Tokenizer
    log("onnx cross-check: loading onnxruntime session (onnx/model.onnx)")
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(MODEL_DIR / "onnx" / "model.onnx"), so,
                                providers=["CPUExecutionProvider"])
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    tok.enable_truncation(max_length=MAX_SEQ_LEN)
    ck = torch.load(MODEL_DIR / "sparse_linear.pt", map_location="cpu", weights_only=True)
    w_sp = ck["weight"].to(torch.float32).numpy()
    b_sp = ck["bias"].to(torch.float32).numpy()

    def encode(texts: list[str]):
        ids_all = [tok.encode(t).ids for t in texts]
        width = max(len(x) for x in ids_all)
        pad = np.zeros((len(ids_all), width), dtype=np.int64)
        att = np.zeros((len(ids_all), width), dtype=np.int64)
        for r, x in enumerate(ids_all):
            pad[r, :len(x)] = x
            att[r, :len(x)] = 1
        te, se = sess.run(None, {"input_ids": pad, "attention_mask": att})
        dense_out, sparse_out = [], []
        for r, x in enumerate(ids_all):
            vec = se[r].astype(np.float32)
            dense_out.append((vec / max(float(np.linalg.norm(vec)), 1e-12)).astype(np.float32))
            h = te[r, :len(x), :]
            lex = np.maximum(h @ w_sp.T + b_sp, 0.0).reshape(-1)
            d: dict[int, float] = {}
            for t_id, w in zip(x, lex.tolist()):
                if t_id <= 2 or w <= 0.0:
                    continue
                if w > d.get(t_id, 0.0):
                    d[t_id] = w
            sparse_out.append(d)
        return dense_out, sparse_out

    cidx = [int(i) for i in np.linspace(0, len(fx.corpus) - 1, n_corpus).astype(int)]
    qidx = [int(i) for i in np.linspace(0, len(fx.queries) - 1, n_query).astype(int)]
    ct = [fx.corpus[i].text for i in cidx]
    qt = [fx.queries[i].text for i in qidx]
    ref_c, ref_c_sp = encode(ct)
    ref_q, ref_q_sp = encode(qt)

    dense_diff = 0.0
    dense_cos = 1.0
    for idxs, ref, mine in ((cidx, ref_c, enc.dense_c), (qidx, ref_q, enc.dense_q)):
        for k, i in enumerate(idxs):
            a, b = ref[k], mine[i]
            dense_diff = max(dense_diff, float(np.max(np.abs(a - b))))
            dense_cos = min(dense_cos, float(np.dot(a, b)))
    sp_worst_abs, sp_worst_rel, sp_pairs, sp_zero = 0.0, 0.0, 0, 0
    for a, i in enumerate(qidx):
        for b, j in enumerate(cidx):
            s_onnx = sum(w * ref_c_sp[b].get(t, 0.0)
                         for t, w in ref_q_sp[a].items() if t in ref_c_sp[b])
            s_torch = float(sum(w * enc.sparse_c[j].get(t, 0.0)
                               for t, w in enc.sparse_q[i].items()))
            sp_pairs += 1
            d = abs(float(s_onnx) - s_torch)
            sp_worst_abs = max(sp_worst_abs, d)
            if abs(s_torch) > 1e-6:
                sp_worst_rel = max(sp_worst_rel, d / abs(s_torch))
            else:
                sp_zero += 1
    res = {"performed": True, "onnx_model": "onnx/model.onnx (opset 11)",
           "sample": {"corpus_rows": cidx, "query_rows": qidx},
           "dense_max_abs_diff": dense_diff, "dense_min_cosine": dense_cos,
           "sparse_pairs": sp_pairs, "sparse_zero_score_pairs": sp_zero,
           "sparse_max_abs_diff": sp_worst_abs, "sparse_max_rel_diff": sp_worst_rel,
           "gate": "dense cosine >= 0.9999 and sparse relative diff <= 1e-3 (plan §1.2)"}
    if dense_cos < 0.9999 or sp_worst_rel > 1e-3:
        raise RuntimeError(f"ONNX/PyTorch cross-check failed: {res}")
    log(f"  onnx cross-check ok: dense max|diff|={dense_diff:.3e} min cos={dense_cos:.9f}, "
        f"sparse {sp_pairs} pairs max|diff|={sp_worst_abs:.3e} max rel={sp_worst_rel:.3e}")
    return res


def plan_invariance_check(texts_q: list[str], texts_c: list[str], model, tok,
                          w_sp: np.ndarray, b_sp: np.ndarray, enc: Encoded,
                          n_query: int = 24, n_passage: int = 96) -> dict:
    """How much does the fp32 CPU result depend on the batching plan?

    A length-stratified probe set is encoded three times: batch 64, batch 17 (different
    padding shapes) and batch 64 again. The second pair must agree *bitwise* (fp32 CPU
    kernels are deterministic for a fixed plan); the first pair bounds the cross-plan
    divergence a caller may see, which is what justifies the |Δscore| < 1e-5 tie window of
    plan §4.3 — the engine does not have to reproduce our padding geometry.
    """
    ci = [int(x) for x in np.linspace(0, len(texts_c) - 1, n_passage).astype(int)]
    qi = [int(x) for x in np.linspace(0, len(texts_q) - 1, n_query).astype(int)]
    sub_c = [texts_c[i] for i in ci]
    sub_q = [texts_q[i] for i in qi]
    froz_c = np.stack([enc.dense_c[i] for i in ci])
    froz_q = np.stack([enc.dense_q[i] for i in qi])

    def run(batch: int, tag: str):
        d_c, s_c, _ = encode_texts(sub_c, model, tok, w_sp, b_sp, f"{tag}/probe-corpus", batch)
        d_q, s_q, _ = encode_texts(sub_q, model, tok, w_sp, b_sp, f"{tag}/probe-query", batch)
        return d_c, s_c, d_q, s_q

    a_c, a_cs, a_q, a_qs = run(64, "probe64")
    b_c, b_cs, b_q, b_qs = run(17, "probe17")
    c_c, c_cs, c_q, c_qs = run(64, "probe64-recheck")
    bitwise = bool(np.array_equal(a_c, c_c) and np.array_equal(a_q, c_q)
                   and a_cs == c_cs and a_qs == c_qs)
    cross_vec = max(float(np.max(np.abs(a_c - b_c))), float(np.max(np.abs(a_q - b_q))))
    vs_frozen_vec = max(float(np.max(np.abs(a_c - froz_c))), float(np.max(np.abs(a_q - froz_q))))
    scores_a = (a_q @ enc.dense_c.T).astype(np.float32)
    scores_b = (b_q @ enc.dense_c.T).astype(np.float32)
    scores_f = (froz_q @ enc.dense_c.T).astype(np.float32)
    cross_score = float(max(np.max(np.abs(scores_a - scores_b)), np.max(np.abs(scores_a - scores_f))))
    min_cos = min(float(np.min(np.sum(a_c * froz_c, axis=1))), float(np.min(np.sum(a_q * froz_q, axis=1))))
    key_diffs = sum(1 for k, i in enumerate(qi) for t in enc.sparse_q[i] if t not in a_qs[k])
    key_diffs += sum(1 for k, i in enumerate(ci) for t in enc.sparse_c[i] if t not in a_cs[k])
    res = {"performed": True, "probed_passages": len(ci), "probed_queries": len(qi),
           "batches": [64, 17, 64],
           "same_plan_bitwise_identical": bitwise,
           "cross_plan_max_abs_embedding_diff": cross_vec,
           "probe_vs_frozen_max_abs_embedding_diff": vs_frozen_vec,
           "probe_vs_frozen_min_cosine": min_cos,
           "cross_plan_max_abs_score_diff": cross_score,
           "sparse_key_set_differences": key_diffs,
           "tie_window": TIE_EPS}
    res["verdict"] = ("plan-sensitive noise stays far below the §4.3 tie window"
                      if cross_score < TIE_EPS and bitwise else
                      "check flags: bitwise=" + str(bitwise) +
                      " cross_score=" + format(cross_score, ".3e"))
    log(f"  plan invariance: same-plan bitwise={bitwise}, cross-plan max|Δemb|={cross_vec:.3e}, "
        f"probe-vs-frozen max|Δemb|={vs_frozen_vec:.3e} min cos={min_cos:.9f}, "
        f"cross-plan max|Δscore|={cross_score:.3e} (tie window {TIE_EPS}), "
        f"sparse key diffs={key_diffs}")
    assert bitwise, "fp32 CPU encoding is not reproducible for a fixed batching plan"
    return res


def encode_all(fx: Fixture, batch: int, do_st: bool, no_probe: bool = False
                 ) -> Encoded:
    import torch
    torch.set_num_threads(TORCH_THREADS)
    torch.set_grad_enabled(False)
    from transformers import AutoModel, AutoTokenizer
    log(f"loading PyTorch bge-m3 fp32 CPU (threads={TORCH_THREADS}) from {MODEL_DIR}")
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModel.from_pretrained(str(MODEL_DIR), dtype=torch.float32)
    model.eval()
    ck = torch.load(MODEL_DIR / "sparse_linear.pt", map_location="cpu", weights_only=True)
    w_sp = ck["weight"].to(torch.float32).numpy()
    b_sp = ck["bias"].to(torch.float32).numpy()
    log(f"sparse_linear W{w_sp.shape} b{b_sp.shape} (fp16 stored, fp32 compute)")
    ct = [p.text for p in fx.corpus]
    qt = [q.text for q in fx.queries]
    dense_c, sparse_c, tl_c = encode_texts(ct, model, tok, w_sp, b_sp, "corpus", batch)
    dense_q, sparse_q, tl_q = encode_texts(qt, model, tok, w_sp, b_sp, "queries", batch)
    enc = Encoded(dense_q, dense_c, sparse_q, sparse_c, tl_q, tl_c, {})
    invariant = {"performed": False}
    if not no_probe:
        invariant = plan_invariance_check(qt, ct, model, tok, w_sp, b_sp, enc)
    del model
    import gc
    gc.collect()
    st_check = {"performed": False}
    if do_st:
        st_check = cross_check_st(ct, qt, dense_c, dense_q, tl_c, tl_q, batch)
    enc.st_check = {"plan_invariance": invariant, "sentence_transformers": st_check}
    return enc


# ------------------------------------------------------------------------ stage: outputs
def topk_with_ties(scores: np.ndarray, k: int, eps: float):
    order = np.lexsort((np.arange(scores.shape[0]), -scores))
    top = [int(i) for i in order[:k]]
    groups: list[list[int]] = []
    run = [top[0]]
    for x, y in zip(top[:-1], top[1:]):
        if abs(float(scores[x]) - float(scores[y])) < eps:
            run.append(y)
        else:
            if len(run) > 1:
                groups.append(run)
            run = [y]
    if len(run) > 1:
        groups.append(run)
    return top, groups


def mat_stats(m: np.ndarray) -> dict:
    flat = m.ravel()
    return {"shape": [int(m.shape[0]), int(m.shape[1])],
            "dtype": "float32-le", "layout": "row-major [query][passage]",
            "min": float(flat.min()), "max": float(flat.max()),
            "mean": float(flat.mean()), "std": float(flat.std()),
            "all_zero_rows": int(np.count_nonzero(np.abs(m).sum(axis=1) == 0.0)),
            "all_zero_cols": int(np.count_nonzero(np.abs(m).sum(axis=0) == 0.0)),
            "n_negative": int((flat < 0).sum())}


LF = chr(10)   # jsonl line terminator, kept byte-exact


def write_jsonl(fx: Fixture) -> tuple[list[str], list[str]]:
    """corpus.jsonl + queries.jsonl; returns the id lists (index == row order)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cids = [f"p{i:04d}" for i in range(len(fx.corpus))]
    qids = [f"q{i:03d}" for i in range(len(fx.queries))]
    with open(OUT_DIR / "corpus.jsonl", "wb") as fh:
        for p, cid in zip(fx.corpus, cids):
            fh.write((json.dumps({"id": cid, "text": p.text}, ensure_ascii=False,
                                  sort_keys=True) + LF).encode("utf-8"))
    with open(OUT_DIR / "queries.jsonl", "wb") as fh:
        for q, qid in zip(fx.queries, qids):
            fh.write((json.dumps({"id": qid, "text": q.text}, ensure_ascii=False,
                                  sort_keys=True) + LF).encode("utf-8"))
    return cids, qids


def select_gibberish(fx: Fixture, enc: Encoded) -> tuple[Fixture, Encoded, dict]:
    """Freeze the GIBBERISH_N lowest-scoring gibberish candidates (plan §4.3 control)."""
    xx = [i for i, q in enumerate(fx.queries) if q.lang == "xx"]
    if len(xx) != len(GIBBERISH_CANDIDATES):
        raise RuntimeError(f"expected {len(GIBBERISH_CANDIDATES)} gibberish candidates")
    top = {i: float((enc.dense_q[i] @ enc.dense_c.T).max()) for i in xx}
    ranked = sorted(xx, key=lambda i: (top[i], i))
    chosen = sorted(ranked[:GIBBERISH_N])
    keep = [i for i, q in enumerate(fx.queries) if q.lang != "xx"] + chosen
    rows = []
    for i in keep:
        q = fx.queries[i]
        rows.append(Query(q.text, q.lang, "synthetic-gibberish" if q.lang == "xx"
                          else q.src, list(q.golds)))
    cand_info = [{"candidate_index": xx.index(k), "text": fx.queries[k].text,
                  "max_dense": top[k], "frozen": k in chosen} for k in xx]
    best = min(c["max_dense"] for c in cand_info if c["frozen"])
    fx2 = Fixture(fx.corpus, rows, fx.provenance, fx.notes,
                  fx.dup_pairs)
    enc2 = Encoded(enc.dense_q[keep], enc.dense_c,
                   [enc.sparse_q[i] for i in keep], enc.sparse_c,
                   [enc.tok_len_q[i] for i in keep], enc.tok_len_c, enc.st_check)
    return fx2, enc2, {"n_candidates": len(xx), "frozen": GIBBERISH_N,
                       "rule": "the GIBBERISH_N candidates with the lowest top dense "
                               "score against the frozen corpus",
                       "lowest_candidate_max_dense": best, "candidates": cand_info}


def write_goldens(fx: Fixture, enc: Encoded, batch: int,
                  cids: list[str], qids: list[str],
                  extra_meta: dict | None = None) -> dict:
    """freeze the two score matrices and build the qrels.json payload."""
    dense = (enc.dense_q @ enc.dense_c.T).astype(np.float32)
    sparse = sparse_dot_matrix(enc.sparse_q, enc.sparse_c)
    assert dense.shape == (QUERY_N, CORPUS_N) and sparse.shape == dense.shape
    assert np.isfinite(dense).all() and np.isfinite(sparse).all()
    sp_abs, sp_rel, sp_fail = sparse_brute_check(enc.sparse_q, enc.sparse_c, sparse)
    log(f"  sparse brute-force spot check (4000 pairs): max |diff| = {sp_abs:.3e}, "
        f"max rel = {sp_rel:.3e}, outside tolerance = {sp_fail}; magnitude "
        f"min={sparse.min():.3f} median={float(np.median(sparse)):.3f} max={sparse.max():.3f}")
    assert sp_fail == 0, (f"sparse accumulation disagrees with brute force: {sp_fail}/4000 "
                          f"sampled pairs outside 1e-5 + 1e-4*|score| (max abs {sp_abs:.3e}, "
                          f"max rel {sp_rel:.3e})")
    nnz_rows = int((np.abs(sparse).sum(axis=1) > 0).sum())
    assert nnz_rows >= len(fx.queries) - GIBBERISH_N, (
        f"only {nnz_rows} query rows carry a non-zero sparse score")
    nz = int((sparse != 0.0).sum())
    assert nz > 0.001 * sparse.size, (
        f"sparse matrix looks degenerate: {nz} non-zero cells "
        f"({100.0 * nz / sparse.size:.4f}% of {sparse.size})")

    dense.astype("<f4").tofile(OUT_DIR / "golden_dense_scores.fp32.bin")
    sparse.astype("<f4").tofile(OUT_DIR / "golden_sparse_scores.fp32.bin")

    recs = []
    tie_stats = {"dense_groups": 0, "sparse_groups": 0, "dense_pairs": 0, "sparse_pairs": 0}
    for i, q in enumerate(fx.queries):
        d_top, d_grp = topk_with_ties(dense[i], TOP_K, TIE_EPS)
        s_top, s_grp = topk_with_ties(sparse[i], TOP_K, TIE_EPS)
        tie_stats["dense_groups"] += len(d_grp)
        tie_stats["sparse_groups"] += len(s_grp)
        tie_stats["dense_pairs"] += sum(len(g) * (len(g) - 1) // 2 for g in d_grp)
        tie_stats["sparse_pairs"] += sum(len(g) * (len(g) - 1) // 2 for g in s_grp)
        recs.append({
            "query_id": qids[i], "query_lang": q.lang, "query_src": q.src,
            "gold_ids": [cids[g] for g in q.golds],
            "dense_top10": [cids[j] for j in d_top],
            "dense_top10_scores": [float(dense[i, j]) for j in d_top],
            "dense_tie_groups": [[cids[j] for j in g] for g in d_grp],
            "sparse_top10": [cids[j] for j in s_top],
            "sparse_top10_scores": [float(sparse[i, j]) for j in s_top],
            "sparse_tie_groups": [[cids[j] for j in g] for g in s_grp],
        })

    # ---- negative controls / sanity ----
    med = float(np.median(dense))
    gib = [{"query_id": qids[i], "max_dense": float(dense[i].max()),
            "median_dense_all": med,
            "max_dense_below_median": bool(dense[i].max() < med),
            "max_sparse": float(sparse[i].max()),
            "sparse_nonzero": bool(np.abs(sparse[i]).sum() > 0.0)}
           for i, q in enumerate(fx.queries) if q.lang == "xx"]
    top1_real = np.array([dense[i].max() for i, q in enumerate(fx.queries)
                          if q.lang != "xx"], dtype=np.float32)
    p25 = float(np.percentile(top1_real, 25))
    for g in gib:
        g["below_real_query_top1_p25"] = bool(g["max_dense"] < p25)
    gib_criterion = {
        "plan_4_3_literal": "max dense score < median of the whole golden matrix",
        "literal_holds": bool(all(g["max_dense_below_median"] for g in gib)),
        "measured": {"median_dense_all": med,
                     "gibberish_max_dense": [g["max_dense"] for g in gib],
                     "real_query_top1_p25": p25,
                     "real_query_top1_median": float(np.median(top1_real))},
        "note": ("the reference model does not satisfy the literal §4.3 bound on real "
                 "multilingual data: bge-m3 scores unrelated passage pairs below a noise "
                 "string's best match, so the whole-matrix median is smaller than the "
                 "gibberish max. Operationalise the control as (a) the engine's gibberish "
                 "row must match these frozen golden values and (b) it must stay below the "
                 "real-query top-1 score distribution recorded here."),
    }

    dup_report = []
    for src_i, dup_i in fx.dup_pairs:
        same_d = bool(np.array_equal(dense[:, src_i], dense[:, dup_i]))
        same_s = bool(np.array_equal(sparse[:, src_i], sparse[:, dup_i]))
        appear_adjacent = 0
        appear_one_only = 0
        for r in recs:
            top = r["dense_top10"]
            has_a, has_b = cids[src_i] in top, cids[dup_i] in top
            if has_a and has_b:
                pa, pb = top.index(cids[src_i]), top.index(cids[dup_i])
                appear_adjacent += int(abs(pa - pb) == 1)
            elif has_a or has_b:
                appear_one_only += 1
        dup_report.append({"pair": [cids[src_i], cids[dup_i]],
                           "identical_dense_column": same_d,
                           "identical_sparse_column": same_s,
                           "top10_both_adjacent": appear_adjacent,
                           "top10_one_only": appear_one_only})

    gram = (enc.dense_c @ enc.dense_c.T).astype(np.float32)
    dup_mask = np.zeros_like(gram, dtype=bool)
    for a, b in fx.dup_pairs:
        dup_mask[a, b] = dup_mask[b, a] = True
    off_diag = (~np.eye(CORPUS_N, dtype=bool)) & (~dup_mask)
    near_dup = {"pairs_cos_gt_0.95": int((gram[off_diag] > 0.95).sum() // 2),
                "pairs_cos_gt_0.90": int((gram[off_diag] > 0.90).sum() // 2),
                "max_cos_excl_dups": float(gram[off_diag].max()),
                "mean_cos_excl_dups": float(gram[off_diag].mean())}

    anchored = [r for r in recs if r["gold_ids"]]
    hits1 = sum(1 for r in anchored if r["dense_top10"][0] in set(r["gold_ids"]))
    hits10 = sum(1 for r in anchored if set(r["gold_ids"]) & set(r["dense_top10"]))
    hits_sparse10 = sum(1 for r in anchored if set(r["gold_ids"]) & set(r["sparse_top10"]))
    mrr10 = float(np.mean([max((1.0 / (k + 1) for k, c in enumerate(r["dense_top10"])
                                    if c in set(r["gold_ids"])), default=0.0)
                           for r in anchored]))
    ndcg10 = float(np.mean([
        sum(1.0 / np.log2(k + 2) for k, c in enumerate(r["dense_top10"])
            if c in set(r["gold_ids"]))
        / sum(1.0 / np.log2(k + 2) for k in range(min(len(r["gold_ids"]), TOP_K)))
        for r in anchored]))

    tok_stats = {
        "corpus": {"min": int(min(enc.tok_len_c)), "max": int(max(enc.tok_len_c)),
                   "mean": round(float(np.mean(enc.tok_len_c)), 1),
                   "at_limit": int(sum(1 for t in enc.tok_len_c if t >= MAX_SEQ_LEN))},
        "queries": {"min": int(min(enc.tok_len_q)), "max": int(max(enc.tok_len_q)),
                    "mean": round(float(np.mean(enc.tok_len_q)), 1),
                    "at_limit": int(sum(1 for t in enc.tok_len_q if t >= MAX_SEQ_LEN))}}
    nnz = {"query_mean": round(float(np.mean([len(d) for d in enc.sparse_q])), 1),
           "query_min": int(min(len(d) for d in enc.sparse_q)),
           "passage_mean": round(float(np.mean([len(d) for d in enc.sparse_c])), 1),
           "passage_min": int(min(len(d) for d in enc.sparse_c))}

    meta = {
        "generator": "scripts/embedding/gen_retrieval_fixtures.py",
        "plan_sections": ["§1.1", "§4.1", "§4.3"],
        "seed": SEED, "encode_batch": batch, "torch_threads": TORCH_THREADS,
        "max_seq_len": MAX_SEQ_LEN, "top_k": TOP_K, "tie_eps": TIE_EPS,
        "model": "bge-m3 (XLM-RoBERTa, 24L/1024H/16heads), PyTorch fp32 CPU oracle",
        "definitions": {
            "corpus_line": '{"id":"p0000".."p4999","text":...} UTF-8 LF, whitespace '
                           'collapsed, no control chars',
            "query_line": '{"id":"q000".."q199","text":...}',
            "matrix": "float32 little-endian, row-major [query][passage], no header; "
                      "row i is queries.jsonl line i (zero-padded id qNNN) and "
                      "column j is corpus.jsonl line j (zero-padded id pNNNN)",
            "dense": "last_hidden_state CLS row, L2-normalized (fp32); score = dot = cosine",
            "sparse": "lexical weight per token = relu(h @ W_sparse.T + b_sparse), "
                      "W_sparse/b_sparse from sparse_linear.pt (fp16 values, fp32 math); "
                      "kept for token ids > 2 (../golden_sparse.json convention), zero "
                      "weights dropped, max-pooled per token id; score = sum of "
                      "w_q*w_p over shared token ids, ascending token-id accumulation",
            "ranking": "sort by (-score, passage index ascending); groups of top-10 "
                       "entries within |Δscore| < 1e-5 are reported unordered (plan §4.3)",
            "hybrid": "not frozen: derivable as dense + alpha*sparse (bge-m3 default "
                      "alpha 0.5)",
            "truncation": f"tokenizer truncation at {MAX_SEQ_LEN} tokens",
            "query_src": "provenance tag per query row: nfcorpus / miracl-zh-triples / "
                         "mmarco-ja-triples / miracl-ko-triples for the 198 eval-set "
                         "queries, synthetic-gibberish for the q198/q199 negative "
                         "controls; corpus rows carry the same tags plus "
                         "miracl-corpus-<lang> (cross-domain filler) and duplicate",
            "gold_ids": "passages the source dataset's human judgments mark relevant for "
                        "that query (1..5 of them are in the corpus; nfcorpus queries "
                        "with more judgments keep only their first 5 — see notes)",
        },
        "sources": {k: REVISIONS[k] for k in sorted(REVISIONS)},
        "expected_rows": {f"{d}|{c}|{s}": v for (d, c, s), v in sorted(EXPECTED_ROWS.items())},
        "counts": {"corpus": len(fx.corpus), "queries": len(fx.queries),
                   "corpus_per_lang": dict(CORPUS_QUOTA), "queries_per_lang": {
                       **{k: QUERY_QUOTA[k] for k in QUERY_QUOTA},
                       "xx_gibberish": GIBBERISH_N,
                       "xx_gibberish_candidates": len(GIBBERISH_CANDIDATES)}},
        "provenance": fx.provenance,
        "notes": fx.notes,
        "dense_matrix": mat_stats(dense), "sparse_matrix": mat_stats(sparse),
        "token_lengths": tok_stats, "sparse_nnz": nnz,
        "tie_stats": tie_stats,
        "sparse_brute_force": {"pairs": 4000, "max_abs_diff": sp_abs,
                               "max_rel_diff": sp_rel, "pairs_outside_tolerance": sp_fail},
        "sparse_signal": {"nonzero_entries": int((sparse != 0.0).sum()),
                          "query_rows_with_signal": nnz_rows,
                          "passage_cols_with_signal": int((np.abs(sparse).sum(axis=0) > 0).sum())},
        "negative_controls": {"gibberish_queries": gib,
                              "gibberish_criterion": gib_criterion,
                              "duplicate_passages": dup_report,
                              "near_duplicates": near_dup},
        "human_qrels_agreement": {"queries_with_human_gold": len(anchored),
                                  "dense_top1_hit": hits1, "dense_top10_hit": hits10,
                                  "sparse_top10_hit": hits_sparse10,
                                  "dense_mrr10": round(mrr10, 6),
                                  "dense_ndcg10_avg": round(ndcg10, 6)},
        "sentence_transformers_check": enc.st_check["sentence_transformers"],
        "plan_invariance_check": enc.st_check["plan_invariance"],
    }
    if extra_meta:
        meta.update(extra_meta)
    payload = {"meta": meta, "queries": recs}
    return payload


def collect_hashes(names: list[str]) -> dict:
    size, sha = {}, {}
    for f in names:
        p = OUT_DIR / f
        if not p.exists():
            continue
        size[f] = p.stat().st_size
        sha[f] = sha256_file(p)
    return {"size": size, "sha256": sha}


def model_hashes() -> dict:
    return {f: sha256_file(MODEL_DIR / f) for f in MODEL_FILES if (MODEL_DIR / f).exists()}


def write_qrels(payload: dict) -> None:
    (OUT_DIR / "qrels.json").write_bytes(
        (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").
        encode("utf-8"))


def write_readme(payload: dict, hashes: dict, mhash: dict, versions: dict,
                 parent_meta: dict) -> str:
    m = payload["meta"]
    d, s = m["dense_matrix"], m["sparse_matrix"]
    nc = m["negative_controls"]
    lines: list[str] = []
    A = lines.append
    A("# L3 retrieval fixtures — `test_embed_retrieval`")
    A("")
    A("Frozen corpus/query/ranking fixtures for the bge-m3 embedding engine retrieval "
      "test (embedding_model_test_plan.md §4.3), generated by the dev-only script "
      "`scripts/embedding/gen_retrieval_fixtures.py` (PyTorch fp32 CPU oracle — never the "
      "engine). These references are frozen: they are never regenerated to make a test "
      "pass (plan §1.1). The sentence-level goldens in the parent directory "
      "(`corpus.txt`, `golden_*.bin/json`, `golden_meta.json`, `MANIFEST.json`) are "
      "untouched by this generator.")
    A("")
    A("## Files and sha256 manifest")
    A("")
    A("| file | bytes | sha256 |")
    A("|---|---:|---|")
    for f in DATA_FILES:
        A(f"| `{f}` | {hashes['size'][f]:,} | `{hashes['sha256'][f]}` |")
    A("| `README.md` | (self) | printed to stdout by the generator |")
    A("")
    A("Regenerate everything - the data stage takes ~4 min cold (~165 pinned ""datasets-server page requests) or seconds from the download cache; then ""~5,200 fp32 CPU encodes plus the SentenceTransformer and ONNX cross-checks. ""Observed end-to-end: 11 min on a 24-thread desktop CPU with a warm cache.")
    A("")
    A("```bash")
    A("python scripts/embedding/gen_retrieval_fixtures.py")
    A("```")
    A("")
    A("Replay the download cache without network (encodes still run):")
    A("")
    A("```bash")
    A("python scripts/embedding/gen_retrieval_fixtures.py --offline")
    A("python scripts/embedding/gen_retrieval_fixtures.py --stage data    # jsonl only")
    A("python scripts/embedding/gen_retrieval_fixtures.py --verify-only   # re-hash + diff")
    A("```")
    A("")
    A("## Layout")
    A("")
    A(f"* `corpus.jsonl` — {m['counts']['corpus']} lines, one JSON object per line: "
      f"`{{\"id\": \"p0000\"..\"p4999\", \"text\": \"…\"}}`, UTF-8, LF-terminated. "
      f"Whitespace runs collapsed to one space, no control characters, so readers can "
      f"use the raw text verbatim.")
    A(f"* `queries.jsonl` — {m['counts']['queries']} lines, same shape (`id` runs `q000`..`q199`).")
    A(f"* `golden_dense_scores.fp32.bin` and `golden_sparse_scores.fp32.bin` — raw "
        f"float32 little-endian, row-major **[{d['shape'][0]}][{d['shape'][1]}]** = "
        f"[query][passage], no header, {d['shape'][0] * d['shape'][1] * 4:,} bytes each. "
        f"Row *i* is line *i* of `queries.jsonl` (id `q000`, `q001`, …) and column *j* "
        f"is line *j* of `corpus.jsonl` (id `p0000`, `p0001`, …).")
    A("* Line endings are LF, files are UTF-8 without BOM; committed sha256 values are of those exact bytes, so a checkout with `core.autocrlf=true` (CRLF) changes the on-disk hash while the content stays equivalent — read the files as text, not as raw hashes, if your git config converts line endings.")
    A("* `qrels.json` — `meta` = this manifest (definitions, dataset revisions, stats, "
      "file hashes); `queries` = per query: `dense_top10` / `sparse_top10` (ordered "
      "golden rankings), the matching scores, the tie groups "
      f"(|Δscore| < {m['tie_eps']}, compared unordered by the test), and `gold_ids` — "
      "the *human* relevant passages inherited from the source datasets (advisory: "
      "recall/MRR/NDCG baseline inputs).")
    A("")
    A("`corpus_embeddings.fp32.bin` (the engine-built corpus embedding cache of plan §4.3) "
      "is deliberately **not** shipped here — the test produces and uses it at review time.")
    A("")
    A("## Provenance")
    A("")
    A("Public eval sets fetched through the Hugging Face datasets-server `/rows` API, "
      "`revision` pinned to these dataset commits, row counts asserted at runtime:")
    A("")
    A("| dataset | pinned revision | role |")
    A("|---|---|---|")
    for k in sorted(REVISIONS):
        A(f"| `{k}` | `{REVISIONS[k]}` | {ROLE[k]} |")
    A("")
    A("Corpus blocks (language-grouped, deterministic id ranges):")
    A("")
    A("| kind | lang | ids | count | note |")
    A("|---|---|---|---:|---|")
    for r in m["provenance"]:
        if r["kind"] in ("lang-block", "duplicate"):
            A(f"| {r['kind']} | {r['lang']} | {r['id_start']}..{r['id_end']} | "
              f"{r['count']} | {r.get('note', '')} |")
    A("")
    A("Passages by source tag:")
    A("")
    A("| source tag | first id | last id | count |")
    A("|---|---|---|---:|")
    for r in m["provenance"]:
        if r["kind"] == "source":
            A(f"| `{r['note'][len('source '):]}` | {r['id_start']} | {r['id_end']} | "
              f"{r['count']} |")
    A("")
    for n in m["notes"]:
        A(f"* {n}")
    A("")
    A(f"* Every query keeps its source dataset's human-judged relevant passages in the "
      f"corpus (1–{MAX_GOLDS_PER_QUERY} each, `gold_ids`); `q198`/`q199` are synthetic "
      f"gibberish negative controls, and 3 corpus rows are exact duplicates of anchored "
      f"golds (both required by plan §4.3).")
    A("* Cross-lingual note: queries are monolingual in EN/ZH/JA/KO and their gold "
      f"passages are in the same language; the corpus additionally contains "
      f"cross-domain Wikipedia filler, so unrelated topically-similar passages act as "
      f"distractors.")
    A("")
    A("## Reference definitions")
    A("")
    for k, v in sorted(m["definitions"].items()):
        A(f"* **{k}**: {v}")
    A("")
    A("## Score-matrix sanity stats (reference model)")
    A("")
    A("| matrix | min | max | mean | std | zero rows | zero cols | negatives |")
    A("|---|---|---|---|---|---|---|---|")
    A(f"| dense | {d['min']:.6f} | {d['max']:.6f} | {d['mean']:.6f} | {d['std']:.6f} | "
      f"{d['all_zero_rows']} | {d['all_zero_cols']} | {d['n_negative']} |")
    A(f"| sparse | {s['min']:.6f} | {s['max']:.6f} | {s['mean']:.6f} | {s['std']:.6f} | "
      f"{s['all_zero_rows']} | {s['all_zero_cols']} | {s['n_negative']} |")
    A("")
    A(f"* token lengths after {m['max_seq_len']}-token truncation: corpus "
      f"{m['token_lengths']['corpus']}, queries {m['token_lengths']['queries']}.")
    A(f"* sparse vector sizes (distinct token ids per doc): {m['sparse_nnz']}.")
    A(f"* tie groups inside the golden top-10 (window {m['tie_eps']}): {m['tie_stats']}.")
    sb = m["sparse_brute_force"]
    A(f"* sparse accumulation spot-check vs a float32 brute-force recompute "
      f"({sb['pairs']} random pairs): max |diff| = {sb['max_abs_diff']:.3e}, max relative = "
      f"{sb['max_rel_diff']:.3e}, pairs outside 1e-5 + 1e-4*|score| = "
      f"{sb['pairs_outside_tolerance']}. Sparse signal: {m['sparse_signal']}.")
    A(f"* near-duplicate report over the corpus (dense cosine, deliberate duplicates "
      f"excluded): {nc['near_duplicates']}.")
    A(f"* human-qrels agreement of the frozen ranking (fixture realism, not a gate): "
      f"{m['human_qrels_agreement']}.")
    gsel = m.get("gibberish_selection", {})
    A(f"* negative control — gibberish queries ({gsel.get('rule', 'synthetic noise')}): "
      f"{gsel.get('n_candidates', '-')} candidates were encoded and "
      f"{gsel.get('frozen', '-')} frozen into `q198`/`q199`; lowest candidate "
      f"max_dense = {gsel.get('lowest_candidate_max_dense', float('nan')):.6f}. "
      f"Frozen rows: {nc['gibberish_queries']}.")
    A("* negative control — gibberish criterion: "
      f"{json.dumps(nc['gibberish_criterion'], ensure_ascii=False)}")
    A(f"* negative control — duplicate passages: {nc['duplicate_passages']}. Duplicate "
      "rows keep identical score columns and are adjacent in the full ranking (the "
      "ascending-index tie-break decides which one is first); at the top-10 cut only one "
      "member of a tied pair can appear, which is the documented tie-break, not a dedup.")
    if "onnx_cross_check" in m:
        oc = m["onnx_cross_check"]
        A(f"* ONNX-export cross-check of both heads (plan §1.1/§4.1: two independent "
          f"implementations must agree before freezing): dense min cosine "
          f"{oc['dense_min_cosine']:.9f} (>= 0.9999), sparse "
          f"{oc['sparse_pairs']} sampled pairs max rel diff "
          f"{oc['sparse_max_rel_diff']:.2e} (<= 1e-3); sample rows "
          f"{oc['sample']}.")
    pi = m["plan_invariance_check"]
    A(f"* determinism probe (length-stratified subset re-encoded with batch 64, batch 17, "
      f"then batch 64 again): the two same-plan passes are bitwise identical = "
      f"{pi['same_plan_bitwise_identical']}; cross-plan max|Δembedding| = "
      f"{pi['cross_plan_max_abs_embedding_diff']:.3e}, probe-vs-frozen max|Δembedding| = "
      f"{pi['probe_vs_frozen_max_abs_embedding_diff']:.3e} (min cosine "
      f"{pi['probe_vs_frozen_min_cosine']:.9f}), cross-plan max|Δscore| = "
      f"{pi['cross_plan_max_abs_score_diff']:.3e} vs the {pi['tie_window']} tie window of "
      f"plan §4.3 — {pi['verdict']}. Sparse key sets differed in "
      f"{pi['sparse_key_set_differences']} of "
      f"{pi['probed_queries'] + pi['probed_passages']} probed docs.")
    A(f"* dense head cross-check against `sentence-transformers` "
    f"(independent reference implementation): {m['sentence_transformers_check']}.")
    A("")
    A("## Model inputs (read-only; must match ../golden_meta.json)")
    A("")
    A("| model file | sha256 |")
    A("|---|---|")
    for k, v in sorted(mhash.items()):
        A(f"| `{k}` | `{v}` |")
    A("")
    A(f"Cross-check vs the frozen parent manifest `golden_meta.json` "
      f"(`sha256` block): {parent_meta}")
    A("")
    A(f"Generator environment: {json.dumps(versions, sort_keys=True)}")
    A("")
    A("## How the test consumes this (plan §4.3)")
    A("")
    A("1. Build the corpus library through the engine (`encode_batch` over `corpus.jsonl`) "
      "with **truncation at 512 tokens** (`meta.max_seq_len`, the same knob as "
      "`../golden_meta.json`); any other truncation changes the reference. Cache the result "
      "as `corpus_embeddings.fp32.bin`.")
    A("2. Encode the 200 queries, brute-force cosine top-k on the host.")
    A("3. Ranking parity: engine ordered top-10 == `dense_top10` for ≥ 99 % of queries, "
      "tie groups (`dense_tie_groups`, |Δscore| < 1e-5) compared unordered and reported; "
      "then repeat for `sparse_top10` and the dense+sparse hybrid (`dense + α·sparse` with "
      "the engine's α) — both score matrices are frozen here, so hybrid rankings are "
      "recomputable from them without re-encoding.")
    A("4. Metrics: `Recall@10 == 1.0` modulo ties against the golden qrels; report "
      "MRR@10 / NDCG@10. `gold_ids` are the *source datasets' human judgments* (advisory "
      "baseline — the frozen rankings themselves are model-derived from the golden "
      "matrices); the reference MRR@10 / NDCG@10 of this fixture are the values recorded "
      "above.")
    A("5. Sparse semantics the engine must match: one lexical weight per token id "
      "(id > 2, i.e. `<s>`/`<pad>`/`</s>` excluded and `<unk>` kept — the "
      "`../golden_sparse.json` convention), relu output, zero weights dropped, max-pooled "
      "per token id inside a document, then the dot over shared token ids.")
    A("5. Negative controls: gibberish queries (`q198`, `q199`) must stay below the "
      "median golden dense score; duplicate pairs must rank adjacent.")
    A("")
    A("## Determinism")
    A("")
    A(f"Sampling uses a SHA-256 counter PRNG (seed {m['seed']}), all table iterations are "
      "sorted or index-addressed, and JSON is written with `sort_keys=True`. Encoding is "
      f"fp32 CPU torch with `torch.set_num_threads({m['torch_threads']})` and a fixed "
      "batch order/padding, so a re-run on the same machine and torch build reproduces "
      "these sha256 values byte-for-byte (`--verify-only`). On other hardware the last fp32 "
      "bits can differ; the committed hashes above stay the frozen reference and ranking "
      "parity relies on the tie rule rather than bitwise equality.")
    A("")
    text = "\n".join(lines)
    (OUT_DIR / "README.md").write_bytes(text.encode("utf-8"))
    return sha256_file(OUT_DIR / "README.md")


ROLE = {
    "mteb/nfcorpus": "EN passages (CORDIS biomedical), EN queries, human qrels (dev)",
    "Hieuman/zh_MIRACL": "ZH queries + positive (gold) + negative passages (MIRACL zh style)",
    "Hieuman/ko_MIRACL": "KO queries + positive (gold) + negative passages (MIRACL ko style)",
    "crystina-z/mmarco-passage": "JA queries + positive (gold) + negative passages "
                                 "(mMARCO japanese)",
    "miracl/miracl-corpus": "cross-domain filler passages (Wikipedia), deterministic "
                            "random windows per language en/zh/ja/ko",
}


# ------------------------------------------------------------------------------- main


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="generate L3 retrieval fixtures (dev-only)")
    ap.add_argument("--stage", choices=["data", "all"], default="all")
    ap.add_argument("--offline", action="store_true", help="replay the HTTP cache only")
    ap.add_argument("--cache-dir", default=os.environ.get(
        "EMBED_FIXTURE_CACHE",
        str(Path(tempfile.gettempdir()) / "bge_m3_retrieval_fixture_cache")))
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--no-st-check", action="store_true")
    ap.add_argument("--no-plan-probe", action="store_true",
                    help="skip the batching-plan invariance probe")
    ap.add_argument("--no-onnx-check", action="store_true",
                    help="skip the onnxruntime cross-check of both heads")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args(argv)

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    log(f"cache dir: {cache}")

    if args.verify_only:
        payload = json.loads((OUT_DIR / "qrels.json").read_text(encoding="utf-8"))
        expect = payload["meta"].get("file_sha256", {})
        got = collect_hashes([f for f in DATA_FILES if f != "qrels.json"])
        bad = [f for f, h in got["sha256"].items() if expect.get(f) != h]
        for f, h in sorted(got["sha256"].items()):
            log(f"  {f:32s} {h} {'OK' if f not in bad else 'MISMATCH'}")
        log("verify: " + ("byte-identical to manifest" if not bad else f"DIFFERS: {bad}"))
        return 0 if not bad else 1

    t0 = time.time()
    http = Http(cache_dir=cache, offline=args.offline)
    fx = build_fixture(http)
    log(f"data stage: {len(fx.corpus)} passages, {len(fx.queries)} queries; "
        f"http calls={http.calls}, cache hits={http.hits}; {time.time() - t0:.1f}s")
    for n in fx.notes:
        log(f"  note: {n}")

    if args.stage == "data":
        dbg = Fixture(fx.corpus, fx.queries[: QUERY_N - GIBBERISH_N] + [
            Query(t, "xx", "gibberish-candidate") for t in GIBBERISH_CANDIDATES[:GIBBERISH_N]],
            fx.provenance, fx.notes, fx.dup_pairs)
        cids, qids = write_jsonl(dbg)
        log(f"stage=data: wrote {OUT_DIR}/corpus.jsonl ({len(cids)} rows) + "
            f"queries.jsonl ({len(qids)} rows, provisional gibberish block); "
            f"score matrices, qrels.json and README.md not rewritten")
        return 0

    enc = encode_all(fx, args.batch, not args.no_st_check, args.no_plan_probe)
    fx, enc, gib_info = select_gibberish(fx, enc)
    log(f"gibberish selection: frozen max_dense={gib_info['lowest_candidate_max_dense']:.6f} "
        f"of {gib_info['n_candidates']} candidates")
    checks: dict = {"gibberish_selection": gib_info}
    if not args.no_onnx_check:
        checks["onnx_cross_check"] = onnx_cross_check(fx, enc)
    cids, qids = write_jsonl(fx)
    log(f"wrote {OUT_DIR}/corpus.jsonl ({len(cids)} rows) + queries.jsonl ({len(qids)} rows)")
    payload = write_goldens(fx, enc, args.batch, cids, qids, checks)

    # hashes of the 4 data files go into qrels.json meta; README carries all 5
    data_hashes = collect_hashes([f for f in DATA_FILES if f != "qrels.json"])
    payload["meta"]["file_sha256"] = data_hashes["sha256"]
    payload["meta"]["file_sizes"] = data_hashes["size"]
    mhash = model_hashes()
    payload["meta"]["model_sha256"] = mhash
    versions = {"python": sys.version.split()[0], "numpy": np.__version__}
    for name in ("torch", "transformers", "sentence_transformers", "tokenizers"):
        try:
            versions[name] = getattr(__import__(name), "__version__", "?")
        except Exception:
            versions[name] = "n/a"
    payload["meta"]["generator_versions"] = versions
    parent_cross = {}
    try:
        pm = json.loads((OUT_DIR.parent / "golden_meta.json").read_text(encoding="utf-8"))
        for k, v in pm.get("sha256", {}).items():
            base = Path(k).name
            if base in mhash:
                parent_cross[base] = "match" if mhash[base] == v else "MISMATCH"
    except Exception as e:                                   # pragma: no cover
        parent_cross = {"error": str(e)}
    payload["meta"]["parent_manifest_crosscheck"] = parent_cross
    write_qrels(payload)

    all_hashes = collect_hashes(DATA_FILES)
    readme_sha = write_readme(payload, all_hashes, mhash, versions, parent_cross)

    m = payload["meta"]
    log("")
    log("outputs:")
    for f in DATA_FILES:
        log(f"  {f:32s} {all_hashes['size'][f]:>12,} B  {all_hashes['sha256'][f]}")
    log(f"  {'README.md':32s} {(OUT_DIR / 'README.md').stat().st_size:>12,} B  {readme_sha}")
    log(f"dense : shape={m['dense_matrix']['shape']} min={m['dense_matrix']['min']:.6f} "
        f"max={m['dense_matrix']['max']:.6f} mean={m['dense_matrix']['mean']:.6f} "
        f"std={m['dense_matrix']['std']:.6f}")
    log(f"sparse: shape={m['sparse_matrix']['shape']} min={m['sparse_matrix']['min']:.6f} "
        f"max={m['sparse_matrix']['max']:.6f} mean={m['sparse_matrix']['mean']:.6f}")
    log(f"ties={m['tie_stats']}")
    log(f"human qrels agreement={m['human_qrels_agreement']}")
    log(f"gibberish={m['negative_controls']['gibberish_queries']}")
    log(f"duplicates={m['negative_controls']['duplicate_passages']}")
    log(f"near_dup={m['negative_controls']['near_duplicates']}")
    log(f"parent manifest cross-check={parent_cross}")
    log(f"total {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
