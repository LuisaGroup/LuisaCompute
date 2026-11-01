# `example_embed_retrieval_lib` — C-FFI retrieval steps

`example_embed_retrieval` drives the bge-m3 retriever through an interactive
console or a JSON-RPC server. This directory turns the same steps into a shared
library with a flat C ABI, so the retriever can be hosted by **any** language or
process through its FFI (Python `ctypes`/`cffi`, C# `DllImport`, Rust `bindgen`,
Java JNA, Go `cgo`, Node `ffi-napi`, ...). No C++ type, STL container or
LuisaCompute handle crosses the boundary: every entry point takes and returns C
scalars, UTF-8 strings and plain C structs only.

| File | Content |
|---|---|
| `example_embed_retrieval_lib.h` | the public C header: every entry point, the value types, the ownership rules and the error contract |
| `example_embed_retrieval_lib.cpp` | the implementation (the only TU that touches engine types) |
| `../embed/` | the engine: session, scheduler, tokenizer, retrieval index/service (chunked documents, snapshots, filters/MMR, ColBERT, stats), disk cache |
| `../../../../src/tests/unit/embedding/test_embed_retrieval_api.cpp` | host-only conformance test that drives every exported step |

The build target is `example_embed_retrieval_lib` (`kind = shared`); it lands next
to the other binaries, e.g. `bin/release/example_embed_retrieval_lib.dll` on
Windows or `bin/release/libexample_embed_retrieval_lib.so` elsewhere.

```bash
xmake build example_embed_retrieval_lib    # the shared library
xmake build test_embed_retrieval_api       # the C-ABI test
xmake run   test_embed_retrieval_api       # 159 asserts, no GPU/model needed
```

## The steps

One call = one console command. Every step that produces data returns an
`eer_status` and fills a caller-allocated payload struct; the payload must be
released with its matching `_free` function, on success **and** on failure.

| Step | Console equivalent | Entry point |
|---|---|---|
| initialize | process start-up (`--model`, `--tokenizer`, `--backend`) | `eer_context_create()` |
| shutdown | `quit` / `exit` | `eer_context_destroy()` |
| load a directory | `load <directory> [<extension-list>]` | `eer_load_dir()` |
| rank documents | `query [--top <N>] <query text>` | `eer_query()` |
| re-index one path | `update <file-or-subdirectory>` | `eer_update_path()` |
| drop one root | `unload <loaded-root-directory>` | `eer_unload_dir()` |
| encode one text | (internal to encode paths) | `eer_encode_text()` |
| load, chunked | `load` with passage splitting | `eer_load_dir_ex()` (+ `eer_chunk_options`, `eer_progress`) |
| add/update/remove a document | (new; no console equivalent) | `eer_add_document()`, `eer_update_document()`, `eer_remove_document()` |
| query with ranking controls | `query` with weights/filter/MMR | `eer_query_ex()` (+ `eer_query_options`) |
| encode a batch | (new) | `eer_encode_batch()` |
| encode ColBERT | (new) | `eer_encode_colbert()` |
| snapshot save/load | (new) | `eer_index_save()`, `eer_index_load()` |
| statistics | `help` diagnostics | `eer_stats()` |
| readiness probe | (new) | `eer_context_ping()` |
| device memory | (new) | `eer_context_device_memory()` |
| park / resume | (new) | `eer_context_park()`, `eer_context_resume()` |
| introspection | `help` diagnostics | `eer_version()`, `eer_status_string()`, `eer_sub_code_string()`, `eer_default_extensions()`, `eer_context_backend()`, `eer_context_disk_cache_enabled()` |

### `eer_context_create` / `eer_context_destroy`

```c
eer_status eer_context_create(const eer_create_info *info,
                              eer_context **out_context,
                              eer_error *out_error);
void       eer_context_destroy(eer_context *context);
```

* Zero the `eer_create_info`, set `struct_size = sizeof(eer_create_info)`, then
  set only the fields you need. Strings are UTF-8 and are copied.
* **Engine mode** (`info.encode == NULL`): `model_path` and `tokenizer_path` are
    required regular files, `backend` names the plugin (`"vk"` when omitted, plus
    `"dx"`, `"cuda"`, `"fallback"`, `"simd"`, `"metal"`), and `num_workers` is the
    scheduler worker count (`0` → hardware concurrency: the CPU core/thread count). Creation loads the weights, compiles the
    kernels and — unless `EER_CREATE_SKIP_WARMUP` is set — performs one warm-up
    encode, exactly like the executable. Engine mode is **hybrid**: every document
    and query is encoded with `EncodeOptions{max_seq_len = 512, want_sparse =
    true}`, so one forward pass yields the dense CLS vector plus the BGE-M3 sparse
    lexical weights, and the index scores
    `0.7 * dense_cosine + 0.3 * sparse_cosine`. The warm-up encode stays
    dense-only.
* **Callback mode** (`info.encode != NULL`): a host-supplied dense encoder
    replaces the model session. No model, no tokenizer, no backend, no GPU. This is
    how the library is tested and how embedders that already produce vectors reuse
    the index/ranking/disk cache. Callback mode stays **dense-only**: the encoder
    returns one dense vector per text, ranking is the plain dense dot product, and
    the disk cache records no sparse sidecar — byte-identical to the pre-hybrid
    behavior. The two modes are selected structurally (which `RetrievalService`
    constructor the library builds), so a callback can never be treated as a
    hybrid encoder with an empty sparse half.
* `program_path` points at the directory holding the backend modules
  (`luisa-vk.dll`, ...). `NULL`/`""` resolves to the current executable, which is
  what a plain executable would pass as `argv[0]`; a host that loads the library
  from another directory should pass the module directory explicitly.
* Configuration mistakes are reported *before* any device work
  (`EER_STATUS_INVALID_ARGUMENT` / `EER_STATUS_NOT_FOUND` with a message). A
  corrupt model still terminates the process inside the engine's own invariant
  checks (`LUISA_ERROR`), exactly as the executable does — validate the model in
  the host if that matters.
* Creation is expensive; create one context per process. `eer_context_destroy`
  drains the scheduler and releases the device; passing `NULL` is a no-op.

### `eer_load_dir` — `load <directory> [<extension-list>]`

```c
eer_status eer_load_dir(eer_context *context, const char *root,
                        const char *extensions, eer_load_result *result);
```

Recursively indexes `root`. `extensions` is a comma-separated rule
(`".txt,.md"`, `"TXT,.Md"`), case-insensitive, normalized to a leading dot and
lowercase; `NULL`/`""` selects `EER_DEFAULT_EXTENSIONS` (`.txt,.md`). Malformed
rules (wildcards, separators, empty entries) are rejected before the directory is
walked.

`result` reports `added` (newly encoded documents), `cached` (embeddings served
from `.lc_embed_cache*` instead of the encoder), `skipped` (regular files that do
not match the rule), `failed` (read/encode errors; the successfully indexed
siblings are kept), `total_documents`, the canonical `root` and the normalized
`extensions` rule that now governs it.

The cache layout is `<root>/.lc_embed_cache.json` (per-file content hash, size,
mtime, dimension, and `sparse_count` for hybrid entries),
`<root>/.lc_embed_cache/<md5>.bin` (the dense embedding) and — hybrid entries only
— `<root>/.lc_embed_cache/<md5>.sparse.bin` (the canonical sparse
`(uint32 token_id, float weight)` pairs, 8 bytes each). In engine mode a hit needs
both halves, so a directory indexed by an older dense-only build (no `sparse_count`,
no sidecar) reports `cached = 0` on its first hybrid `load` while it is re-encoded
and the sidecars are written. In callback mode no sidecar is ever written and the
layout is exactly the pre-hybrid one.

A root owns its rule and cannot be loaded twice or nested inside another loaded
root; `unload` it first to change the rule. The root is canonicalized, so a path
that does not exist or is not a directory is `EER_STATUS_INVALID_ARGUMENT`.

### `eer_query` — `query [--top <N>] <query text>`

```c
eer_status eer_query(eer_context *context, const char *utf8_text,
                     size_t text_length, uint32_t top_k,
                     eer_query_result *result);
```

Ranks every indexed document against the query text. The text is given as
`(pointer, byte length)` and need not be NUL-terminated; `top_k == 0` selects
`EER_DEFAULT_TOP_K` (5) and values above `EER_MAX_TOP_K` (100) are rejected. Hits
are sorted by descending score, ties broken by ascending canonical path. The
`eer_hit::score` field is one JSON/C `double` in both modes; only its meaning
depends on the mode:

* engine mode (hybrid): `0.7 * dense_cosine + 0.3 * sparse_cosine`. Dense vectors
  are L2-normalized by the session, so their dot product is the cosine; the sparse
  weights are canonicalized (duplicate token ids merged by a double-precision sum,
  exact zeros dropped, sorted by token id, L2-normalized) so the sparse term is a
  cosine in `[0, 1]`. The fused score is therefore in `[-0.7, 1]`.
* callback mode (dense only): the unchanged dense dot product, a cosine in
  `[-1, 1]`.

Querying an empty corpus, or an empty/blank text, is a recoverable
`EER_STATUS_OPERATIONAL` / `EER_STATUS_INVALID_ARGUMENT` — never "match
everything".

### `eer_update_path` — `update <file-or-subdirectory>`

```c
eer_status eer_update_path(eer_context *context, const char *path,
                           eer_update_result *result);
```

Re-indexes one path with the extension rule of the root that owns it, and reports
`added` / `updated` / `removed` / `unchanged` / `cached` / `failed` plus
`total_documents`. Byte-identical files are `unchanged` and are not encoded
again. Updates are transactional per file: a read/encode failure keeps the
previous retrievable version and increments `failed`. A path outside every loaded
root is `EER_STATUS_NOT_FOUND`.

### `eer_unload_dir` — `unload <loaded-root-directory>`

```c
eer_status eer_unload_dir(eer_context *context, const char *root,
                          eer_unload_result *result);
```

Drops exactly the named loaded root and every document it owns (a descendant is
not accepted as an alias) and reports `removed_documents` / `total_documents`.
Unloading a root that is not loaded is `EER_STATUS_NOT_FOUND`.

### `eer_encode_text`

```c
eer_status eer_encode_text(eer_context *context, const char *utf8_text,
                           size_t text_length, float *embedding,
                           size_t embedding_capacity, size_t *embedding_dimension,
                           eer_error *out_error);
```

Encodes one text without touching the index. Pass `embedding == NULL` with
`embedding_capacity == 0` to query the dimension only. Useful to pre-compute
query vectors, to check the model dimension, or to verify a host encoder.

The output is always the dense vector only — the sparse lexical half is not
exported through this entry point, so the contract is unchanged by hybrid
retrieval. In engine mode the sparse head is not computed here (only the indexing
and query paths that need the fused score enable it); in callback mode this simply
forwards the host encoder.

### Host encoder (`callback mode`)

```c
typedef int32_t (*eer_encode_fn)(void *user_data, const char *utf8_text,
                                 size_t text_length, float *embedding,
                                 size_t embedding_capacity,
                                 size_t *embedding_dimension);
```

Return `EER_ENCODE_OK` after writing `*embedding_dimension` finite floats, or
`EER_ENCODE_BUFFER_TOO_SMALL` after storing the required size (the library
retries once, up to 16384 floats) — anything else counts as a failed encode. The
callback runs on the calling thread inside the index lock and must not re-enter
the same context. Embeddings must have one consistent dimension per index.

Callback mode is dense-only when only `encode` is supplied — that default is a hard
boundary rather than a policy: the library then builds the dense `RetrievalService`,
and its scores, cache files and `eer_encode_text` output are exactly what they were
before hybrid retrieval existed. Since ABI 1.2 a second, optional callback
(`eer_sparse_encode_fn` in `eer_create_info`, struct_size-gated) supplies the BGE-M3
sparse lexical pairs; when BOTH callbacks are set, callback mode builds the hybrid
service instead, so a GPU-free host stays hybrid-capable (scores fuse dense + sparse
exactly like engine mode, per-query weights apply through `eer_query_ex`, and the
same `.lc_embed_cache*` sidecars are written).


### `eer_load_dir_ex` — chunked load with progress and cancellation
```c
eer_status eer_load_dir_ex(eer_context *context, const char *root,
                           const char *extensions,
                           const eer_chunk_options *chunk_options,
                           eer_load_result *result,
                           const eer_progress *progress);
```
`load` with passage splitting (ABI 1.2, report.md §2/§5). `chunk_options == NULL`
or `chunk_by == 0` keeps the whole-file behavior of `eer_load_dir`; otherwise every
file is split into chunks of at most `chunk_size` bytes (never inside a UTF-8 code
point; `chunk_overlap` bytes repeat the previous tail) along paragraph (`1`),
sentence (`2`) or token/word (`3`) boundaries. The index stores one record per
chunk and every hit carries `doc_id`, `chunk_index`, `byte_offset`, `byte_length`:
the chunk text is exactly `source[byte_offset, byte_offset + byte_length)`.
`max_seq_len` (0 = 512) is the per-chunk encode truncation length in engine mode.
Chunked disk-cache entries key on (content hash, chunk_index, chunk params) in a
second meta file `.lc_embed_cache.chunks.json`; the per-file v1 cache stays
byte-identical.
`progress` is optional (`eer_progress`): `on_progress(user_data, done, total)` runs
on the calling thread (per-chunk totals for documents, per-file totals for loads),
and a non-zero `cancel_flag` stops between units: applied content stays, the
in-flight document is applied all-or-nothing, the load's root registration is
rolled back so a follow-up load resumes, and the result carries
`EER_SUB_CANCELLED` with a `cancelled` message plus `EER_STATUS_OPERATIONAL` (a
clean, recoverable partial state).
### `eer_add_document` / `eer_update_document` / `eer_remove_document`
```c
eer_status eer_add_document(eer_context *context, const char *doc_id,
                            const char *utf8_text, size_t text_length,
                            const char *metadata_json,
                            const eer_chunk_options *chunk_options,
                            eer_document_result *result,
                            const eer_progress *progress);
eer_status eer_remove_document(eer_context *context, const char *doc_id,
                               eer_document_result *result);
```
Index content that never exists as a file (database rows, chat history, web
pages). `doc_id` is caller-chosen and is the idempotency key: re-ingesting the
same doc_id replaces its chunks transactionally (all new chunks encode before the
old ones drop), and identical text + metadata + chunking is a no-op reported in
`chunks_unchanged`. `metadata_json` must be empty or a JSON object; it is kept
verbatim for `eer_query_ex` filters and snapshots. Memory-only documents have no
path; `doc_id` is the only identity a hit reports for them. `eer_update_document`
is the replace alias; `eer_remove_document` drops every chunk (unknown ids are
`EER_STATUS_NOT_FOUND`).
### `eer_query_ex` — ranking controls
```c
eer_status eer_query_ex(eer_context *context, const char *utf8_text,
                        size_t text_length, uint32_t top_k,
                        const eer_query_options *options,
                        eer_query_result *result);
```
`query` with controls (ABI 1.2, report.md §4); `options == NULL` reproduces
`eer_query` exactly.
* `dense_weight` / `sparse_weight`: NaN or negative selects the compiled-in
  defaults (0.7 / 0.3); any non-negative pair fuses as
  `score = w_d * dense + w_s * sparse` (dense mode ignores both).
* `filter_json`: a Mongo-style metadata predicate applied BEFORE top-k selection,
  so filtered pages never under-fill: top-level keys combine with implicit AND;
  values may be scalars (equality) or objects with `$eq/$gte/$lte/$gt/$lt` /
  `$in` (array); `$and`/`$or` take arrays of objects. Example:
  `{"tenant":"acme","year":{"$gte":2023}}`. Only documents ingested with
  `metadata_json` can match.
* `diversity` (0..1): MMR re-ranking over the fused scores — higher values
  suppress same-document chunk repetition (novelty term `d * (1 - max_cosine)`).
* `max_seq_len` (0 = 512): per-query encode truncation; with flags bit0 set,
  `eer_query_result::truncated` reports that the query text was cut.
  * `instruction_prefix`: prepended to the query text before tokenization (engine
    mode). Note: NULL/empty keeps the query raw — 1.2 deliberately does NOT
    auto-prepend the bge-m3 instruction, preserving 1.1 query encodings verbatim.
    For bge-m3 retrieval quality pass
    `Represent this sentence for searching relevant passages:` explicitly for short
    queries.
  * `want_colbert`: re-rank the dense/sparse shortlist by ColBERT MaxSim (token-level
    late interaction). Requires engine mode and the `EER_CREATE_WANT_COLBERT` create
    flag; the MaxSim value becomes the hit `score` and is reported in
    `colbert_score`. A record without a stored matrix (restored from a snapshot that
    predates ColBERT serialization) is not an error: it keeps its fused dense/sparse
    score for that entry instead of failing the query. `want_colbert` is the last
    struct field and is read only when `struct_size` covers it.
Every hit carries the fused `score` plus the `dense_score` / `sparse_score` /
`colbert_score` breakdown (sparse is 0 in dense mode; colbert is 0 unless
requested), the passage location (`doc_id`, `chunk_index`, `byte_offset`,
`byte_length`), and results order by descending score with ties broken by
ascending `(doc_id, chunk_index)`.
### `eer_encode_batch` / `eer_encode_colbert`
```c
eer_status eer_encode_batch(eer_context *context, const eer_text *texts,
                            size_t count, float *out_vectors, size_t capacity,
                            size_t *out_dimension, eer_error *out_error);
eer_status eer_encode_colbert(eer_context *context, const char *utf8_text,
                              size_t text_length, float *out, size_t capacity,
                              size_t *out_values, uint32_t *out_rows,
                              eer_error *out_error);
```
`eer_encode_batch` (report.md §5) amortizes ingestion: engine mode submits one
scheduler batch, callback mode loops the host encoder. Output is row-major
`[count][dimension]`; a uniform dimension is validated (mismatch is
`EER_STATUS_OPERATIONAL` / `EER_SUB_ENCODE`). `out_vectors == NULL` with
`capacity == 0` probes the dimension (texts are still encoded and validated).
`eer_encode_colbert` serves the bge-m3 ColBERT head: `out_rows` rows of H=1024
float32, per-row L2-normalized; engine mode only (the host callback has no ColBERT
head).
### `eer_index_save` / `eer_index_load` — corpus snapshots
```c
eer_status eer_index_save(eer_context *context, const char *path, eer_error *out_error);
eer_status eer_index_load(eer_context *context, const char *path, eer_error *out_error);
```
  O(1) cold start for an already-ingested corpus (report.md §3): `save` writes a JSON
  manifest at `path` (version, dimension, hybrid mode, roots with their chunk
  options, every chunk record with doc_id / metadata / byte range / sparse pairs
  inline, and the text of memory-only documents) plus a packed-float32 blob at
  `path + ".bin"` with the dense vectors in manifest order. When the corpus stores
  ColBERT matrices, a second packed blob `path + ".colbert.bin"` holds them
  (per-record row counts are in the manifest) and `load` restores them — but only
  into a context that created with `EER_CREATE_WANT_COLBERT`; the create flag's RAM
  bound applies to snapshots too. `load` restores roots, documents, metadata and
  embeddings with NO re-encode; the pair relocates as a unit and loads on any
  backend (the blob is host byte order). Version, dimension or hybrid-mode
  mismatches, blob-size mismatches (checked with overflow-safe arithmetic against
  corrupt manifests) and unreadable files are clean `EER_STATUS_OPERATIONAL`
  errors (`EER_SUB_IO` / `EER_SUB_INDEX`) that leave the current index untouched.
  The per-file `.lc_embed_cache*` layer is independent.
### `eer_stats` / `eer_context_ping` / `eer_context_device_memory` / park-resume
```c
eer_status eer_stats(eer_context *context, eer_stats_data *out_stats);
eer_status eer_context_ping(eer_context *context, eer_error *out_error);
eer_status eer_context_device_memory(eer_context *context,
                                     uint64_t *out_used_bytes,
                                     uint64_t *out_total_bytes);
eer_status eer_context_park(eer_context *context, eer_error *out_error);
eer_status eer_context_resume(eer_context *context, eer_error *out_error);
```
Production observability (report.md §6). `eer_stats` reports document/chunk counts,
the embedding dimension, the hybrid flag, the cache hit ratio and the mean encode
latency over the index lifetime. `eer_context_ping` runs one tiny encode through
the full pipeline for load-balancer readiness. `eer_context_device_memory` reports
device memory where the backend exposes it; LuisaCompute currently exposes no
memory-stats API, so every context honestly reports 0/0 with `EER_STATUS_OK`.
`eer_context_park` / `eer_context_resume` exist so hosts can code against them, but
ship as `EER_STATUS_OPERATIONAL` ("not supported in this build"): the Session
lifetime is tied to the device context member order documented on `eer_context`.

## Memory ownership

* Payload structs (`eer_load_result`, `eer_unload_result`, `eer_update_result`,
  `eer_query_result`, `eer_document_result`, `eer_error`) are caller-allocated,
  usually on the stack. The library fills them; strings and the hit array inside
  them are allocated with the library's own allocator (ABI 1.1) and belong to
  the caller afterwards.
* Release them with `eer_*_free()`, always, success or failure. The `_free`
  functions zero the struct and are idempotent. Since 1.1 the allocations live
  on the library's own heap: C `free()` / the host allocator is NOT guaranteed
  to work across the shared-library boundary — always go through the `_free`
  entry point.
* Only `eer_version()`, `eer_status_string()`, `eer_sub_code_string()` and
  `eer_default_extensions()` (process lifetime) and `eer_context_backend()`
  (context lifetime) return borrowed pointers.

## Error model

| Code | Meaning |
|---|---|
| `EER_STATUS_OK` | step completed, payload valid |
| `EER_STATUS_INVALID_ARGUMENT` | NULL/empty/invalid argument, malformed extension rule, `top_k` out of range, root that cannot be canonicalized, embedding buffer too small — the engine was not touched |
| `EER_STATUS_NOT_FOUND` | a required file is missing (model/tokenizer), the root is not loaded (`unload`), or the update path is outside every root |
| `EER_STATUS_OPERATIONAL` | well-formed request the engine could not complete: no backend, encoder failure, empty corpus, dimension mismatch, cache I/O |
| `EER_STATUS_INVALID_HANDLE` | the `eer_context *` is NULL (or, best effort, already destroyed) |

Every failure carries a human-readable `message`; `eer_status_string()` maps the
code to a stable name for logging. Since ABI 1.2 every payload also carries an
appended `int32_t sub_code` (`EER_SUB_NONE` on success): `device`, `model`, `io`,
`encode`, `index`, or `cancelled`, so orchestrators can retry/reload
appropriately without parsing messages; `eer_sub_code_string()` maps them.
A cancelled load/document step is NOT an error state: the result counters report
the work done so far, `sub_code` is `EER_SUB_CANCELLED`, and the return status is
`EER_STATUS_OPERATIONAL` by design (a clean, recoverable partial result).

## Threading

All steps (`load`, `query`, `update`, `unload`, `encode`) are safe to call
concurrently on one context: the engine serializes index mutation and GPU
batching internally, exactly like the RPC server does. `eer_context_create` /
`eer_context_destroy` are not thread-safe — no other call may be in flight while
they run.

## Runtime layout

The shared library must sit next to (or find through `program_path`) the
LuisaCompute runtime and the selected backend modules:

```text
bin/release/
  example_embed_retrieval_lib.dll
  luisa-runtime.dll  luisa-core.dll  luisa-dsl.dll  luisa-vstl.dll
  luisa-backend-vk.dll | luisa-backend-dx.dll | luisa-backend-cuda.dll | ...
```

In callback mode only the runtime DLLs are needed (no backend is loaded).

## C example

```c
#include <stdio.h>
#include <string.h>
#include "example_embed_retrieval_lib.h"

int main(void) {
    eer_create_info info;
    eer_context *ctx = NULL;
    eer_error error;

    memset(&info, 0, sizeof(info));
    info.struct_size = sizeof(info);
    info.model_path = "D:/models/bge-m3/model.bgm";
    info.tokenizer_path = "D:/models/bge-m3/tokenizer.json";
    info.backend = "vk";

    memset(&error, 0, sizeof(error));
    if (eer_context_create(&info, &ctx, &error) != EER_STATUS_OK) {
        fprintf(stderr, "init failed: %s\n", error.message);
        eer_error_free(&error);
        return 1;
    }

    eer_load_result load;
    memset(&load, 0, sizeof(load));
    if (eer_load_dir(ctx, "D:/docs", NULL, &load) == EER_STATUS_OK) {
        printf("indexed %zu document(s) from %s\n", load.total_documents, load.root);
    } else {
        printf("load failed: %s\n", load.message);
    }
    eer_load_result_free(&load);

    eer_query_result hits;
    memset(&hits, 0, sizeof(hits));
    if (eer_query(ctx, "how do I mount a texture", 24u, 5u, &hits) == EER_STATUS_OK) {
        size_t i;
        for (i = 0; i < hits.hit_count; i++) {
            printf("%zu %.6f %s\n", i + 1u, hits.hits[i].score, hits.hits[i].path);
        }
    }
    eer_query_result_free(&hits);

    eer_context_destroy(ctx);
    return 0;
}
```

`clang-cl /TC example.c example_embed_retrieval_lib.lib` (MSVC) or
`gcc example.c -L. -lexample_embed_retrieval_lib` (MinGW) — the header compiles
clean as C99/C11 and C++20 with `-Wall -Wextra -pedantic`.

## Python example (`ctypes`)

```python
import ctypes as C

lib = C.CDLL("bin/release/example_embed_retrieval_lib.dll")   # or libexample_embed_retrieval_lib.so
lib.eer_version.restype = C.c_char_p
print(lib.eer_version().decode())          # 1.0.0

class CreateInfo(C.Structure):
    _fields_ = [("struct_size", C.c_size_t),
                ("flags", C.c_uint32),
                ("program_path", C.c_char_p),
                ("backend", C.c_char_p),
                ("model_path", C.c_char_p),
                ("tokenizer_path", C.c_char_p),
                ("num_workers", C.c_uint32),
                ("encode", C.c_void_p),
                ("encode_user_data", C.c_void_p)]

class Hit(C.Structure):
    _fields_ = [("path", C.c_char_p), ("score", C.c_double)]

class QueryResult(C.Structure):
    _fields_ = [("message", C.c_char_p),
                ("hit_count", C.c_size_t),
                ("hits", C.POINTER(Hit))]

info = CreateInfo(struct_size=C.sizeof(CreateInfo),
                  model_path=b"D:/models/bge-m3/model.bgm",
                  tokenizer_path=b"D:/models/bge-m3/tokenizer.json",
                  backend=b"vk")
ctx = C.c_void_p()
class Error(C.Structure):                        # eer_error { char *message; }
    _fields_ = [("message", C.c_char_p)]

error = Error()
status = lib.eer_context_create(C.byref(info), C.byref(ctx), C.byref(error))
if status != 0:                                  # EER_STATUS_OK == 0
    raise RuntimeError(error.message.decode())

result = QueryResult()
text = "how do I mount a texture".encode()
lib.eer_query.argtypes = [C.c_void_p, C.c_char_p, C.c_size_t, C.c_uint32, C.POINTER(QueryResult)]
lib.eer_query(ctx, text, len(text), 5, C.byref(result))
for i in range(result.hit_count):
    print(result.hits[i].score, result.hits[i].path.decode())
lib.eer_query_result_free(C.byref(result))
lib.eer_context_destroy(ctx)
```

(The `eer_*_free` functions terminate the payloads that came from the library, so
Python never frees memory the DLL allocated.)

## Verification

`src/tests/unit/embedding/test_embed_retrieval_api.cpp` is a host-only
(`LUISA_EMBED_HOST_ONLY=1`) Boost.UT test registered as
`test_embed_retrieval_api` in both build systems. The library itself is a single
translation unit (`api/*.cpp`), so no unity-build batching is involved.
It drives the library in callback mode with deterministic encoders and checks:

* version / status-string / default-extension introspection,
* NULL-handle and NULL-argument rejection for every entry point, plus the
  idempotent `_free` functions,
* `eer_context_create` argument validation (struct size, missing model,
  missing tokenizer) without touching a GPU,
* the full `load → query → update → unload` cycle, including the documented
  counts, the ranking order/tie-break, `top_k` bounds, blank-query and
  empty-corpus errors, the duplicate/overlapping-root rejection and
  `eer_encode_text` dimension queries,
* the on-disk embedding cache shared by two consecutive contexts (second context
  is served from `.lc_embed_cache`, with identical ranking),
* per-file encoder-failure accounting, and the host-encoder buffer retry plus its
  16384-float cap,
* the ABI 1.2 RAG surface (report.md §2-§6): add/update/remove document
  round-trips with idempotent re-ingestion and paragraph/sentence/token chunk
  offsets; snapshot save/load across a context recreate plus the
  corrupt-snapshot safety; `eer_query_ex` weight overrides (deterministic
  reordering), metadata filters (`$gte`/`$and`/`$in`) applied pre-top-k, MMR
  diversity, the score-breakdown identity and malformed-filter rejection;
  `eer_encode_batch` parity with per-text encodes; load progress + mid-load
  cancellation with resumable partial state; the sparse-callback hybrid index;
  stats counters, ping, the honest 0/0 device-memory path and the park/resume
  unsupported degradation.

Because the test stays in callback mode, it also pins the promise that hybrid
retrieval did not change the dense path: the recorded rankings and exact scores of
a host encoder are the pre-hybrid ones. Hybrid fusion itself (weighted dense +
sparse cosine, sparse canonicalization, sidecar hit/miss/re-encode) is covered by
the host-only `test_embed_retrieval_index`, which drives the same engine code with
deterministic dense + sparse encoders.

For a run against a real model, use the executable
(`xmake run example_embed_retrieval --model ... --tokenizer ...`) or call
`eer_context_create` in engine mode from your host. Retrieval quality depends on
the corpus and on the package precision (fp32 > bf16/fp16 > int8 for both halves);
no fixed hit rate is promised.
