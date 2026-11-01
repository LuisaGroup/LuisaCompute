# Embedding retrieval example

`example_embed_retrieval` is an interactive, in-memory file retriever built on the
existing bge-m3 `Session`. It provides the same `load`, `unload`, `query`, and
`update` operations through a console REPL or newline-delimited JSON-RPC 2.0 over
TCP. The one-shot `example_embed_text` demo is a separate executable and is
unchanged.

## Start

```text
example_embed_retrieval --model D:/models/bge-m3/model.bgm \
    --tokenizer D:/models/bge-m3/tokenizer.json

example_embed_retrieval --model D:/models/bge-m3/model.bgm \
    --tokenizer D:/models/bge-m3/tokenizer.json --backend dx
```

The backend defaults to **`vk`** when `--backend` is omitted. Backend names are
the installed LuisaCompute plugin names; startup reports the installed names if
the selected plugin cannot be created. `--model` and `--tokenizer` are required.
Startup performs one dense warm-up encode to initialize and drain the execution
path before accepting commands; the warm-up stays dense-only on purpose, while
every later document and query encode also computes the sparse lexical head. Use
`--help` for the complete process command line.

Normal startup enters the console. Server mode replaces the console (the two
input loops never run together):

```text
example_embed_retrieval --model ... --tokenizer ... --server
example_embed_retrieval --model ... --tokenizer ... --server \
    --listen 127.0.0.1 --port 8765
```

The listener defaults to `127.0.0.1:8765`. Port `0` asks the OS for an ephemeral
port and is useful for automated tests; the selected port is logged.

## Console commands

```text
load <directory> [<extension-list>]
unload <loaded-root-directory>
query [--top <N>] <query text>
update <file-or-subdirectory>
help
quit
exit
```

- `load` recursively admits regular files matching a comma-separated extension
  rule. The default is `.txt,.md`. Rules are case-insensitive, normalized to a
  leading dot and lowercase (for example, `TXT,.Md` becomes `.md,.txt`). Empty
  entries, separators, wildcards, and malformed rules are rejected. `skipped`
  counts nonmatching regular files; unreadable/invalid files and encoding
  failures count as `failed` without discarding successfully indexed siblings.
- A loaded root owns its extension rule. Duplicate and overlapping/nested roots
  are rejected. To change a rule, `unload` the exact root and load it again.
- `query` defaults to five hits; `--top` accepts 1 through 100. Model-backed
  retrieval is hybrid: every document and every query carries a dense BGE-M3 CLS
  vector and a sparse BGE-M3 lexical weight vector, and the reported score is
  `0.7 * dense_cosine + 0.3 * sparse_cosine` (`kRetrievalDenseWeight` /
  `kRetrievalSparseWeight`). Dense vectors are L2-normalized by `Session`, so
  their dot product is the cosine. Sparse weights are canonicalized before use:
  duplicate token ids are merged by summing their weights, exact zeros are
  dropped, the pairs are sorted by token id, and the weights are L2-normalized
  in double precision — the sparse term is therefore a cosine in `[0, 1]` and the
  fused score is at most 1. Results sort by descending score and then ascending
  canonical path. Querying an empty corpus, including loaded roots with no
  documents, is a recoverable error.
- `update` accepts the loaded root, one descendant subdirectory, or one
  descendant file. It uses the owning root's stored extension rule. Byte-identical
  files are `unchanged` and are not encoded again. New files are added, changed
  files are updated, and deleted tracked files are removed. A subdirectory update
  does not touch siblings or other roots.
- Updates are transactional per file: a read/encode/embedding-validation failure
  retains the previous retrievable version. `failed` is reported in the mutation
  counts. Updating an untracked missing path or a path outside every root is an
  error.
- `unload` removes exactly the named loaded root and its documents. It does not
  accept an arbitrary descendant as a root alias.

Arguments can be quoted when paths or query text contain spaces. A backslash
escapes a single or double quote; other backslashes are preserved as ordinary
path characters. `quit` and `exit` are console-only controls, not RPC methods.

## Index scope and lifetime

This version stores **one document and one hybrid embedding per file**: the dense
BGE-M3 CLS vector plus the sparse BGE-M3 lexical weight vector of the same
forward pass. Every document and query uses `EncodeOptions{max_seq_len = 512,
want_sparse = true}`, so the sparse head costs no extra forward pass. A long file
is still truncated by the encoder; its embedding is not whole-document semantic
coverage. Chunking, automatic filesystem watching, and persistence of the
in-memory index are out of scope (the on-disk embedding cache described below is
a per-file encode cache, not an index). The index itself exists only in process
memory and is discarded at exit; only the encode cache survives.

Retrieval quality is corpus-dependent and is not a guaranteed number: the hybrid
score improves on dense-only ranking for documents that share rare tokens with
the query, but a quantized package (bf16/fp16/int8) lowers the quality of both
halves. Measure on your own corpus; no hit-rate figure is promised here.

## Embedding cache (`.lc_embed_cache*`)

Indexing a directory writes a per-root cache next to it:

| Entry | Content |
|---|---|
| `.lc_embed_cache.json` | per-file content hash, size, mtime, dimension, and (hybrid entries only) `sparse_count` |
| `.lc_embed_cache/<md5>.bin` | the dense embedding, `dimension` fp32 values (4096 bytes at dimension 1024) |
| `.lc_embed_cache/<md5>.sparse.bin` | the canonical sparse pairs as raw `(uint32 token_id, float weight)` records, 8 bytes each |

A hybrid cache hit needs **both** halves: the dense record and a valid sparse
sidecar. When the sidecar is missing, unreadable, or the metadata carries no
`sparse_count` (that is exactly the layout an older dense-only build wrote), the
file is re-encoded and both halves are rewritten — the first `load` of a
dense-only cache therefore reports `cached=0`, and the second one reports
`cached=<n>` again. A zero-byte sidecar is a valid empty sparse vector. Dense-only
use never creates sidecars, so such caches stay byte-identical to before.

All paths refer to the server's filesystem. Traversal does not follow directory
symlinks, uses deterministic canonical-path ordering, and never treats a string
prefix such as `docs2` as a child of `docs`.

## JSON-RPC over TCP

Each request is one complete UTF-8 JSON object followed by `\n`; each response is
one JSON object followed by `\n`. The transport buffers fragmented TCP reads and
handles coalesced requests. A request line is limited to 1 MiB. Oversized input
receives a protocol error and the offending connection is closed cleanly.
JSON-RPC batch arrays are outside the supported subset and are rejected.

Supported requests:

```json
{"jsonrpc":"2.0","id":1,"method":"load","params":{"path":"D:\\docs","extensions":[".txt",".md"]}}
{"jsonrpc":"2.0","id":2,"method":"query","params":{"text":"dispatch semantics","top_k":5}}
{"jsonrpc":"2.0","id":3,"method":"update","params":{"path":"D:\\docs\\chapter"}}
{"jsonrpc":"2.0","id":4,"method":"unload","params":{"path":"D:\\docs"}}
```

`load.params.extensions` is optional and defaults to `[".md", ".txt"]` after
normalization. `query.params.top_k` is optional and defaults to `5`. IDs may be
strings or numbers and are preserved in responses. A request without `id` is a
notification: it is executed, failures are logged, and no response is sent.

Successful result objects have these schemas (all counts are nonnegative
numbers, paths are canonical UTF-8 strings, and scores are JSON numbers):

```json
{"path":"D:/docs","extensions":[".md",".txt"],"added":2,"skipped":1,"failed":0,"total_documents":2}
{"path":"D:/docs/chapter","added":0,"updated":1,"removed":0,"unchanged":3,"failed":0,"total_documents":4}
{"path":"D:/docs/a.md","score":0.812345}]}
  {"path":"D:/docs","removed_documents":2,"total_documents":0}
  ```
  Hybrid retrieval changed the *value* of `score`, not the schema: it is still one
  JSON number (`0.7 * dense_cosine + 0.3 * sparse_cosine`), no field was added,
  renamed, or removed, and no RPC method changed. Score ranges differ from a
  dense-only build, so a client that hard-coded a dense threshold must be
  re-tuned.

Errors use JSON-RPC `-32700` (parse), `-32600` (invalid request), `-32601`
(method not found), `-32602` (invalid params), `-32000` (operational failure),
and `-32603` (response construction failure). Every error contains a readable
`error.message`; malformed requests and disconnects do not terminate the server.

Each accepted connection has its own handler thread. Calls into the shared
retrieval service are mutex-serialized, so mutations cannot race and every call
sees a consistent index. A long load/update blocks calls from other clients.
Orderly shutdown closes the listener, wakes client handlers, and joins all
threads.

### Minimal Python client

```python
import json
import socket

with socket.create_connection(("127.0.0.1", 8765)) as sock:
    request = {"jsonrpc": "2.0", "id": 1, "method": "query",
               "params": {"text": "dispatch semantics", "top_k": 5}}
    sock.sendall((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))
    response = sock.makefile("rb").readline()
    print(json.loads(response))
```

## Security

The default listener is loopback-only. There is deliberately **no authentication,
authorization, or TLS**. Remote exposure is out of scope: do not bind this simple
example to an untrusted interface. RPC paths grant clients the server's filesystem
access within the loaded-root operations, so an exposed listener would also
expose an administrative indexing interface.

## Build and host tests

CMake and xmake both register `example_embed_retrieval`. The retrieval core,
service, protocol, and transport are compiled into `lc-example-embedding`; the
executable links the existing engine and `ws2_32` only on Windows. Host-only
tests require no model, backend, or GPU:

```text
test_embed_retrieval_index
test_embed_retrieval_rpc
```

They use deterministic fake embeddings and are registered under CTest labels
`unit;unit_embedding`. Existing fixtures under `src/tests/data/embedding` are not
used or regenerated.

## C-FFI library (`example_embed_retrieval_lib`)

The console and the JSON-RPC server are not the only front ends. The same steps
are exported as a documented C ABI from a shared library so any host language
can drive the retriever through its FFI:

```text
example_embed_retrieval_lib.h (ABI v1.2) eer_context_create / load_dir(_ex) /
 add_document / update_document / remove_document / query(_ex) /
 update_path / unload_dir / encode_text / encode_batch / encode_colbert /
 index_save / index_load / stats / ping / device_memory / park / resume / destroy
```

One call per console command, C scalars and UTF-8 strings only, caller-owned
result structs released with the matching `*_free`. Besides the model-backed
"engine mode" it also supports a GPU-free "callback mode" where the host
supplies the dense encoder (and, since 1.2, an optional sparse lexical encoder
for hybrid ranking), which is how it is unit-tested
(`test_embed_retrieval_api`, host-only, no model required). The 1.2 RAG growth
(report.md §2-§6) adds chunked documents with passage byte ranges, corpus
snapshots, per-query ranking controls (weights / metadata filters / MMR /
truncation / instruction prefix / ColBERT), batch + ColBERT encoding, progress
and cancellation, structured sub-codes, stats/ping and the honest
park/resume-unsupported path — all additive behind `struct_size`.

The two modes rank differently and deliberately so:

| Mode | Encoder | Scoring | Cache |
|---|---|---|---|
| engine (model-backed) | `Session` with `want_sparse = true` | `0.7 * dense_cosine + 0.3 * sparse_cosine` | dense record + `<md5>.sparse.bin` sidecar |
| callback, dense only (default) | `eer_encode_fn` | plain dense dot product (unchanged) | dense record only, exactly as before |
| callback, hybrid (1.2) | `eer_encode_fn` + `eer_sparse_encode_fn` | `0.7 * dense + 0.3 * sparse` (per-query overridable) | dense + sparse sidecars |
Callback mode is dense-only unless BOTH callbacks are supplied (1.2): the service
is then constructed from a `RetrievalHybridEncodeCallback`, so a dense-only host
can never be silently promoted — the sparse half is either supplied explicitly or
absent. `eer_encode_text` likewise still returns the dense vector only.
returns the dense vector only.

```bash
xmake build example_embed_retrieval_lib
xmake run test_embed_retrieval_api
```

Full per-step contract, error table, ownership rules and C / Python `ctypes`
examples: [`api/README.md`](api/README.md).

## Quantized weight packages (bf16 / fp16 / int8)

`tools/pack_model.cpp` bakes the element format into the `.bgm` file, so
quantization is a **packaging** decision rather than a runtime reinterpretation
of one blob as another:

```bash
xmake build example_embed_pack_model
xmake run example_embed_pack_model <model_dir> <out.bgm> \
    --precision fp32|bf16|fp16|int8 --check
```

| `--precision` | BGM3 version | Element encoding | Device path |
|---|---|---|---|
| `fp32` (default) | v1 | 4-byte LE floats | original fp32 kernels (byte-identical to the historical package); or the packed-half kernels over weights converted on the device once at start-up when `LUISA_EMBED_WEIGHTS=bf16\|fp16` |
| `bf16`, `fp16` | v2 | 2-byte LE elements, round-to-nearest-even at pack time | GEMM weights uploaded packed, converted at the kernel B-tile load |
| `int8` | v3 | one signed int8 code per element plus one fp32 per-tensor scale in the tensor's table entry | GEMM weights uploaded packed as `[scale word, then four codes per word]`, decoded as `code * scale` at the B-tile load |

The v1 and v2 formats are unchanged: existing packages load and behave exactly
as before. `--check` reopens the written file, dequantizes on read-back and runs
the reference forward, so the parity gate measures the real quantization effect
(cosine >= 0.9999 for fp32, >= 0.999 for v2, >= 0.99 for v3) instead of the
pre-quantization weights.

### Selecting the device path: `LUISA_EMBED_WEIGHTS`

`LUISA_EMBED_WEIGHTS=fp32|bf16|fp16|int8` (case-insensitive) is read once per
`WeightStore` and logged. Unset means **auto**: use the package's own format.
Every request is validated against the loaded package and a mismatch fails
closed with a diagnostic naming the package version to pack:

| Package | unset (= auto) | `fp32` | `bf16` / `fp16` | `int8` |
|---|---|---|---|---|
| v1 fp32 | fp32 | fp32 | packed half, converted on the device once at start-up | refused |
| v2 bf16 / fp16 | packed half | fp32 A/B arm | same format only | refused |
| v3 int8 | packed int8 | fp32 A/B arm | refused | packed int8 |

The `bf16` / `fp16` cell for a **v1 fp32** package is the *fp32-source*
conversion: the package stays exactly what it is (nothing is re-packed on disk),
and at `WeightStore` construction the four per-layer GEMM matrices (`qkv`,
`attn.out`, `ffn.up`, `ffn.down`) are copied to the device, packed there by the
fp32 -> packed-half kernel into the *same* word layout a v2 blob holds, and then
read by the packed-half GEMM kernels — bit-identically to packing a v2 package of
that format out of the same weights. Every other tensor stays fp32 (biases,
LayerNorm parameters, embeddings, head vectors), the accumulation stays fp32, and
there is no format branch in the encode path: only the B-operand load differs,
and it differs in a separately compiled shader. `int8` on a v1 package stays
refused: fp32 -> int8 needs a per-tensor calibrated scale that only the packer
computes, so the engine never re-quantizes weights on the device.

That conversion is a **one-shot start-up cost**, not a second resident copy: the
fp32 staging buffers (~1.1 GiB for bge-m3) exist only until the pack dispatches
retire. The `Scheduler` takes ownership of them, lets them overlap the Executors'
shader compilation, then synchronizes the init stream once and releases them —
logged as `released <N> MiB of transient fp32 conversion staging`. Sessions that
convert nothing never add that synchronize, and the staging is never kept for the
lifetime of the store.

The `fp32` override on a v2 or v3 package is the **A/B arm**: every tensor is
up-converted or dequantized once on the host at upload, so the engine runs its
original fp32 kernels over quantized-derived weights. Diffing that arm against
the packed arm isolates the steady-state effect of the storage format from every
other change. Both arms evaluate the same `q * scale`, so on the reference v3
package they agree to print precision (the packed arm multiplies inside the
kernel's B-tile fill, the A/B arm once on the host); only the surrounding fp32
rounding can differ.

### How the specialization works

All arms share one set of DSL kernel lambdas. The host switches
`GemmWeightFormat::{Fp32, Bf16, Fp16, Int8}` when compiling them, so each weight
format becomes a **separately compiled shader** — named `gemm_bias_int8`,
`gemm_bias_gelu_erf_int8`, ... and reported as `weight_format='int8'` by the
tensor-path log line. There is no format branch in the inner loop and no
per-element format test at encode time; the only difference between the compiled
variants is the B-tile decode step.

Accumulation is **fp32 in every arm**. bf16, fp16 and int8 are storage formats
only: each element is converted to fp32 before the multiply, and the engine uses
no integer/DP4A and no half-precision tensor-core arithmetic. Biases, LayerNorm
parameters and head weights stay fp32 in every packed mode (their traffic is
O(rows) or O(1), so packing them would buy no bandwidth), and the head.sparse.b
scalar is dequantized on the host into a plain float.

### Residual + LayerNorm fusion: `EMBED_FUSE_RESIDUAL_LN`

Each encoder layer runs a `residual add` -> `LayerNorm` pair twice, so a 24-layer
encode issues 96 dispatches for what is one `LN(a + b)` per row. The fused
`residual_layernorm` kernel does each pair in one dispatch — 48 fewer dispatches
per 24-layer encode — because it computes the fp32 residual once and caches the
whole row in shared memory, then runs the LayerNorm's three passes over that cache
instead of re-reading the row plane three times from global memory: the pair
traverses the row plane seven times (2R+1W for the add, 3R+1W for the LayerNorm),
the fused kernel three times (2R+1W). The host contract is `H <= 1024`
(`kResidualNormMaxHidden`), which sizes that static cache (~5 KiB per block
including the reduction tree); a longer row keeps the legacy pair and the log says
why. The fusion is orthogonal to the weight format: the fused shader is fp32 in
every `LUISA_EMBED_WEIGHTS` arm, and only the GEMM operands are packed.

The fused kernel preserves the pair's reduction and expression order, so its
output is numerically equivalent within the focused `1e-5` absolute tolerance that
`test_embed_kernel_layernorm` asserts — but it is not bit-identical: emitting the
same op sequence as one kernel instead of two lets dx re-emit ~1 ULP of
difference (measured: the pair gives -0.13910183 and the fused kernel gives
the neighbouring value -0.13910185). The swap is a rounding-level substitution,
never a numerical change.

`EMBED_FUSE_RESIDUAL_LN=0` is the kill switch: it restores the legacy
`residual_add` -> `layernorm` dispatch sequence unchanged (perf A/B and the
toggle-regression gate). Any other value, including unset, keeps the fusion on,
which is the default. The decision is made once per Executor at construction —
exactly one shader set is compiled, so no run can mix the two — and logged once as
either `residual+layernorm: fused 'residual_layernorm' (H=..., 1 dispatch per
pair; kill switch EMBED_FUSE_RESIDUAL_LN=0)` or `... legacy pair ... (reason:
...)`.

Release measurements (fp32-source fp16 arm: a v1 fp32 package with
`LUISA_EMBED_WEIGHTS=fp16`, default tensor path, one encode timed as record +
commit + sync, best of the iterations after 3 warm-ups):

| Backend | batch=1 legacy pair | batch=1 fused | batch=8 legacy pair | batch=8 fused |
|---|---|---|---|---|
| dx | 39.852 ms | 39.618 ms | 235.089 ms | 233.675 ms |
| vk | 41.163 ms | 40.935 ms | 236.732 ms | 235.618 ms |
| cuda | 38.333 ms | 38.241 ms | 230.645 ms | 229.863 ms |

20 iterations at batch 1 and 10 at batch 8, RTX 4080 SUPER, release build. The
gain is 0.09–0.23 ms (0.2–0.6 %) at batch 1 and 0.78–1.41 ms (0.3–0.6 %) at
batch 8: modest, and workload-specific, because it is the cost of 48 dispatch
records removed from a GEMM-dominated encode. A smaller model, a shorter
sequence, or a backend with cheaper dispatches shows less, so this is not a
universal speedup claim — use the kill switch as the A/B arm on your own package.

Rejected experiment, kept for the record: a CUDA QKV GEMM tile-shape variant won
its kernel microbenchmark but regressed end-to-end encode latency, so it was not
adopted. Kernel-level wins here are only kept when the E2E encode improves.

Honest expectations

FP16, BF16 and INT8 reduce *weight storage and weight-read bandwidth*. That is
the entire promise. Whether it turns into latency depends on shape, backend and
where the encode is bound, and no latency or retrieval-accuracy number is
guaranteed here: measure on your own device in a release build, with the `fp32`
override as the A/B reference and the fallback backend as the correctness
reference. Debug builds are for correctness gates, not for statements about
speed. Packing is not free work removed: every packed arm adds a decode step per
B-tile element (a 16-bit upcast, or a byte extract + int-to-float + multiply), so
a shape that is not weight-bandwidth bound can get slower rather than faster —
that is exactly what the `fp32` A/B arm is for. INT8 is per-tensor symmetric
quantization and is **lossy** — validate retrieval quality on your own corpus
before adopting a v3 package; if it drops, pack v2 instead. The `bf16` / `fp16`
arm on a v1 fp32 package costs one extra thing that a v2 package does not: a
one-shot device conversion at start-up, during which a transient fp32 staging
copy of the GEMM weights is resident (released right after the pack dispatches
retire and reported in the log). Steady state — device footprint, weight-read
bandwidth, kernels — is then identical to a v2 package of the same format, so a
start-up that pays ~1.1 GiB briefly is the price of not re-packing the blob; if
that start-up cost is unacceptable, pack a v2 package once instead. As an
artifact data point only (not a performance claim), the reference bge-m3 model
packs to about 568 MB as v3 versus 2.27 GB as v1 (4x smaller) and 1.14 GB as v2.
