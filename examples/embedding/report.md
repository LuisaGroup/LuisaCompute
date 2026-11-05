# bge-m3 retrieval — the 1.2 RAG growth report

This is the design record the retrieval sources cite as `report.md §2`–`§6`
(`embed/retrieval_index.{h,cpp}`, `embed/retrieval_service.h`,
`embed/chunker.h`, `embed/retrieval_disk_cache.h`,
`api/example_embed_retrieval_lib.{h,cpp}`). It describes what the ABI 1.2 growth
IS and why each piece exists, in the order the sections are numbered. The code is
the authority on details; this file is the authority on intent.

The 1.1 engine answered "which FILE matches this query": one dense embedding per
document, one hit per document, a single flag-free query call. That shape cannot
serve retrieval-augmented generation, where the caller needs the PASSAGE that
answers a question, the byte range to quote it, metadata to filter on, progress
for a multi-minute corpus load, and a cancel that actually stops the GPU. The 1.2
growth adds exactly those, and nothing else: every addition is APPEND-ONLY in the
C ABI (new fields at the end of a struct, new structs behind a `struct_size`,
new `eer_*` entry points), so a 1.1 caller keeps working bit-for-bit.

## §1 Scope and invariants

* **One context per process.** `eer_context_create`/`destroy` are not
  thread-safe and a context owns one GPU device; every step below assumes a single
  context driven from one thread at a time (the service serializes all steps
  through one mutex).
* **No authorization state in the engine.** Visibility is the caller's job: each
  query carries an optional `filter_json` over document ids/metadata (§4). The
  engine stores no access rules, so a cache or a snapshot can never bypass one.
* **The engine is a derivative.** PostgreSQL (or any host store) stays the
  authority; the index, its disk cache (`.lc_embed_cache*`) and its snapshots are
  rebuildable (§3).
* **Byte offsets, not characters.** Every passage is reported as
  `[byte_offset, byte_length)` into the UTF-8 source. A Chinese character is 3
  bytes, so a host must slice bytes, never `String.Substring` (§2).
* **Nothing is silently dropped.** Text the encoder had to cut, files that failed,
  steps a cancel interrupted and chunks skipped all appear in counters or
  structured sub-codes (§2, §6).

## §2 Chunked document ingestion

`embed/chunker.{h,cpp}` — `chunk_text()` splits a document into byte-range chunks
along paragraph (blank line), sentence or token boundaries, packs the units
greedily to `chunk_size` bytes, and repeats up to `chunk_overlap` bytes of the
previous chunk's tail.

* **UTF-8 aware in both directions.** A chunk boundary never lands inside a
  multi-byte code point, so a host can decode any reported range.
* **CJK first class.** A sentence ends after `。`/`！`/`？`/`；`/`…` (and any
  closing quote) whether or not a space follows — "第一句。第二句。" is two units;
  the ASCII `.!?` still require trailing whitespace, which keeps "3.14" and
  "e.g." whole. U+3000 IDEOGRAPHIC SPACE is a whitespace run like U+0020, and a
  unit bigger than `chunk_size` is hard-split at code-point boundaries stepping by
  `chunk_size - overlap`, so ~170 Chinese characters per chunk still hold and the
  overlap survives inside a split unit.
* **One record per chunk.** `_ingest_document` encodes every chunk BEFORE mutating
  the index (a per-document transaction: a failing or cancelled ingest leaves the
  previous version fully intact), then swaps the document's chunk records.
  Idempotent re-ingestion of identical text/metadata/chunking reports
  `chunks_unchanged` and encodes nothing.
* **`add_document` / `update_document` / `remove_document`** address documents by
  a caller-chosen `doc_id` (caller-chosen for in-memory text, the display path for
  directory loads), which is what makes memory-only documents indexable at all.
* **Chunk-aware cache.** Entries are keyed on (content digest, chunk index, chunk
  parameters) through the `*_variant` hooks of `RetrievalEmbedCache`, so a
  re-chunk asks for different entries instead of serving a stale whole-file vector.

## §3 Corpus snapshots

`retrieval_index.cpp` (`save` / `load_snapshot`) — a snapshot is a JSON manifest
plus a packed float blob: roots, documents, chunks, metadata and embeddings
restore with **no re-encode** (O(1) cold start), and memory-only documents are
serialized with their text. Bump `kSnapshotVersion` on any layout change; a
mismatching, truncated or corrupt snapshot is a clean, recoverable operational
error that leaves the live index untouched.

Snapshots are NOT the authority (§1): they hold document text, so a purge or a
permission revocation has to reach them too. Treat "load the snapshot, replay the
deletes" as the recovery shape, not "the snapshot is the corpus".

## §4 Query controls

`RetrievalQueryOptions` (and `eer_query_options` in the C ABI) adds, per query:

* **Fusion weights** — `dense_weight` / `sparse_weight` override the compiled
  defaults (0.7 / 0.3). NaN or negative selects the default; dense mode keeps the
  plain dot product bit-for-bit, so a dense-only caller sees no change.
* **Metadata filters** — `filter_json` is a small predicate tree: top-level keys
  with scalar equality, `{"$gte"/"$lte"/"$gt"/"$lt": value}` leaves and `$in`
  lists, combined by implicit/explicit `$and` and `$or`. Applied BEFORE top-k
  selection, so a filtered query returns up to top-k matching passages.
* **`max_seq_len`** — per-query encode truncation length (engine mode), with
  `report_truncation` asking for the "the query text was cut" flag back.
* **Instruction prefix** — prepended verbatim when the caller wants an
  asymmetric-retrieval instruction, without rewriting the stored passages.
* **`diversity`** — 0..1 MMR re-ranking: relevance stays `(1 - d) * fused score`
  and the penalty is `d * max cosine to an already-selected passage`, computed on
  the L2-normalized dense vectors.
* **`want_colbert`** — ColBERT MaxSim re-ranking of the fused candidate list (§5).

`filter_json` is deliberately data, not SQL: the engine evaluates it over the
metadata it stored verbatim at ingest, which keeps the "no authorization state"
invariant and makes the filter trivially cacheable by the host.

## §5 Encoding surface, ColBERT and long-running steps

* **`encode` / `encode_batch` / `encode_colbert`** expose the encoder without
  touching the index, so a host can pre-compute vectors or compare modes. The
  implementation encodes ONCE per call: the buffer is sized from the cached
  embedding dimension (16 KiB cap) instead of running a native size probe that
  would repeat the encode — in callback mode a paid remote encoder must not be
  billed twice.
* **Sparse encoder callback** (`eer_sparse_encode_fn`) turns callback mode into a
  hybrid index without a GPU: the host supplies both the dense vector and the
  sparse lexical (token id, weight) pairs. The index canonicalizes every sparse
  vector (sort, merge duplicates, L2-normalize in double), so a fused score is a
  cosine in [0, 1] and a malformed vector is a recoverable error, never a silent
  0.
* **ColBERT** is opt-in (`eer_context_set_store_colbert`): keeping the
  per-token matrices costs RAM, so when it is off the payload is dropped and
  `want_colbert` queries fail loudly; a record that predates ColBERT
  serialization degrades to the fused dense/sparse score instead of failing.
  Matrices are never cached on disk.
* **Progress and cancellation** (`eer_progress`): `on_progress(done, total)` runs
  on the calling thread after every completed unit and `cancel_flag` is polled
  between units, so a cancel stops at a consistent boundary — already-ingested
  content stays, the in-flight document is applied all-or-nothing, and the result
  reports `EER_SUB_CANCELLED`. A directory load reports FILE ticks only (one per
  ingested file, `total` = the file count); document ingestion reports CHUNK
  ticks. A cancel that lands inside a file — the LAST one included — is treated as
  the cancellation it is: the status is the cancelled one, the root registration
  is rolled back, and a later load without the flag resumes the directory. The
  hook runs inside the service mutex, so it must not call back into the context.

## §6 Observability and failure taxonomy

* **Statistics** (`eer_stats` → `eer_stats_data`): distinct document count, chunk
  count, dimension, hybrid flag, cache hit ratio, mean encode latency and encode/
  cache counters. The snapshot is taken under the same mutex every step takes —
  the serving surface may be read while a load runs, so an unlocked read would be
  a data race. `struct_size` gates every field, so a host built against 1.1 still
  gets its own prefix back.
* **Structured sub-codes** (`eer_sub_code`): which subsystem failed — device,
  model, I/O, encode, index, cancelled — so an orchestrator can retry, reload or
  give up without parsing a message. Statuses stay the flat four (OK,
  invalid_argument, not_found, operational).
* **Truncation accounting**: a chunk the encoder cut at `max_seq_len` is embedded
  truncated, which means the tail of that passage is in no embedding. Every
  result that can do this says so — `eer_load_result.truncated_chunks`,
  `eer_document_result.truncated_chunks`, `eer_query_result.truncated` — and 0 is
  the healthy value: re-ingest with a smaller `chunk_size` or a larger
  `max_seq_len` to index the passage whole.
* **Serving surface**: `ping` (readiness, no corpus access) and `backend`
  (which device the context runs on) sit next to stats. `park`, `resume` and
  `get_device_memory` are declared for ABI symmetry but are NOT supported by the
  engine; they answer an operational error with a message that says so, and the
  C# facade marks them `[Obsolete]` rather than pretending.

## §7 Verification boundaries

* The host-only unit tests (`src/tests/unit/embedding/`) drive everything above
  without a GPU or a model: `test_embed_chunker` (CJK sentence terminals, U+3000,
  the oversized hard split, byte-exact ranges), `test_embed_retrieval_index`
  (chunked ingest, ranking, updates, cancellation, the disk cache),
  `test_embed_retrieval_api` (every `eer_*` step in callback mode) and
  `test_embed_retrieval_rpc` (the JSON-RPC front end).
* Device tests (`src/tests/integration/embedding/`) need a backend and the packed
  model; `tests/Semevia.Ffi.Tests/Native` in the Semevia workspace runs the same
  surface through the C# binding, opt-in (`SEMEVIA_EMBED_RETRIEVAL_TEST=1`).
* A 1.1 library must be refused up front by the host (`VerifyAbi`: major 1, minor
  >= 2) instead of failing later with a missing entry point, and the append-only
  rule above is what makes a 1.2 library usable by a 1.1 host.
