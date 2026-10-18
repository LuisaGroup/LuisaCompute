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
path before accepting commands. Use `--help` for the complete process command line.

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
- `query` defaults to five hits; `--top` accepts 1 through 100. Dense vectors are
  L2-normalized by `Session`, so their dot product is the cosine score. Results
  sort by descending score and then ascending canonical path. Querying an empty
  corpus, including loaded roots with no documents, is a recoverable error.
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

This first version stores **one document and one dense embedding per file**.
Every document and query uses the existing default `EncodeOptions`, including
`max_seq_len = 512`. Consequently, a long file is truncated by the encoder; its
embedding is not whole-document semantic coverage. Chunking, sparse/hybrid
retrieval, automatic filesystem watching, and persistence are out of scope.
Embeddings exist only in process memory and are discarded at exit.

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
{"results":[{"path":"D:/docs/a.md","score":0.812345}]}
{"path":"D:/docs","removed_documents":2,"total_documents":0}
```

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
