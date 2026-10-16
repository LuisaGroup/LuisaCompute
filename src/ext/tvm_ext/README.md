# src/ext/tvm_ext

TVM-side sources that this repository owns, instead of patching the
`src/ext/tvm` submodule. `src/ext/xmake.lua` adds `*.cc` from this directory to
the `tvm_compiler` target, so a `git submodule update` (which would drop an
untracked file inside the submodule) cannot lose them.

## `cuda_header_generator.cc`

`src/backend/cuda/codegen/codegen_cuda.cc` unconditionally requires two FFI
globals:

* `tirx.intrinsics.cuda.header_generator` — `tags -> CUDA preamble`
  (`CodeGenCUDA::Finish()`)
* `tirx.intrinsics.cuda.get_codegen` — `op name -> optional codegen`
  (`CodeGenCUDA::Dispatch_(CallNode)`)

Upstream registers both only from the Python package
(`python/tvm/backend/cuda/codegen/header.py`, `registry.py`). A pure C++
embedding of TVM (this project's xmake/CMake TVM port) never imports Python, so
the CUDA-C route — used by the TileIR → TIRx → CUDA device artifacts — aborted on
an `ICHECK` before emitting any source.

The file therefore registers:

* the header generator, as a mechanical port of the upstream Python
  `header_generator()` (identical payloads and tag conditions), and
* an empty `get_codegen` registry, so ops that exist only as Python codegen
  registrations keep failing closed (the built-in CUDA ops stay handled by
  `codegen_cuda.cc` itself) instead of aborting.

Do not edit the generated file by hand:

```bash
python scripts/port_cuda_header_generator.py    # regenerate from the submodule's Python source
python scripts/verify_cuda_header_port.py       # check payload/condition parity
```

Re-run both when `src/ext/tvm` is updated and
`python/tvm/backend/cuda/codegen/header.py` changed.
