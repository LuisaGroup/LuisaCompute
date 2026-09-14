# CUDA Workflow-B: TVMx + CUDA build and first-run checklist

Workflow-B is the environment that can actually **compile and run** the CUDA
TIRx-gated paths: the `#ifdef LUISA_CUDA_TILE_TIRX` factory body in
`src/backends/cuda/tile/cuda_tile.cpp`, the bridge CUDA/NVPTX device-artifact
route in `src/tile/bridge/tirx`, and the two TIRX-gated CUDA suites. Ordinary
development machines without the pinned TVMx tree only build the fail-closed
non-TIRX variant; that is expected and is not a signal that the CUDA Tile work is
stubbed.

This page is an executable recipe, not a claim that it has been run on this
machine. Update it if the pinned commits or commands change.

## 1. TVMx checkout/build (CUDA codegen)

```bash
TVM_SRC=/path/to/tvm-src
TVM_BUILD=/path/to/tvm-build
git -C "$TVM_SRC" checkout c7b458e946bc4266915da582457476bdcd9705ae
git -C "$TVM_SRC/3rdparty/tvm-ffi" fetch
git -C "$TVM_SRC/3rdparty/tvm-ffi" checkout 12dbf053b3d9ba4ebd9da3123b1aeca79cf74229
git -C "$TVM_SRC/3rdparty/tvm-ffi" submodule update --init --recursive

cmake -S "$TVM_SRC" -B "$TVM_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_CUDA=ON \
  # optional: -DUSE_CUDA_ARCHITECTURES=<host CC>
  # Keep Metal OFF; no MPP patches are required for the CUDA reference route.
cmake --build "$TVM_BUILD"
```

Verify both runtime builders exist before touching LuisaCompute:

```bash
# target.build.cuda  -> InspectSource("cuda")  -> DeviceArtifact::CUDA_SOURCE
# target.build.nvptx -> InspectSource("ptx")   -> DeviceArtifact::Format::PTX
```

If the pinned tree lacks either builder, record the missing-builder failure that
`compile_device()` reports; do **not** patch silently.

## 2. LuisaCompute CMake configure

```bash
cmake -S /path/to/luisa -B /path/to/luisa-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLUISA_COMPUTE_ENABLE_CUDA=ON \
  -DLUISA_COMPUTE_ENABLE_TILE_TIRX_BRIDGE=ON \
  -DLUISA_COMPUTE_TVM_INCLUDE_DIR="$TVM_SRC/include" \
  -DLUISA_COMPUTE_TVM_LIBRARY_DIR="$TVM_BUILD/lib" \
  -DLUISA_COMPUTE_TVM_FFI_INCLUDE_DIR="$TVM_SRC/3rdparty/tvm-ffi/include" \
  -DLUISA_COMPUTE_TVM_FFI_LIBRARY_DIR="$TVM_BUILD/lib" \
  -DLUISA_COMPUTE_BUILD_TESTS=ON
cmake --build /path/to/luisa-build --target luisa-compute-backend-cuda test_tirx_device_cuda test_tile_cuda_ptx test_cuda_ptx_version
```

Expected wiring:
- `luisa-compute-backend-cuda` links `luisa-compute-tile-bridge-tirx` and is
  compiled with `LUISA_CUDA_TILE_TIRX=1`;
- `test_tile_cuda_ptx` links `luisa::tile-tirx` and defines
  `LUISA_TEST_TILE_CUDA_TIRX=1`;
- `test_tirx_device_cuda` links `luisa::tile-tirx`.

## 3. First-run triage list

The first compile-fix pass on Workflow-B should expect (at least) the risks
listed below. Every failure becomes a source fix in this phase's file set;
never remove or silence a test.

- TVM C++ API drift versus the pinned headers used by
  `src/tile/bridge/tirx/compiler.cpp` and `cuda_tile.cpp`
  (`max_num_threads`/`max_shared_memory_per_block` int64 attrs,
  `PrimFunc::GetAttr` returns, `InspectSource` lifetimes, IRModule function
  counts).
- The warp-aligned mapper change interacting with the CUDA gates.
- `compile_device` for `nvptx`: `InspectSource("ptx")` must be non-empty and
  NUL-safe for `cuModuleLoadData`.
- C++ errors in the `#ifdef LUISA_CUDA_TILE_TIRX` factory body that only
  surface with the define set.
- Host test build issues in `test_tirx_device_cuda.cpp`.

## 4. CTest invocations

```bash
ctest --test-dir /path/to/luisa-build -R 'test_cuda_ptx_version' --output-on-failure   # host, no GPU
ctest --test-dir /path/to/luisa-build -R 'test_tirx_device_cuda' --output-on-failure   # host artifacts
ctest --test-dir /path/to/luisa-build -R 'test_tile_cuda_ptx' --output-on-failure      # device oracle + cache/retry
```

`test_tile_cuda_ptx` passes `cuda` as the backend argument; it needs a healthy
CUDA device. The cache round-trip suite uses an in-memory `BinaryIO`
(`DeviceConfig::binary_io`), and the simulated old-driver retry sets
`LUISA_CUDA_TILE_FORCE_UNSUPPORTED_PTX=1` inside the test process for the first
compile and clears it for the cold-cache compile.

## 5. Environment caveats

- **No CUDA device**: the host suites (`test_cuda_ptx_version`,
  `test_tirx_device_cuda`) still run; `test_tile_cuda_ptx` cannot.
- **Old driver**: the patch-retry test also exercises the real
  unsupported-version path.
- xmake: enable the optional bridge with `lc_tile_tirx_bridge=y` plus the
  `lc_tvm_*` paths (same TVM options as the CMake configure). When it is off,
  xmake builds intentionally run the fail-closed variant; when it is on, the
  tile XIR bridge is always compiled into `lc-tile` and the CUDA backend/test
  define `LUISA_CUDA_TILE_TIRX` / `LUISA_TEST_TILE_CUDA_TIRX` exactly like the
  CMake TIRX bridge wiring.
