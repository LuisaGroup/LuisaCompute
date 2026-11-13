# Device-library sources (`builtin/`)

This directory holds the HLSL text that the LuisaCompute backends embed into their
device libraries, plus the pre-compiled artifacts built from it.

## Rule: no comments in any code file here

**Do not write comments in the code files of this directory. Not `//`, not `/* */`, not
commented-out code, not `TODO`s.** The rule applies to every file below, without
exception:

- every `*.bytes` fragment (`hlsl_header`, `raytracing_header`, `accel_header`,
  `bfloat16_fallback`, `vk_linalg`, `dx_linalg`, the BC6/BC7 encoder pieces, ...);
- every extensionless HLSL fragment (`bc6_*`, `bc7_*`, `bindless_upload`,
  `bindless_upload_vk`, `load_bdls_dxil`);
- `hlsl_builtin.hpp`, even though it is ordinary C++.

Binary artifacts (`.dxil`) are not edited directly at all - they are regenerated from
the fragment they are built from (see below). This `README.md` is the only file in this
directory that may contain prose.

A comment added here will be removed again. The files were stripped once in a bulk pass
that was verified to be token-identical to the previous revision, so "but the comment
explains why" is never a reason to put it back: the explanation belongs in one of the
places listed below, where it can be reviewed, kept in sync with the code that emits or
loads the fragment, and where it does not cost anything at build time.

## Why

1. **The text ships.** These fragments are embedded byte for byte into the runtime
   (`rules/utils.bin2obj` on xmake, `utils/embed_device_lib.cpp` on CMake), so every
   comment becomes part of `luisa-runtime` / `luisa-backend-*`. It is dead payload in
   every build and in every shader dump a user has to read.
2. **Comments break shader caching.** The generated translation unit contains this text
   verbatim, and the Vulkan HLSL-to-SPIR-V route keys its cache on the MD5 of the
   generated source (`src/backends/vk/compute_shader.cpp`). A comment-only edit changes
   that MD5, so every cached SPIR-V module is invalidated for a change the compiler will
   throw away - and a review sees "the device library changed" when nothing did.
3. **The `.dxil` artifacts are compiled from these fragments.** `bc6_*`/`bc7_*` pieces
   are turned into `bc6_encodeblock.dxil` and friends with `lc_compile_builtin`; a
   comment-only difference changes the artifact bytes and hides a real code change in a
   binary diff.
4. **These are fragments, not standalone programs.** They are spliced into one generated
   translation unit in a fixed order (`detail::AddHeader` in
   `src/backends/common/hlsl/codegen_utils/entry_points.cpp`), so a comment describing
   "the file above" is usually describing the wrong fragment by the time it is included.

## Where the documentation goes instead

| What you wanted to explain | Where it belongs |
| --- | --- |
| Why an instruction / extension is used, and which device feature gates it | The emitting code: `src/backends/common/hlsl/codegen_utils/*.cpp`, `src/backends/common/hlsl/hlsl_codegen.h`, `src/backends/common/hlsl/shader_compiler.cpp` |
| Which extension a fragment's instruction needs, and how the host negotiates it | `src/backends/vk/vk_shader_capabilities.h` and `docs/source/internals/vulkan_optional_extensions.md` |
| Which of the files here is embedded, under which key, and how to regenerate an artifact | The `hlsl` skill document (`.agents/skills/hlsl/SKILL.md`), sections "Header Files" and "Adding Builtins" |
| Why a fragment is included at all, and in which order | `detail::AddHeader` (`src/backends/common/hlsl/codegen_utils/entry_points.cpp`) |
| History of a change | The git commit message |

## Checking a file before you commit it

```bash
grep -nE '//|/\*' src/backends/common/hlsl/builtin/*.bytes \
    src/backends/common/hlsl/builtin/bc6_* \
    src/backends/common/hlsl/builtin/bc7_* \
    src/backends/common/hlsl/builtin/bindless_upload* \
    src/backends/common/hlsl/builtin/load_bdls_dxil \
    src/backends/common/hlsl/builtin/hlsl_builtin.hpp
```

No output means the file is clean. `hlsl_builtin_embed.cpp` is generated from these files
(gitignored) and `.dxil` files are binary, so neither is part of the check.

## Editing notes

- Fragments are hand-written HLSL text; there is no generator behind them. Edit them in
  place, keep the existing line-ending style of the file (most are CRLF), and keep the
  code token-identical to what you tested.
- `bindless_upload.bytes` is compiled to `load_bdls.dxil` and the `bc6_*`/`bc7_*` pieces
  to their `.dxil` artifacts with `xmake run lc_compile_builtin`; see the `hlsl` skill for
  the exact invocations. Regenerate the artifact in the same commit as the fragment.
- Under xmake the embedding is per-object (`rules/utils.bin2obj` +
  `LUISA_BIN_2_OBJ`); under CMake the `luisa-compute-backend-hlsl-embed` target rewrites
  `builtin/hlsl_builtin_embed.cpp` and checks it. A new `.bytes`/`.dxil` needs the
  registration steps in `hlsl_builtin.hpp` and `src/backends/common/hlsl/CMakeLists.txt`
  described in that skill.
