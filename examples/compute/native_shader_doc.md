# `example_native_shader` — dispatch document and resource format

This note introduces the **dispatch JSON** the `example_native_shader` program
consumes and the **resource format** its `resources` array declares. It is a
practical companion to the exhaustive reference in
[`native_shader_examples/README.md`](native_shader_examples/README.md): that
file is the specification (every key, every diagnostic), this one is the short
tour of the document and of the resources it creates.

The example is split across four translation units; the format lives in two of
them:

| file | owns |
|---|---|
| `native_shader.cpp` | command line, mode orchestration, shader registry glue, ImGui display pass, self test |
| `native_shader_dispatch.{h,cpp}` | the in-memory model of the document, the JSON codec (parse/write) and the semantic validator — **no device, no filesystem** |
| `native_shader_runtime.{h,cpp}` | device-side execution: resource creation, inputs, the workflow, output sinks, verification |
| `native_shader_embedded.h` | the default document plus one embedded `scale` shader per native language |

Schema id `luisa.native_shader.dispatch`, `version: 1`.

```
xmake run example_native_shader <backend> <dispatch.json> [shader...] [options]
xmake run example_native_shader <backend> <shader...>          # embedded default workflow
xmake run example_native_shader <backend> --self-test          # codec + execution corpus
xmake run example_native_shader <backend> <doc.json> --dump-dispatch out.json
xmake run example_native_shader <backend> --print-schema
```

`xmake run <target> ...` runs the binary with the target's output directory as
the working directory (`bin/<mode>/`), so a relative document path is relative
to that directory (use an absolute path otherwise). `--print-schema` prints the
tables this document quotes (command kinds, resource types, buffer elements,
pixel storages, usages, DSL kernels, and the default limit budgets) straight from
the codec, so it can never drift from the validator.

## 1. Document skeleton

```json
{
  "version": 1,
  "mode":      { ... },
  "config":    { ... },
  "shaders":   [ ... ],
  "resources": [ ... ],
  "workflow":  [ ... ]
}
```

* `version` is optional (default `1`); a version greater than `1` is rejected.
* The root keys are exactly the six above. An unknown key is a **warning**;
  `--strict` (or `"config": {"strict": true}`) turns it into an **error**.
* A `handle` key is always an error: the document never carries device handles;
  every reference to a resource or a shader is a **name**.
* A key that exists but belongs to a different command is an **error** naming
  the valid set (a typo is a warning instead).
* Diagnostics are `<json path>: <message>`, the codec collects up to 32 of
  them, the program never aborts on a reportable problem, and the exit code is
  non-zero on the first failure.

### `mode`

| key | type | default | meaning |
|---|---|---|---|
| `type` | `"offline"` \| `"interactive"` | `"offline"` | interactive needs a GUI build |
| `frames` | int ≥ 1 | `1` | offline: how many times the workflow runs |
| `gui` | bool | `true` | interactive: open the ImGui window (`--no-gui` closes it) |
| `window` | `{title, width, height, vsync}` | 1024×1024, vsync | |
| `display_image` | name | – | interactive: the HDR source texture (float4/half4) |
| `display_destination` | `"auto"` \| name | `"auto"` | interactive: destination (a float image) |
| `display_scale` | float > 0 | `1.0` | HDR exposure fed to the display kernel |
| `display_kernel` | string | `"hdr_to_display"` | a registered DSL kernel |
| `dispatch_per_frame` | bool | `true` | re-run the workflow every frame |
| `exit_after_frames` | int ≥ 0 | `0` | interactive: `0` = until the window closes |
| `snapshot` | `{every, path}` | `{0, "frame.png"}` | `every: 0` = off |

### `config`

| key | default | notes |
|---|---|---|
| `backend` | – | optional; the CLI backend wins and a mismatch warns |
| `default_language` | `"hlsl"` | language of shaders that state neither `language` nor a known extension |
| `shader_model` | `65` | DX only (ignored by GLSL / CUDA) |
| `optimize` / `fast_math` / `debug_info` | `true` / `false` / `false` | compiler flags |
| `block_size` | `[0,0,0]` | `0` = reflect per shader |
| `push_constant_size` | `null` | uniform bytes; `null` = reflect per shader |
| `include_dirs` | `[]` | extra `#include` search paths |
| `output_dir` | `"native_shader_output"` | prefix for relative output sinks (§6) |
| `dstorage` | `{enabled:true, staging_buffer_size:67108864, compression:"none"}` | file-input path |
| `strict` | `false` | warnings become errors |
| `log_level` | `"info"` | `verbose` \| `info` \| `warning` \| `error` |
| `limits` | see §7 | document budgets |

## 2. `shaders` — the native sources

A `native_dispatch` needs a compiled native shader; a `shader_dispatch` may
instead name one of the three DSL kernels the example registers
(`hdr_to_display`, `fill_hdr_gradient`, `scale_buffer`).

| key | default | meaning |
|---|---|---|
| `name` | sanitized file stem | unique per name **per language** |
| `language` | inferred, else `default_language` | `hlsl` \| `glsl` \| `cuda_nvrtc` |
| `path` / `source` | – | a file or inline text; exactly one of them |
| `source_type` | inferred (`"file"` if `path`) | `"file"` \| `"code"` |
| `entry_point` | `"main"` | HLSL needs the real name, e.g. `CSMain` |
| `block_size` | `[0,0,0]` | required for CUDA without `__launch_bounds__` |
| `push_constant_size` | `0` | `0` = no uniform block / reflect |
| `include_dirs` | `[]` | |
| `optimize` / `fast_math` / `debug_info` | inherit `config` | |

The language is inferred from the path extension when `language` is omitted
(`.hlsl` → HLSL, `.glsl` → GLSL, `.cuda`/`.cu` → CUDA), else from
`config.default_language`.

The **same name may be declared once per language** as long as every entry
describes the same interface (same bindings, same uniform-block size). The
backend picks the variant it speaks — dx: HLSL, vk: GLSL then HLSL, cuda: CUDA
C++ — and logs the skipped ones. This is how the sample corpora run unmodified
on all three backends:

```json
"shaders": [
  { "name": "scale", "language": "glsl",       "path": "shaders/scale.glsl", "entry_point": "main",   "push_constant_size": 8, "block_size": [64,1,1] },
  { "name": "scale", "language": "hlsl",       "path": "shaders/scale.hlsl", "entry_point": "CSMain", "push_constant_size": 8, "block_size": [64,1,1] },
  { "name": "scale", "language": "cuda_nvrtc", "path": "shaders/scale.cuda", "entry_point": "scale",  "push_constant_size": 8, "block_size": [64,1,1] }
]
```

Backend/language compatibility is validated *before* compiling: dx → HLSL only,
vk → HLSL or GLSL, cuda → CUDA C++ only. Note the uniform-block convention is
backend-specific — DirectX feeds the launcher's uniforms through
`cbuffer ... : register(b0)`, the Vulkan route needs `[[vk::push_constant]]`.
Two entries that cannot be told apart (both without a language, or both claiming
the same one) are a duplicate-name error.

## 3. `resources` — the resource format

Every resource is a JSON object with three common keys — `name` (unique,
`[A-Za-z0-9_.-]{1,64}`), `type`, and an optional creation-time `input` — plus the
type-specific keys below. A resource is created from the *declaration*; nothing
is bound to a dispatch until a command says so, and the workflow can only see
resource **names**.

| `type` | type-specific keys | created with |
|---|---|---|
| `buffer` | `element` **+** `count`, or `byte_size` | `create_buffer<T>` / byte buffer |
| `texture` | `storage`, `size:[w,h]`, `levels` (default `1`), `element` (optional) | `create_image<T>` |
| `volume` | `storage`, `size:[w,h,d]`, `levels`, `element` (optional) | `create_volume<T>` |
| `bindless_array` | `slot_count`, `slot_type` (`multiple` \| `buffer` \| `texture2d` \| `texture3d`, default `multiple`) | bindless array |
| `accel` | – (instances come from `accel_build`) | acceleration structure |
| `mesh` | `vertex_buffer`, `triangle_buffer` | `create_mesh` |
| `procedural_primitive` | `aabb_buffer` (optional creation-time AABB range) | `create_procedural_primitive` |

`curve`, `motion_instance` and `indirect_dispatch_buffer` are deliberately **not**
resource types: they are rejected with a diagnostic explaining why (no backend
implements curve or motion-blur acceleration structures).

### 3.1 `buffer`

`element` + `count`, or `byte_size` alone. The element names and their byte
sizes are exactly the ones the codec's spelling table accepts:

| element | bytes | element | bytes | element | bytes |
|---|---|---|---|---|---|
| `float` / `float2` / `float4` | 4 / 8 / 16 | `uint` / `uint2` / `uint4` | 4 / 8 / 16 | `int` / `int2` / `int4` | 4 / 8 / 16 |
| `float3` | **16** | `uint3` | **16** | `int3` | **16** |
| `byte` | 1 | `triangle` | 12 | `aabb` | 24 |

`float3`/`uint3`/`int3` occupy 16 bytes because Luisa pads the 3-vector, so a
three-vertex `float3` buffer is 48 bytes, not 36. `triangle` and `aabb` exist
because a mesh's triangle buffer must be a `Buffer<Triangle>` and a procedural
primitive's buffer a `Buffer<AABB>`.

```json
{ "name": "src",  "type": "buffer", "element": "float", "count": 64 }
{ "name": "blob", "type": "buffer", "byte_size": 1024 }
```

Validation: `element` without `count` (or a `count` of `0`) is an error; a
buffer with neither `element`+`count` nor `byte_size` has no size; when both
`byte_size` and `element`+`count` are given they must agree exactly
(`element_size * count == byte_size`).

### 3.2 `texture` / `volume`

`storage` is the pixel storage (shared with the texture copy commands), `size`
is `[w,h]` for a texture (z is implicitly 1 — a non-1 z warns) and `[w,h,d]` for
a volume, `levels` is the mip level count (default 1, and at least 1).

`storage` spellings:

```
byte1 byte2 byte4 byte4_srgb
short1 short2 short4
int1 int2 int4
half1 half2 half4
float1 float2 float4
r10g10b10a2 r11g11b10
```

`element` is the **scalar channel type** the image is created with, and it
defaults from the storage: `float` for `float*` storages, `int` for `int*`, and
`uint` for everything else. Block-compressed storages are rejected with a
dedicated message.

```json
{ "name": "hdr", "type": "texture", "storage": "float4", "element": "float",
  "size": [512, 512], "levels": 1 }
{ "name": "vol", "type": "volume", "storage": "float4", "size": [2, 2, 2], "levels": 1 }
```

Validation: a missing `storage`, a zero `levels` or a zero extent are errors.

### 3.3 Ray-tracing and bindless resources

```json
{ "name": "vert",  "type": "buffer", "element": "float4",   "count": 3 },
{ "name": "tri",   "type": "buffer", "element": "triangle", "count": 1 },
{ "name": "aabbs", "type": "buffer", "element": "aabb",     "count": 1 },
{ "name": "mesh0", "type": "mesh", "vertex_buffer": "vert", "triangle_buffer": "tri" },
{ "name": "prim0", "type": "procedural_primitive", "aabb_buffer": "aabbs" },
{ "name": "as",    "type": "accel" },
{ "name": "heap",  "type": "bindless_array", "slot_count": 8, "slot_type": "multiple" }
```

A `mesh`'s two buffers and a `procedural_primitive`'s `aabb_buffer` must name
`buffer` resources, and the procedural primitive's buffer must hold `aabb`
elements. The BLAS itself is built later by the matching `*_build` command.
A `bindless_array` needs `slot_count > 0`.

### 3.4 Creation order and `input`

Resources are created in **dependency order** (buffers → mesh / procedural
primitive → accel); a cycle through `input.resource` references is an error.

`input` names exactly one of three sources, plus optional `offset`/`size`
(`size: 0` = the rest of the source):

| form | meaning |
|---|---|
| `{"inline": {"hex": "..."}}` | bytes decoded from a hex string (a bare hex string is also accepted and canonicalised) |
| `{"file": "src.bin", "offset": N, "size": M}` | a region of a file; `compression` may be `none` \| `gdeflate` |
| `{"resource": "other", "offset": N, "size": M}` | a region of another resource |

```json
"input": { "inline": { "hex": "0000803f00000040" } }
"input": { "file": "selftest_src.bin", "offset": 65536, "size": 16 }
"input": { "resource": "src", "offset": 0, "size": 256 }
```

A file input is opened through the DirectStorage extension's stream when the
backend provides one (dx, cuda), otherwise it is read on the host and uploaded,
with a single warning (`--strict` makes that an error). The file must exist, be
regular and non-empty, the offset must be below the file size and
`offset + size` within it, and the region may not exceed the destination
resource. Naming more than one of `inline`/`file`/`resource` is an error.

## 4. `workflow`

One JSON object per command; `cmd` is the discriminator and **offset/size are
bytes everywhere** (`size: 0` means "rest of the resource from offset"). The
buffer and texture groups use `offset`/`size`/`src_offset`/`dst_offset`, with a
bare number for byte offsets and a 3-element array for a texture/volume
`[x,y,z]` region; `buffer_offset` is always a byte count.

| `cmd` | fields |
|---|---|
| `buffer_upload` | `resource`, `offset`, `size`, `input` |
| `buffer_download` | `resource`, `offset`, `size`, `output`, optional `verify` |
| `buffer_copy` | `src`, `src_offset`, `dst`, `dst_offset`, `size` |
| `texture_upload` | `resource`, `level`, `offset:[3]`, `size:[3]`, `storage`, `input` |
| `texture_download` | `resource`, `level`, `offset:[3]`, `size:[3]`, `storage`, `output` |
| `texture_copy` | `storage`, `src`, `dst`, `src_level`, `dst_level`, `size:[3]`, `src_offset:[3]`, `dst_offset:[3]` |
| `buffer_to_texture_copy` | `buffer`, `buffer_offset`, `texture`, `storage`, `level`, `size:[3]`, `offset:[3]` |
| `texture_to_buffer_copy` | `buffer`, `buffer_offset`, `texture`, `storage`, `level`, `size:[3]`, `offset:[3]` |
| `native_dispatch` | `shader`, `dispatch:[x,y,z]` (threads) **or** `grid:[x,y,z]`, `bindings`, `uniforms`, `allow_usage_override` |
| `shader_dispatch` | `shader`, `arguments`, `dispatch:[x,y,z]` \| `batched:[[x,y,z],...]` |
| `bindless_array_update` | `resource`, `mode`, `modifications` |
| `mesh_build` | `resource`, `request`, `vertex_buffer`, `vertex_buffer_offset`/`_size`, `vertex_stride`, `triangle_buffer`, `triangle_buffer_offset`/`_size` |
| `procedural_primitive_build` | `resource`, `request`, `aabb_buffer`, `aabb_buffer_offset`/`_size` |
| `accel_build` | `resource`, `instance_count`, `request`, `update_instance_buffer_only`, `modifications` |
| `custom_command` | `uuid` (number or name) + type-specific fields |
| `log` | `message` — host-side `LUISA_INFO`, no device command |
| `synchronize` | `label` — flush the segment and `stream.synchronize()` |

`request` is `prefer_update` \| `force_build`. `dispatch` and `grid` are
alternatives — supplying both, or neither, is an error. There is no `indirect`
form: the codec recognises the key and rejects it with a dedicated diagnostic.

### 4.1 `native_dispatch.bindings`

```json
{ "index": 0, "resource": "src", "usage": "read" }
{ "register": 0, "space": 0, "resource": "dst", "usage": "write" }
{ "resource": "src", "usage": "read" }
```

`index` selects a row of the shader's reflection table; `register`/`space`
selects by declaration; a binding with **neither** is *positional* and fills the
canonical `(space, register)` order of the reflection table. On DirectX
`register(t0)` and `register(u0)` share one bind point, so the **index form is
the portable one**; the register form is resolved by the declared usage, and a
copy needs one selector per argument. Every binding must name a resource. `usage`
is `read` \| `write` \| `read_write` (`none` is rejected: a binding without a
usage does nothing) and is cross-checked against the reflected class (an SRV
declared `write` is always rejected; a UAV declared `read` needs
`allow_usage_override: true`). A dispatch may not name the same index twice, nor
the same explicit `register`/`space` pair twice (positional bindings cannot
collide and are resolved by the launcher).

### 4.2 `native_dispatch.uniforms`

```json
{ "type": "float32", "value": 2.0 }
{ "type": "hex", "hex": "00000040" }
```

Types are `float32`, `uint32`, `int32`, `float32x2..4`, `uint32x2..4`,
`int32x2..4`. The payload must fit `config.limits.max_uniform_bytes` and the
shader's `push_constant_size` when that is non-zero.

### 4.3 `shader_dispatch.arguments`

```json
{ "kind": "buffer",         "resource": "src" }
{ "kind": "texture",        "resource": "img", "level": 0 }
{ "kind": "bindless_array", "resource": "heap" }
{ "kind": "accel",          "resource": "as" }
{ "kind": "uniform",        "type": "float32", "value": 1.0 }
```

The argument list must match the kernel's declaration **exactly**: the count is
checked before encoding and reported as an error (the runtime would otherwise
abort). Uniforms are laid out with the alignment implied by their width, so
prefer scalars for kernel parameters.

### 4.4 Update modifications

`bindless_array_update.modifications`:

```json
{ "slot": 0, "kind": "buffer", "op": "emplace", "resource": "buf", "offset": 0, "size": 0 }
{ "slot": 1, "kind": "texture2d", "op": "emplace", "resource": "img",
  "sampler": { "filter": "linear_linear", "address": "repeat" } }
{ "slot": 2, "op": "remove" }
```

`kind` is `buffer` \| `texture2d` \| `texture3d`, `op` is `emplace` \| `remove`,
the sampler is `{filter: point|linear_point|linear_linear|anisotropic,
address: edge|repeat|mirror|zero}`. One command may not touch a slot twice (the
Vulkan backend asserts on that), so an emplace and a remove of the same slot are
two commands.

`accel_build.modifications`:

```json
{ "index": 0, "user_id": 1, "opaque": true, "visibility": 255,
  "transform": [16 floats], "primitive": "mesh0" }
```

The transform is row-major.

### 4.5 Output sinks and verification

An output sink is `{"file": "dst.bin", "format": "raw", "overwrite": true}` or
`{"discard": true}`. **Naming a file is what makes a sink write**; `discard:
true` (or a sink without a file) keeps the payload on the host only.
`format: "png"` is valid only for a full 2-D `byte4`/`byte4_srgb`/`float4` image
download.

`verify` is available on `buffer_download` only:

```json
{ "kind": "linear", "source": "src", "k": 2.0, "c": 1.0, "tolerance": 0 }
{ "kind": "copy",   "source": "other" }
```

`linear` compares the downloaded floats with `src * k + c`, `copy` compares
bytes with another resource. A mismatch fails the run with a message naming the
element and the compared numbers.

### 4.6 `custom_command`

`custom_command` is the escape hatch for extension commands; the example
registers two UUIDs, spellable by name or number:

| name | uuid | fields |
|---|---|---|
| `native_shader_dispatch` | `1536` | the whole `native_dispatch` field set (alias form) |
| `dstorage_read` | `512` | `resource`, `offset`, `input` |

```json
{ "cmd": "custom_command", "uuid": "native_shader_dispatch", "shader": "scale",
  "dispatch": [64,1,1],
  "bindings": [ { "register": 0, "space": 0, "resource": "src", "usage": "read" },
                { "index": 1, "resource": "check2", "usage": "write" } ],
  "uniforms": [ { "type": "float32", "value": 3.0 },
                { "type": "float32", "value": 0.0 } ] }
```

An unknown UUID is rejected by the semantic pass.

## 5. A complete example

[`native_shader_examples/scale_offline.json`](native_shader_examples/scale_offline.json)
in full, annotated:

```json
{
  "version": 1,
  "mode": { "type": "offline", "frames": 1 },
  "config": {
    "default_language": "hlsl",
    "include_dirs": ["shaders"],          // resolved next to the JSON file
    "output_dir": "native_shader_output"  // relative sinks land here
  },
  "shaders": [
    { "name": "scale", "language": "glsl",       "path": "shaders/scale.glsl", "entry_point": "main",   "push_constant_size": 8, "block_size": [64,1,1] },
    { "name": "scale", "language": "hlsl",       "path": "shaders/scale.hlsl", "entry_point": "CSMain", "push_constant_size": 8, "block_size": [64,1,1] },
    { "name": "scale", "language": "cuda_nvrtc", "path": "shaders/scale.cuda", "entry_point": "scale",  "push_constant_size": 8, "block_size": [64,1,1] }
  ],
  "resources": [
    { "name": "src", "type": "buffer", "element": "float", "count": 64,
      "input": { "inline": { "hex": "000000000000803f..." } } },  // 0, 1, 2, ...
    { "name": "dst", "type": "buffer", "element": "float", "count": 64 }
  ],
  "workflow": [
    { "cmd": "log", "message": "scaling 64 elements by 2 and adding 1" },
    { "cmd": "native_dispatch", "shader": "scale", "grid": [1,1,1],
      "bindings": [ { "index": 0, "resource": "src", "usage": "read" },
                    { "index": 1, "resource": "dst", "usage": "write" } ],
      "uniforms": [ { "type": "float32", "value": 2.0 },
                    { "type": "float32", "value": 1.0 } ] },
    { "cmd": "buffer_download", "resource": "dst",
      "output": { "file": "dst.bin", "format": "raw", "overwrite": true },
      "verify": { "kind": "linear", "source": "src", "k": 2.0, "c": 1.0 } }
  ]
}
```

Run on `dx`, the log ends with:

```
resource 'src': buffer (256 byte(s))
resource 'dst': buffer (256 byte(s))
[workflow 0] scaling 64 elements by 2 and adding 1
[workflow 2] wrote 256 bytes to 'native_shader_output\dst.bin'
dispatch document executed on 'dx': 1 frame(s), 3 device command(s),
  per-kind [buffer_download=1, native_dispatch=1, log=1]
```

(`log` counts as a command in the per-kind tally, the inline `input` of `src`
is uploaded during resource creation (not as a workflow command), and the
written path proves the `output_dir` prefix rule.)

### Round-tripping with `--dump-dispatch`

`--dump-dispatch out.json` writes the **effective** document — after CLI
overrides and shader merging — in the codec's canonical form, and exits. The
canonical form states every key, including the defaults, so it doubles as a
machine-readable answer to "what defaults am I getting?":

```jsonc
"config": {
  "push_constant_size": null,          // null = reflect per shader
  "output_dir": "native_shader_output",
  "dstorage": { "enabled": true, "staging_buffer_size": 67108864, "compression": "none" },
  "limits": { "max_resources": 256, ... }
},
"workflow": [
  { "cmd": "native_dispatch", "shader": "scale",
    "dispatch": [0,0,0], "grid": [1,1,1],        // both written, one is zero
    "bindings": [ { "index": 0, "resource": "src", "offset": 0, "size": 0, "usage": "read" }, ... ],
    "allow_usage_override": false },
  { "cmd": "buffer_download", "resource": "dst", "offset": 0, "size": 0,
    "output": { "discard": false, "file": "dst.bin", "format": "raw", "overwrite": true },
    "verify": { "kind": "linear", "source": "src", "k": 2.0, "c": 1.0, "tolerance": 0.0 } }
]
```

The codec round-trip (parse → write → parse) is part of `--self-test`.

## 6. Path resolution and `native_shader_output/`

Paths inside the document are resolved by `PathResolver`
(`native_shader_runtime.h`):

* **inputs** (`shaders[].path`, `input.file`, includes) resolve against the
  document's directory, or `--workdir DIR` when given; absolute paths are kept.
* **output sinks** (`output.file`, `mode.snapshot.path`) get
  `config.output_dir` (or `--output-dir DIR`) prefixed when they are relative —
  the parent directory is created on demand.
* shader sources follow the input rule.

Because `xmake run` sets the working directory to `bin/<mode>/`, the default
`output_dir` of `"native_shader_output"` produces artifacts under
`bin/<mode>/native_shader_output/`. A checkout that ran the example therefore
looks like:

| file | produced by | example size | content |
|---|---|---|---|
| `dst.bin` | `buffer_download` of `dst` in `scale_offline.json` | 256 B | 64 `float`s, `src[i] * 2 + 1` |
| `frame.png` | interactive `mode.snapshot` of `scale_interactive.json` | ~28 KB | the 1024×1024 display destination |
| `selftest_src.bin` | `--self-test` | 256 B | the file-input corpus (the default workflow's `src` loaded from this file at offset 0) |

`dst.bin` is the verified readback (the sink only writes because it names a
file; `discard: true` would have skipped it), `frame.png` is a `float4`
full-image PNG download, and `selftest_src.bin` is what exercises the
DirectStorage path on the backends that have one — the self test rewrites it
(256 zero bytes) and points a *rejected* document at offset 65536 to cover the
bounds check as well.

## 7. Limits, exit codes, deliberately unsupported

`config.limits` bounds the document and is printed by `--print-schema`:

| limit | default |
|---|---|
| `max_document_bytes` | 64 MiB |
| `max_resources` / `max_shaders` / `max_commands` | 256 / 64 / 4096 |
| `max_inline_bytes` / `max_uniform_bytes` | 32 MiB / 65536 |
| `max_bindings_per_dispatch` | 256 |
| `max_resource_bytes` | 16 GiB |
| `max_string_bytes` / `max_errors` / `max_depth` | 1 MiB / 32 / 64 |

`max_resource_bytes` bounds one resource (a buffer's total, or the sum over all
mip levels of a texture/volume). Without it an absurd size would reach a backend
allocator that aborts the process instead of reporting anything — with it the
run fails with e.g. `resources[0]: texture 't' is 160000000000 byte(s), which
exceeds the limit of 17179869184 byte(s) (config.limits.max_resource_bytes)`.

Exit code `0` means success, non-zero means the first reported failure; the
program never aborts on a reportable problem. Error messages are
`<json path>: <message>` (or `<resource>: <message>` for device-side ones), and
the classes the self test pins down are catalogued at the end of
`native_shader_examples/README.md` — missing/oversized documents, structural
errors, unknown spellings, cross-command keys, duplicates, unsupported features,
dangling references, size/limit violations, backend/language mismatches,
reflected-contract violations, device refusals and verification failures.

Three features are never accepted on any backend: **curve BLAS**,
**motion-blur instances** and **indirect dispatch** — the corresponding words
(`curve`, `motion_instance`, `curve_build`, `motion_instance_build`,
`indirect`) are recognised and rejected with a message saying why.
