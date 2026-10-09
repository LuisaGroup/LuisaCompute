# `example_native_shader` — JSON-driven native shader dispatch

The example compiles one or more *native* shader sources (HLSL, GLSL or CUDA
C++), creates the resources a **dispatch document** declares, loads their inputs
(through the DirectStorage extension when the backend has one, otherwise from
the host), replays the document's **workflow** as a `CommandList` on a `Stream`,
downloads and verifies the results, and — in interactive mode — displays an HDR
image through an ImGui window with a DSL kernel that converts it to the display
destination.

The example is split across four translation units:

| file | owns |
|---|---|
| `native_shader.cpp` | command line, mode orchestration, shader registry glue, ImGui display pass, self test |
| `native_shader_dispatch.{h,cpp}` | the in-memory model of the document, the JSON codec (parse/write) and the semantic validator — **no device, no filesystem** |
| `native_shader_runtime.{h,cpp}` | device-side execution: resource creation, inputs, the workflow, output sinks, verification |
| `native_shader_embedded.h` | the default document plus one embedded `scale` shader per native language |

Schema id `luisa.native_shader.dispatch`, `version: 1`.

```
example_native_shader <backend> <dispatch.json> [shader...] [options]
example_native_shader <backend> <shader...>                  # embedded default workflow
example_native_shader <backend> --self-test                  # codec + execution corpus
example_native_shader <backend> --dump-dispatch FILE          # effective document
example_native_shader --print-schema | --help
```

* `--offline` / `--interactive`, `--frames N`, `--exit-after-frames N`,
  `--output-dir DIR`, `--workdir DIR`, `--strict`, `--no-gui`, `--stats`,
  `--sync-uploads`, `--dump-dispatch FILE`, `--self-test`, `--print-schema`,
  `--entry NAME`, `--push-constant-size N`, `--language LANG`, `--help`.
* The command line's shader paths are appended to the document's `shaders` and
  override an entry with the same name (warning; error with `--strict`). With no
  document at all, a single command-line shader takes over the embedded default
  workflow's `scale` shader.
* Exit code is `0` on success and non-zero on the first failure. The example
  never aborts on a reportable problem: diagnostics go to stderr and the
  document codec collects up to 32 of them before failing.

`xmake run <target> ...` runs the binary with the *target's* output directory as
the working directory, so a relative document path must be given relative to
`bin/<mode>/` (or use an absolute path). `--self-test` locates its corpus by
searching upwards from the working directory, so it works from anywhere.

## Dispatch document (v1)

```json
{
  "version": 1,
  "mode":    { ... },
  "config":  { ... },
  "shaders": [ ... ],
  "resources": [ ... ],
  "workflow": [ ... ]
}
```

* `version` is optional (default `1`); `version > 1` is rejected.
* Unknown keys are warnings (`--strict` turns them into errors), a `handle` key
  is always an error (the document never carries device handles), and a key that
  belongs to a different `cmd` is an error naming the valid set.
* Relative paths resolve against **the directory of the JSON file**, overridable
  with `--workdir DIR`. Relative output sinks get `config.output_dir` (or
  `--output-dir DIR`) as a prefix.
* Every reference to a resource or a shader is a **name**.

### `mode`

| key | type | default | meaning |
|---|---|---|---|
| `type` | `"offline"` \| `"interactive"` | `"offline"` | interactive needs a GUI build |
| `frames` | int ≥ 1 | `1` | offline: how many times the workflow runs |
| `gui` | bool | `true` | interactive: open the ImGui window (`--no-gui` closes it) |
| `window` | object | `{title, 1024, 1024, vsync:true}` | `{title, width, height, vsync}` |
| `display_image` | name | – | interactive: source texture (`float4`/`half4`) |
| `display_destination` | `"auto"` \| name | `"auto"` | interactive: destination texture (a `float` image) |
| `display_scale` | float > 0 | `1.0` | HDR exposure scale fed to the display kernel |
| `display_kernel` | string | `"hdr_to_display"` | registered DSL kernel name |
| `dispatch_per_frame` | bool | `true` | interactive: re-run the workflow every frame |
| `exit_after_frames` | int ≥ 0 | `0` | interactive: `0` = until the window closes |
| `snapshot` | object | `{"every":0,"path":"frame.png"}` | `{every, path}`; `0` = off |

A snapshot writes `config.output_dir / path` as a PNG of the display
destination, every `every` frames.

### `config`

| key | type | default | meaning |
|---|---|---|---|
| `backend` | string | – | optional; the CLI backend wins (a mismatch warns) |
| `default_language` | `hlsl` \| `glsl` \| `cuda_nvrtc` | `hlsl` | language of shaders that omit it |
| `shader_model` | int | `65` | DX shader model (ignored by GLSL/CUDA) |
| `optimize` / `fast_math` / `debug_info` | bool | `true`/`false`/`false` | compiler flags |
| `block_size` | `[x,y,z]` | `[0,0,0]` | `0` = reflect per shader |
| `push_constant_size` | int (omit for null) | reflect | uniform bytes; see the language notes |
| `include_dirs` | [path] | `[]` | extra `#include` search path |
| `output_dir` | path | `"native_shader_output"` | prefix for relative sinks |
| `dstorage` | object | `{enabled:true, staging_buffer_size:67108864, compression:"none"}` | file-input path |
| `strict` | bool | `false` | warnings become errors |
| `log_level` | `verbose`\|`info`\|`warning`\|`error` | `info` | host log level |
| `limits` | object | see `--print-schema` | document budgets |

### `shaders`

| key | type | default | meaning |
|---|---|---|---|
| `name` | string | sanitized file stem | unique |
| `language` | `hlsl`\|`glsl`\|`cuda_nvrtc` | from the extension, else `default_language` | |
| `path` / `source` | string | – | a file *or* inline text (one of them) |
| `source_type` | `"file"`\|`"code"` | inferred | |
| `entry_point` | string | `"main"` | HLSL needs the real name (e.g. `CSMain`) |
| block_size | [x,y,z] | [0,0,0] | required for CUDA without `__launch_bounds__` |
| push_constant_size | int | 0 | 0 = no uniform block / reflect |
| bindings | [object] | [] | manual reflection metadata; see below |
| include_dirs | [path] | [] | |
| optimize/fast_math/debug_info | bool | inherit config | |

 bindings (manual reflection metadata). By default the binding table
  comes from the compiler's reflection of the compiled shader. A non-empty
  bindings list replaces that table before `load()`: each entry is
  {"kind", "space", "register", "array_size", "usage"} with kind one of
  constant_buffer, sampler, structured_buffer, rw_structured_buffer,
  byte_address_buffer, rw_byte_address_buffer, typed_buffer,
  rw_typed_buffer, texture2d, rw_texture2d, texture3d, rw_texture3d,
  acceleration_structure, space/register defaulting to 0, array_size to 1
  and usage to the kind's reflection default. A usage of none is rejected,
  and an SRV/CBV-class kind (everything but the rw_* kinds) can never be
  declared writable. The declared table must describe the shader's real
  interface - `load()` still cross-checks every entry against the compiled
  binary (on the DirectX route it must therefore keep the
  `cbuffer ... : register(b0)` row, which `load()` lifts out as the
  uniform block), and a dispatch's binding index/register still address it.
  A row count differing from the compiler's reflection warns (the declared
  table wins); the count is bounded by
  config.limits.max_bindings_per_dispatch and a duplicate
  (space, register) address is an error. See
  simple_add_manual.json for a runnable declaration.

Language/backend compatibility is validated before compiling: `dx` → HLSL only,
`vk` → HLSL or GLSL, `cuda` → CUDA C++ only. Note that the HLSL *uniform-block
convention* is backend-specific: DirectX feeds the launcher's uniforms through
`cbuffer ... : register(b0)`, while the Vulkan route needs
`[[vk::push_constant]]`.

**One variant per language.** A shader *name* may be declared more than once as
long as every entry states a different `language`: the example then compiles the
variant the selected backend speaks (vk prefers GLSL, then HLSL; dx HLSL; cuda
CUDA C++) and logs the ones it skips. All the variants of a name must describe
the same interface (same bindings, same uniform block size), because a
`native_dispatch` addressing the name uses the first variant's metadata for the
document checks. The samples use this to run unmodified on all three backends:

```json
"shaders": [
  {"name": "scale", "language": "glsl", "path": "shaders/scale.glsl", "entry_point": "main", "push_constant_size": 8},
  {"name": "scale", "language": "hlsl", "path": "shaders/scale.hlsl", "entry_point": "CSMain", "push_constant_size": 8},
  {"name": "scale", "language": "cuda_nvrtc", "path": "shaders/scale.cuda", "entry_point": "scale", "push_constant_size": 8}
]
```

Two entries that cannot be told apart (both without a `language`, or both
claiming the same one) are a duplicate-name error.

### `resources`

Common keys: `name` (unique, `[A-Za-z0-9_.-]{1,64}`), `type`, and an optional
creation-time `input`.

| `type` | keys | created with |
|---|---|---|
| `buffer` | `element` + `count`, **or** `byte_size` | `create_buffer<T>` / `create_byte_buffer` |
| `texture` | `storage`, `size:[w,h]`, `levels` (default 1), optional `element` | `create_image<T>` |
| `volume` | `storage`, `size:[w,h,d]`, `levels` | `create_volume<T>` |
| `bindless_array` | `slot_count`, `slot_type` | `create_bindless_array` |
| `accel` | – | `create_accel` (instances come from `accel_build`) |
| `mesh` | `vertex_buffer`, `triangle_buffer` | `create_mesh` |
| `procedural_primitive` | `aabb_buffer` (optional creation-time range) | `create_procedural_primitive` |

Curve BLAS, motion-blur instances and indirect dispatch buffers are deliberately
**not** resource types here: see "Deliberately unsupported" below.

* `element` for buffers is one of `float float2 float3 float4 uint uint2 uint3
  uint4 int int2 int3 int4 byte triangle aabb` — `triangle` (12 bytes) and
  `aabb` (24 bytes) exist because a mesh's triangle buffer must be a
  `Buffer<Triangle>` and a procedural primitive's buffer a `Buffer<AABB>`.
  A `float3` element occupies 16 bytes (luisa pads the 3-vector), so a
  three-vertex `float3` buffer is 48 bytes, not 36.
* For textures and volumes `element` is the scalar channel type
  (`float`, `uint`, `int`); it defaults to `float` for `float*` storages, `int`
  for `int*`, and `uint` for everything else.
* `storage` names: `byte1..4`, `byte4_srgb`, `short1..4`, `int1..4`,
  `half1..4`, `float1..4`, `r10g10b10a2`, `r11g11b10`. Block-compressed storages
  are rejected with a dedicated message.
  * Resources are created in dependency order (buffers → mesh / procedural
    primitive → accel); a cycle is an error.
  * `element` byte size × `count` must equal `byte_size` when both are given.
  * A resource may not exceed `config.limits.max_resource_bytes` (default 16 GiB;
    a buffer's total, or the sum over all mip levels of a texture/volume). An
    absurd size would otherwise reach a backend allocator that aborts instead of
    reporting, so the codec refuses it with a message naming the size and the
    budget.
  `input` (read before the workflow runs):

```json
{"file": "data/src.bin", "offset": 0, "size": 262144, "compression": "none"}
{"resource": "other", "offset": 0, "size": 1024}
{"inline": {"hex": "0000803f"}}
```

A file input is opened through the DirectStorage extension's stream when the
backend provides one (`dx`, `cuda`); otherwise it is read on the host and
uploaded, with a single warning (`--strict` makes that an error). The file must
exist, be regular and non-empty, `offset` must be below the file size and
`offset + size` within it, and the region may not exceed the resource.

### `workflow`

One JSON object per command; `cmd` is the discriminator and `offset`/`size` are
**bytes** everywhere (`size: 0` means "rest of the resource from `offset`").

The five buffer/texture groups use `offset`/`size`/`src_offset`/`dst_offset`
with a *bare number* for byte offsets and a *3-element array* for a
texture/volume `[x,y,z]` region; `buffer_offset` is always a byte count.

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
| `native_dispatch` | `shader`, `dispatch:[x,y,z]` (threads) or `grid:[x,y,z]`, `bindings`, `uniforms`, `allow_usage_override` |
| `shader_dispatch` | `shader`, `arguments`, `dispatch:[x,y,z]` \| `batched:[[x,y,z],...]` |
| `bindless_array_update` | `resource`, `mode`, `modifications` |
| `mesh_build` | `resource`, `request`, `vertex_buffer`, `vertex_buffer_offset/_size`, `vertex_stride`, `triangle_buffer`, `triangle_buffer_offset/_size` |
| `procedural_primitive_build` | `resource`, `request`, `aabb_buffer`, `aabb_buffer_offset/_size` |
| `accel_build` | `resource`, `instance_count`, `request`, `update_instance_buffer_only`, `modifications` |
| `custom_command` | `uuid` (number or name) + type-specific fields |
| `log` | `message` — host-side `LUISA_INFO`, no device command |
| `synchronize` | `label` — flush the segment and `stream.synchronize()` |

`request` is `prefer_update` | `force_build`. `dispatch` and `grid` are
alternatives — supplying both, or neither, is an error. There is no `indirect`
form: the codec recognises the key and rejects it with a dedicated diagnostic.

`native_dispatch.bindings` entries:

```json
{"index": 0, "resource": "src", "offset": 0, "size": 0, "usage": "read"}
{"register": 1, "space": 0, "resource": "dst", "usage": "write"}
{"resource": "src", "usage": "read"}
```

`index` selects a row of the shader's reflection table; `register`/`space`
selects by declaration; a binding with **neither** is *positional* and fills the
canonical `(space, register)` order of the reflection table. Every binding must
name a resource. On DirectX `register(t0)` and `register(u0)` share a
bind point, so the *index* form is the portable one — the `register` form is
resolved by the declared usage, and a copy needs one selector per argument.
`usage` is `read`, `write` or `read_write` (`none` is rejected: a binding without
a usage does nothing), cross-checked against the reflected
class (an SRV declared `write` is always rejected; a UAV declared `read` needs
`allow_usage_override`). A dispatch may not name the same `index` twice, nor the
same explicit `register`/`space` pair twice.

`native_dispatch.uniforms` entries: `{"type": "float32", "value": 2.0}` or
`{"type": "hex", "hex": "00000040"}`; types are `float32`, `uint32`, `int32`,
`float32x2..4`, `uint32x2..4`, `int32x2..4`. The payload must fit
`config.limits.max_uniform_bytes` and the shader's `push_constant_size` when it
is non-zero.

`shader_dispatch.arguments` entries: `{"kind":"buffer","resource":"src"}`,
`{"kind":"texture","resource":"img","level":0}`,
`{"kind":"bindless_array","resource":"heap"}`,
`{"kind":"accel","resource":"as"}`, `{"kind":"uniform","type":"float32","value":1.0}`.
The argument list must match the kernel's declaration exactly (the count is
checked before encoding and reported as an error; the runtime would otherwise
abort). Uniforms are laid out with the alignment implied by their width, so
prefer scalars for kernel parameters.

`bindless_array_update.modifications` entries:
`{"slot":0,"kind":"buffer","op":"emplace","resource":"buf","offset":0,"size":0}`,
`{"slot":1,"kind":"texture2d","op":"emplace","resource":"img","sampler":{"filter":"linear_linear","address":"repeat"}}`,
`{"slot":2,"op":"remove"}`. One update may not touch a slot twice (the Vulkan
backend asserts on that), so an emplace and a remove of the same slot are two
commands.

`accel_build.modifications` entries:
`{"index":0,"user_id":1,"opaque":true,"visibility":255,"transform":[16 floats],"primitive":"mesh0"}`
(the transform is row-major).

`output` sinks: `{"file":"dst.bin","format":"raw","overwrite":true}` or
`{"discard":true}`. Naming a file is what makes a sink write; `discard: true`
(or a sink without a file) keeps the payload on the host. `format:"png"` is only
valid for a full 2-D `byte4`/`float4` image download. `verify` is available on
`buffer_download` only: `{"kind":"linear","source":"src","k":2.0,"c":1.0,"tolerance":0}`
compares the downloaded floats with `src * k + c`, `{"kind":"copy","source":"other"}`
compares bytes.

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

## Registered DSL kernels

Besides the native shaders the document declares, the example registers these
DSL kernels, which a `shader_dispatch` command can name without declaring a
`shaders` entry (a `native_dispatch` cannot: it needs a native source):

| name | signature | used by |
|---|---|---|
| `hdr_to_display` | `(ImageFloat hdr, ImageFloat display, Float scale, Float width, Float height)` | interactive display pass |
| `fill_hdr_gradient` | `(ImageFloat image, Float width, Float height, Float exposure)` | `scale_interactive.json` |
| `scale_buffer` | `(BufferFloat src, BufferFloat dst, Float k, Float c)` | `all_commands_offline.json` |

## Samples

| file | what it covers |
|---|---|
| `scale_offline.json` | one buffer in, one native dispatch (`grid` form), one verified readback with a raw sink |
| `scale_interactive.json` | an HDR image filled by `shader_dispatch`, displayed through `hdr_to_display` |
| `all_commands_offline.json` | every portable command: uploads (inline, resource-to-resource, file), copies, the three texture/buffer copy pairs, bindless updates, `native_dispatch`, the `custom_command` alias, `shader_dispatch`, mesh/procedural-primitive/accel builds, `log`, `synchronize` and the verifications |

The corpora are self-contained (inline payloads) and declare one shader variant
per native language, so they run unmodified on `dx`, `vk` and `cuda`:

```
xmake run example_native_shader dx   <abs path>/scale_offline.json
xmake run example_native_shader vk   <abs path>/all_commands_offline.json
xmake run example_native_shader cuda <abs path>/scale_offline.json
```

`--self-test` additionally writes a small binary file and runs a file-input
document through it, which is what exercises the dstorage path on the backends
that have one.

## A complete example

`scale_offline.json` in full, annotated:

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

## Path resolution and `native_shader_output/`

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

## Limits

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

## A minimal compute shader in three languages

The smallest useful native dispatch is one read-only buffer, one
read-write buffer and one scalar uniform: `dst[i] = src[i] + c`. The
runnable document is `simple_add.json`; it carries
the three language variants inline (`"source"` + `"source_type": "code"`
instead of `"path"`), so the file is the whole example. All three entries
share the name `add`; the backend picks the variant it speaks (dx →
HLSL, vk → GLSL then HLSL, cuda → CUDA C++) and logs the skipped ones.

HLSL (dx; vk also accepts it through DXC). Resources are declared by
`register`, the uniform block is the root 32-bit constant block, and the
entry point is the real function name:

```hlsl
StructuredBuffer<float> src : register(t0);
RWStructuredBuffer<float> dst : register(u0);
cbuffer Params : register(b0) { float c; };
[numthreads(32, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    dst[tid.x] = src[tid.x] + c;
}
```

GLSL (vk). `(set, binding)` pairs replace registers and the uniform block
is `layout(push_constant)`; the entry point stays `main`:

```glsl
#version 450
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) readonly buffer A { float a[]; } src;
layout(set = 0, binding = 1) buffer B { float b[]; } dst;
layout(push_constant) uniform Push { float c; } params;
void main() {
    uint i = gl_GlobalInvocationID.x;
    dst.b[i] = src.a[i] + params.c;
}
```

CUDA C++ (cuda, NVRTC). The kernel signature is the binding declaration:
`const` pointer = read-only buffer, mutable pointer = read-write buffer,
trailing scalar parameters are the uniforms in declaration order, and
`extern "C"` keeps the name unmangled:

```cuda
extern "C" __global__ void add(const float *src, float *dst, float c) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    dst[i] = src[i] + c;
}
```

The matching document parts: each entry states `"push_constant_size": 4`
(one float) and `"block_size": [32, 1, 1]`; `src` is a 32-element `float`
buffer with an inline hex input (0.0 … 31.0) and `dst` a 32-element `float`
buffer. The workflow is the canonical three commands — a `log`, a
`native_dispatch`, a verified download:

```
{"cmd": "native_dispatch", "shader": "add", "dispatch": [32, 1, 1],
 "bindings": [{"index": 0, "resource": "src", "usage": "read"},
              {"index": 1, "resource": "dst", "usage": "write"}],
 "uniforms": [{"type": "float32", "value": 10.0}]}
```

`dispatch` is in threads, exactly one per element, so no bounds check is
needed. Run it (from the repository root, or pass an absolute document
path — `xmake run` makes `bin/<mode>/` the working directory, see the note
above):

```
bin/release/example_native_shader.exe cuda examples/compute/native_shader_examples/simple_add.json
```

On cuda the log ends with the variant selection, the reflection and the
verified sink:

```
shader 'add': skipping the glsl variant (this backend uses hlsl)
shader 'add': using the cuda_nvrtc variant
compiled a native cuda_nvrtc shader 'add': 878 bytes, workgroup size (32 1 1), 2 reflected binding(s)
binding 0: set 0 register 0 array 1 usage read
binding 1: set 0 register 1 array 1 usage read_write
[workflow 2] wrote 128 bytes to 'native_shader_output\simple_add.bin'
dispatch document executed on 'cuda': 1 frame(s), 3 device command(s), ...
```

The download's verify (`{"kind": "linear", "source": "src", "k": 1.0,
"c": 10.0}`) fails the run unless `dst` holds `src[i] + 10`; the same
document runs unmodified on dx (HLSL variant) and vk (GLSL variant).

## Error catalogue

Every diagnostic is `<json path>: <message>` (or `<resource>: <message>` for the
device-side ones), the process never aborts on a reportable problem, and the
exit code is non-zero. The classes the self test pins down:

| class | example |
|---|---|
| a missing/unreadable/oversized document | `nope.json: no such file` |
| a structural error | `workflow: expected an array, got object`, `mode.frames: expected an integer, got real` |
| an unknown spelling | `workflow[1].cmd: unknown cmd 'explode' (expected one of buffer_upload, ...)`, `resources[0].storage: unknown pixel storage 'float9' (expected one of ...)` |
| a key that belongs to another command | `workflow[0].verify: key is not valid for cmd 'log' (the cmd accepts: message)` |
| an unknown key (warning, error under `--strict`) | `nope: unknown key`, `config.zap: unknown key` |
| a duplicate | `resources[1].name: duplicate resource name 'a'`, `shaders[1].name: duplicate shader name 's' (a name may only be declared once per language)` |
| an unsupported feature | `workflow[1].cmd: 'curve_build' is not supported: no backend implements curve or motion-blur acceleration structures, ...`, `workflow[0].indirect: indirect dispatch is not supported by any backend; use 'dispatch' or 'batched'` |
| a dangling reference | `workflow[0].dst: unknown resource 'b'`, `workflow[0].shader: unknown shader 'nope'` |
| a size/limit violation | `resources[0].count: ... must be non-zero`, `workflow[0].size: 1024 exceeds the 16 bytes of 'a'` |
| a resource over the document budget | `resources[0]: texture 't' is 160000000000 byte(s), which exceeds the limit of 17179869184 byte(s) (config.limits.max_resource_bytes)` |
| a binding that names no resource, or no usage | `workflow[0].bindings[1].resource: a buffer or texture or volume resource name is required`, `workflow[0].bindings[0].usage: a binding needs a usage (read, write or read_write)` |
| a backend/language mismatch | `shaders (scale): backend 'cuda' cannot compile a native hlsl shader (dx: HLSL only; vk: HLSL or GLSL; cuda: CUDA C++ (cuda_nvrtc) only)` |
| a reflected-contract violation (the launcher's own message, verbatim) | `Native shader binding (register 0, space 0) was supplied more than once.` |
| a device refusal | `a PNG sink needs byte4, byte4_srgb or float4 pixels, got 'byte1'`, `the input offset 65536 is not below the size 256 of 'x.bin'` |
| a verification failure | `verification failed: element 1 of 'check2' is 0 but 1 * 3 + 0 is 3 (tolerance 0)` |

`--self-test` runs this catalogue as data: ~40 malformed documents (parse and
semantic stages), 6 documents that must still be accepted as warnings, 8 that
parse and validate but must be refused at run time, and 3 execution corpora.

## Deliberately unsupported

Three features are **never** accepted, on any backend, and asking for one is a
hard error with a message that says why (never a silent skip):

| rejected name | where | message |
|---|---|---|
| `curve_build` | `cmd` | `workflow[i].cmd: 'curve_build' is not supported: no backend implements curve or motion-blur acceleration structures, so this example rejects them everywhere` |
| `motion_instance_build` | `cmd` | same, naming the command |
| `curve` | resource `type` | `resources[i].type: 'curve' is not supported: no backend implements curve or motion-blur acceleration structures, so this example rejects them everywhere` |
| `motion_instance` | resource `type` | same, naming the type |
| `indirect_dispatch_buffer` | resource `type` | `resources[i].type: 'indirect_dispatch_buffer' is not supported: indirect dispatch is unsupported by this example on every backend, so the resource has no use here` |
| `indirect` | `shader_dispatch` key | `workflow[i].indirect: indirect dispatch is not supported by any backend; use 'dispatch' or 'batched'` |

`--print-schema` reports the same two lists as
`unsupported_command_kinds` and `unsupported_resource_types`, and the self test
runs one document per row above. The implementation is gone rather than dormant:
there is no curve, motion-instance or indirect code path left in the example, and
the semantic validator, the writer and the executor no longer mention them.

## Backend notes and known limitations

* Native shaders are **compute only**: the extension has no ray-tracing or
  rasterization injection, so the example's ray-tracing resources are built and
  fed through the ordinary runtime commands. Textures, samplers and acceleration
  structures are *reflected* by `compile()` but `load()` rejects them: buffers
  and uniform blocks are the supported resource classes of this iteration. The
  DSL `shader_dispatch` path has no such restriction, which is why the HDR image
  of the interactive sample is filled by a DSL kernel and only buffers are fed to
  the native shader.
* The CUDA sample uses the `.cuda` extension, not `.cu`: the extension is data
  read at run time, and `.cu` would be claimed by the CUDA language rule (the
  same convention the `lookdev` example uses).
* `dx`: HLSL native shaders (DXIL through DXC) and the DirectStorage extension.
* `vk`: HLSL and GLSL native shaders (SPIR-V). No `DStorageExt` (file inputs
  take the host fallback). A buffer↔texture copy needs a row that is a multiple
  of 256 bytes.
* `cuda`: CUDA C++ native shaders (PTX through NVRTC) and the DirectStorage
  extension. The CUDA codegen has no `uint2 → float2` `cast`, so pixel
  coordinates are converted component-wise.
* A backend without `NativeShaderExt` skips the native shader work with an
  `LUISA_INFO` and still runs the DSL parts (the codec, the DSL kernels and the
  host data paths stay exercised) — the process does not fail.
* `texture_download` results cannot be `verify`-ed in place; round-trip a texture
  through `texture_to_buffer_copy` into a buffer and verify that download
  against the buffer the texture was uploaded from (as the corpus does).
* Interactive mode needs a GUI build (`lc_enable_gui` / `LUISA_COMPUTE_ENABLE_GUI`);
  asking for it without one (or with `--no-gui`) is an error, never a silent
  fallback.
