// native_shader_embedded.h - the example's built-in corpus.
//
// Header only on purpose: the corpus needs no build-system wiring, and every
// on-disk twin is cross-checked by `--self-test` (native_shader.cpp compares
// each table entry with the file it names, so the copies below cannot drift
// silently).
//
// The table holds exact copies of the sample data under
// `examples/compute/native_shader_examples/`; the self-contained shader
// variants and the default dispatch document at the bottom are the only
// entries without an on-disk twin (they inline the shared helper so that the
// default workflow runs from any working directory).
#pragma once

#include <cstddef>
#include <iterator>
#include <string_view>

namespace luisa::native_shader {

// One embedded file: `path` is relative to `examples/compute/`, so it is both
// the provenance record and the lookup key.
struct EmbeddedFile {
    const char *path;
    std::string_view contents;
};

inline constexpr EmbeddedFile kEmbeddedFiles[] = {
    // native_shader_examples/shaders/scale.hlsl
    {"native_shader_examples/shaders/scale.hlsl",
     R"luisa_embedded(// Sample native compute shader source (DirectX / Vulkan, HLSL).
//
// `dst[i] = src[i] * k + c` with a structured buffer pair and a uniform block,
// fed by the launcher's `add_uniform` values:
//   * dx: the `cbuffer Uniforms : register(b0)` block is the root 32-bit
//     constant block (`push_constant_size = 8`).
//   * vk: the same block, bound as the push-constant range.
//
// The body comes from `native_shader_math.h`, found through the dispatch
// document's `config.include_dirs` (and through this file's own directory when
// the source is compiled from a path).
#include "native_shader_math.h"

StructuredBuffer<float> src : register(t0);
RWStructuredBuffer<float> dst : register(u0);
cbuffer Uniforms : register(b0) { float k; float c; };

[numthreads(64, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    dst[tid.x] = luisa_native_shader_transform(src[tid.x], k, c);
}
)luisa_embedded"},
    // native_shader_examples/shaders/scale.glsl
    {"native_shader_examples/shaders/scale.glsl",
     R"luisa_embedded(// Sample native compute shader source (Vulkan, GLSL 450).
//
// The same `dst[i] = src[i] * k + c` as the HLSL sample; the (set, binding)
// pairs are spelled out, and the `layout(push_constant)` block is fed by the
// launcher's `add_uniform` values. `GL_GOOGLE_include_directive` must be
// requested after `#version` and before any `#include`.
#version 450
#extension GL_GOOGLE_include_directive : require
#include "native_shader_math.h"

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) readonly buffer A { float a[]; } src;
layout(set = 0, binding = 1) buffer B { float b[]; } dst;
layout(push_constant) uniform Push { float k; float c; } uniforms;

void main() {
    uint i = gl_GlobalInvocationID.x;
    dst.b[i] = luisa_native_shader_transform(src.a[i], uniforms.k, uniforms.c);
}
)luisa_embedded"},
    // native_shader_examples/shaders/scale.cuda
    {"native_shader_examples/shaders/scale.cuda",
     R"luisa_embedded(// Sample native compute shader source (CUDA C++ for NVRTC).
//
// The kernel signature *is* the binding declaration: `const float *src` is a
// read-only structured buffer, `float *dst` a read-write one (in declaration
// order), and the non-pointer parameters `k`/`c` are scalar kernel parameters
// fed by `add_uniform` in declaration order (`push_constant_size = 8`).
// `extern "C"` keeps the name unmangled, and the launcher's block size must be
// 64x1x1 (`__launch_bounds__` would also work).
//
// The file uses the `.cuda` extension so that neither the build system nor a
// CUDA toolchain rule picks it up: it is data, read by the example at run time.
#include "native_shader_math.h"

extern "C" __global__ void scale(const float *src, float *dst, float k, float c) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    dst[i] = luisa_native_shader_transform(src[i], k, c);
}
)luisa_embedded"},
    // native_shader_examples/shaders/native_shader_math.h
    {"native_shader_examples/shaders/native_shader_math.h",
     R"luisa_embedded(// Shared helper of the sample native shaders, included through the dispatch
// document's `config.include_dirs` (or through the source file's own directory
// for the `FilePath` route). It is plain HLSL / GLSL 450 / CUDA C++ at once.
//
// NVRTC's JIT mode only allows execution-space-annotated functions, so under
// `__CUDACC__` the helper is marked `__host__ __device__` as well.
#if defined(__CUDACC__)
#define LUISA_NATIVE_SHADER_INLINE __host__ __device__ __forceinline__
#else
#define LUISA_NATIVE_SHADER_INLINE
#endif

LUISA_NATIVE_SHADER_INLINE float luisa_native_shader_transform(float x, float k, float c) {
    return x * k + c;
}

#undef LUISA_NATIVE_SHADER_INLINE
)luisa_embedded"},
    // native_shader_examples/scale_offline.json
    {"native_shader_examples/scale_offline.json",
     R"luisa_embedded({
  "version": 1,
  "mode": {
    "type": "offline",
    "frames": 1
  },
  "config": {
    "default_language": "hlsl",
    "include_dirs": [
      "shaders"
    ],
    "output_dir": "native_shader_output"
  },
  "shaders": [
    {
      "name": "scale",
      "language": "glsl",
      "path": "shaders/scale.glsl",
      "entry_point": "main",
      "push_constant_size": 8,
      "block_size": [
        64,
        1,
        1
      ]
    },
    {
      "name": "scale",
      "language": "hlsl",
      "path": "shaders/scale.hlsl",
      "entry_point": "CSMain",
      "push_constant_size": 8,
      "block_size": [
        64,
        1,
        1
      ]
    },
    {
      "name": "scale",
      "language": "cuda_nvrtc",
      "path": "shaders/scale.cuda",
      "entry_point": "scale",
      "push_constant_size": 8,
      "block_size": [
        64,
        1,
        1
      ]
    }
  ],
  "resources": [
    {
      "name": "src",
      "type": "buffer",
      "element": "float",
      "count": 64,
      "input": {
        "inline": {
          "hex": "000000000000803f0000004000004040000080400000a0400000c0400000e0400000004100001041000020410000304100004041000050410000604100007041000080410000884100009041000098410000a0410000a8410000b0410000b8410000c0410000c8410000d0410000d8410000e0410000e8410000f0410000f84100000042000004420000084200000c4200001042000014420000184200001c4200002042000024420000284200002c4200003042000034420000384200003c4200004042000044420000484200004c4200005042000054420000584200005c4200006042000064420000684200006c4200007042000074420000784200007c42"
        }
      }
    },
    {
      "name": "dst",
      "type": "buffer",
      "element": "float",
      "count": 64
    }
  ],
  "workflow": [
    {
      "cmd": "log",
      "message": "scaling 64 elements by 2 and adding 1"
    },
    {
      "cmd": "native_dispatch",
      "shader": "scale",
      "grid": [
        1,
        1,
        1
      ],
      "bindings": [
        {
          "index": 0,
          "resource": "src",
          "usage": "read"
        },
        {
          "index": 1,
          "resource": "dst",
          "usage": "write"
        }
      ],
      "uniforms": [
        {
          "type": "float32",
          "value": 2.0
        },
        {
          "type": "float32",
          "value": 1.0
        }
      ]
    },
    {
      "cmd": "buffer_download",
      "resource": "dst",
      "output": {
        "file": "dst.bin",
        "format": "raw",
        "overwrite": true
      },
      "verify": {
        "kind": "linear",
        "source": "src",
        "k": 2.0,
        "c": 1.0
      }
    }
  ]
}
)luisa_embedded"},
    // native_shader_examples/scale_interactive.json
    {"native_shader_examples/scale_interactive.json",
     R"luisa_embedded({
  "version": 1,
  "mode": {
    "type": "interactive",
    "gui": true,
    "window": {
      "title": "native shader (HDR display)",
      "width": 1024,
      "height": 1024,
      "vsync": true
    },
    "display_image": "hdr",
    "display_destination": "auto",
    "display_scale": 0.125,
    "display_kernel": "hdr_to_display",
    "dispatch_per_frame": false,
    "exit_after_frames": 0,
    "snapshot": {
      "every": 0,
      "path": "frame.png"
    }
  },
  "config": {
    "default_language": "hlsl",
    "output_dir": "native_shader_output"
  },
  "shaders": [],
  "resources": [
    {
      "name": "hdr",
      "type": "texture",
      "storage": "float4",
      "element": "float",
      "size": [
        512,
        512
      ],
      "levels": 1
    }
  ],
  "workflow": [
    {
      "cmd": "log",
      "message": "filling the 512x512 HDR display image"
    },
    {
      "cmd": "shader_dispatch",
      "shader": "fill_hdr_gradient",
      "arguments": [
        {
          "kind": "texture",
          "resource": "hdr"
        },
        {
          "kind": "uniform",
          "type": "float32",
          "value": 512.0
        },
        {
          "kind": "uniform",
          "type": "float32",
          "value": 512.0
        },
        {
          "kind": "uniform",
          "type": "float32",
          "value": 0.125
        }
      ],
      "dispatch": [
        512,
        512,
        1
      ]
    }
  ]
}
)luisa_embedded"},
    // native_shader_examples/all_commands_offline.json
    {"native_shader_examples/all_commands_offline.json",
     R"luisa_embedded({
  "version": 1,
  "mode": {
    "type": "offline",
    "frames": 1
  },
  "config": {
    "default_language": "hlsl",
    "include_dirs": [
      "shaders"
    ],
    "output_dir": "native_shader_output",
    "dstorage": {
      "enabled": true,
      "staging_buffer_size": 67108864,
      "compression": "none"
    }
  },
  "shaders": [
    {
      "name": "scale",
      "language": "glsl",
      "path": "shaders/scale.glsl",
      "entry_point": "main",
      "push_constant_size": 8,
      "block_size": [
        64,
        1,
        1
      ]
    },
    {
      "name": "scale",
      "language": "hlsl",
      "path": "shaders/scale.hlsl",
      "entry_point": "CSMain",
      "push_constant_size": 8,
      "block_size": [
        64,
        1,
        1
      ]
    },
    {
      "name": "scale",
      "language": "cuda_nvrtc",
      "path": "shaders/scale.cuda",
      "entry_point": "scale",
      "push_constant_size": 8,
      "block_size": [
        64,
        1,
        1
      ]
    }
  ],
  "resources": [
    {
      "name": "src",
      "type": "buffer",
      "element": "float",
      "count": 64,
      "input": {
        "inline": {
          "hex": "000000000000803f0000004000004040000080400000a0400000c0400000e0400000004100001041000020410000304100004041000050410000604100007041000080410000884100009041000098410000a0410000a8410000b0410000b8410000c0410000c8410000d0410000d8410000e0410000e8410000f0410000f84100000042000004420000084200000c4200001042000014420000184200001c4200002042000024420000284200002c4200003042000034420000384200003c4200004042000044420000484200004c4200005042000054420000584200005c4200006042000064420000684200006c4200007042000074420000784200007c42"
        }
      }
    },
    {
      "name": "dst",
      "type": "buffer",
      "element": "float",
      "count": 64
    },
    {
      "name": "stage",
      "type": "buffer",
      "element": "float",
      "count": 64
    },
    {
      "name": "check",
      "type": "buffer",
      "element": "float",
      "count": 64
    },
    {
      "name": "check2",
      "type": "buffer",
      "element": "float",
      "count": 64
    },
    {
      "name": "check3",
      "type": "buffer",
      "element": "float",
      "count": 64
    },
    {
      "name": "bytes",
      "type": "buffer",
      "element": "byte",
      "count": 1024,
      "input": {
        "inline": {
          "hex": "00000000000000000000003f0000803f8988883d000000000000003f0000803f8988083e000000000000003f0000803fcdcc4c3e000000000000003f0000803f8988883e000000000000003f0000803fabaaaa3e000000000000003f0000803fcdcccc3e000000000000003f0000803fefeeee3e000000000000003f0000803f8988083f000000000000003f0000803f9a99193f000000000000003f0000803fabaa2a3f000000000000003f0000803fbcbb3b3f000000000000003f0000803fcdcc4c3f000000000000003f0000803fdedd5d3f000000000000003f0000803fefee6e3f000000000000003f0000803f0000803f000000000000003f0000803f00000000abaaaa3e0000003f0000803f8988883dabaaaa3e0000003f0000803f8988083eabaaaa3e0000003f0000803fcdcc4c3eabaaaa3e0000003f0000803f8988883eabaaaa3e0000003f0000803fabaaaa3eabaaaa3e0000003f0000803fcdcccc3eabaaaa3e0000003f0000803fefeeee3eabaaaa3e0000003f0000803f8988083fabaaaa3e0000003f0000803f9a99193fabaaaa3e0000003f0000803fabaa2a3fabaaaa3e0000003f0000803fbcbb3b3fabaaaa3e0000003f0000803fcdcc4c3fabaaaa3e0000003f0000803fdedd5d3fabaaaa3e0000003f0000803fefee6e3fabaaaa3e0000003f0000803f0000803fabaaaa3e0000003f0000803f00000000abaa2a3f0000003f0000803f8988883dabaa2a3f0000003f0000803f8988083eabaa2a3f0000003f0000803fcdcc4c3eabaa2a3f0000003f0000803f8988883eabaa2a3f0000003f0000803fabaaaa3eabaa2a3f0000003f0000803fcdcccc3eabaa2a3f0000003f0000803fefeeee3eabaa2a3f0000003f0000803f8988083fabaa2a3f0000003f0000803f9a99193fabaa2a3f0000003f0000803fabaa2a3fabaa2a3f0000003f0000803fbcbb3b3fabaa2a3f0000003f0000803fcdcc4c3fabaa2a3f0000003f0000803fdedd5d3fabaa2a3f0000003f0000803fefee6e3fabaa2a3f0000003f0000803f0000803fabaa2a3f0000003f0000803f000000000000803f0000003f0000803f8988883d0000803f0000003f0000803f8988083e0000803f0000003f0000803fcdcc4c3e0000803f0000003f0000803f8988883e0000803f0000003f0000803fabaaaa3e0000803f0000003f0000803fcdcccc3e0000803f0000003f0000803fefeeee3e0000803f0000003f0000803f8988083f0000803f0000003f0000803f9a99193f0000803f0000003f0000803fabaa2a3f0000803f0000003f0000803fbcbb3b3f0000803f0000003f0000803fcdcc4c3f0000803f0000003f0000803fdedd5d3f0000803f0000003f0000803fefee6e3f0000803f0000003f0000803f0000803f0000803f0000003f0000803f"
        }
      }
    },
    {
      "name": "img_bytes",
      "type": "buffer",
      "element": "byte",
      "count": 1024
    },
    {
      "name": "img_bytes2",
      "type": "buffer",
      "element": "byte",
      "count": 1024
    },
    {
      "name": "img_a",
      "type": "texture",
      "storage": "float4",
      "element": "float",
      "size": [
        16,
        4
      ],
      "levels": 1,
      "input": {
        "inline": {
          "hex": "00000000000000000000003f0000803f8988883d000000000000003f0000803f8988083e000000000000003f0000803fcdcc4c3e000000000000003f0000803f8988883e000000000000003f0000803fabaaaa3e000000000000003f0000803fcdcccc3e000000000000003f0000803fefeeee3e000000000000003f0000803f8988083f000000000000003f0000803f9a99193f000000000000003f0000803fabaa2a3f000000000000003f0000803fbcbb3b3f000000000000003f0000803fcdcc4c3f000000000000003f0000803fdedd5d3f000000000000003f0000803fefee6e3f000000000000003f0000803f0000803f000000000000003f0000803f00000000abaaaa3e0000003f0000803f8988883dabaaaa3e0000003f0000803f8988083eabaaaa3e0000003f0000803fcdcc4c3eabaaaa3e0000003f0000803f8988883eabaaaa3e0000003f0000803fabaaaa3eabaaaa3e0000003f0000803fcdcccc3eabaaaa3e0000003f0000803fefeeee3eabaaaa3e0000003f0000803f8988083fabaaaa3e0000003f0000803f9a99193fabaaaa3e0000003f0000803fabaa2a3fabaaaa3e0000003f0000803fbcbb3b3fabaaaa3e0000003f0000803fcdcc4c3fabaaaa3e0000003f0000803fdedd5d3fabaaaa3e0000003f0000803fefee6e3fabaaaa3e0000003f0000803f0000803fabaaaa3e0000003f0000803f00000000abaa2a3f0000003f0000803f8988883dabaa2a3f0000003f0000803f8988083eabaa2a3f0000003f0000803fcdcc4c3eabaa2a3f0000003f0000803f8988883eabaa2a3f0000003f0000803fabaaaa3eabaa2a3f0000003f0000803fcdcccc3eabaa2a3f0000003f0000803fefeeee3eabaa2a3f0000003f0000803f8988083fabaa2a3f0000003f0000803f9a99193fabaa2a3f0000003f0000803fabaa2a3fabaa2a3f0000003f0000803fbcbb3b3fabaa2a3f0000003f0000803fcdcc4c3fabaa2a3f0000003f0000803fdedd5d3fabaa2a3f0000003f0000803fefee6e3fabaa2a3f0000003f0000803f0000803fabaa2a3f0000003f0000803f000000000000803f0000003f0000803f8988883d0000803f0000003f0000803f8988083e0000803f0000003f0000803fcdcc4c3e0000803f0000003f0000803f8988883e0000803f0000003f0000803fabaaaa3e0000803f0000003f0000803fcdcccc3e0000803f0000003f0000803fefeeee3e0000803f0000003f0000803f8988083f0000803f0000003f0000803f9a99193f0000803f0000003f0000803fabaa2a3f0000803f0000003f0000803fbcbb3b3f0000803f0000003f0000803fcdcc4c3f0000803f0000003f0000803fdedd5d3f0000803f0000003f0000803fefee6e3f0000803f0000003f0000803f0000803f0000803f0000003f0000803f"
        }
      }
    },
    {
      "name": "img_b",
      "type": "texture",
      "storage": "float4",
      "element": "float",
      "size": [
        16,
        4
      ],
      "levels": 1
    },
    {
      "name": "vol_a",
      "type": "volume",
      "storage": "float4",
      "element": "float",
      "size": [
        2,
        2,
        2
      ],
      "levels": 1,
      "input": {
        "inline": {
          "hex": "0000000000000000000000000000803f0000803f00000000000000000000803f000000000000803f000000000000803f0000803f0000803f000000000000803f00000000000000000000803f0000803f0000803f000000000000803f0000803f000000000000803f0000803f0000803f0000803f0000803f0000803f0000803f"
        }
      }
    },
    {
      "name": "vert",
      "type": "buffer",
      "element": "float4",
      "count": 3,
      "input": {
        "inline": {
          "hex": "0000000000000000000000000000803f0000803f00000000000000000000803f000000000000803f000000000000803f"
        }
      }
    },
    {
      "name": "tri",
    )luisa_embedded"
     R"luisa_embedded(  "type": "buffer",
      "element": "triangle",
      "count": 1,
      "input": {
        "inline": {
          "hex": "000000000100000002000000"
        }
      }
    },
    {
      "name": "aabbs",
      "type": "buffer",
      "element": "aabb",
      "count": 1,
      "input": {
        "inline": {
          "hex": "0000000000000000000000000000803f0000803f0000803f"
        }
      }
    },
    {
      "name": "mesh0",
      "type": "mesh",
      "vertex_buffer": "vert",
      "triangle_buffer": "tri"
    },
    {
      "name": "prim0",
      "type": "procedural_primitive",
      "aabb_buffer": "aabbs"
    },
    {
      "name": "as",
      "type": "accel"
    },
    {
      "name": "heap",
      "type": "bindless_array",
      "slot_count": 8,
      "slot_type": "multiple"
    }
  ],
  "workflow": [
    {
      "cmd": "log",
      "message": "all-commands corpus: buffers"
    },
    {
      "cmd": "buffer_upload",
      "resource": "dst",
      "offset": 0,
      "size": 64,
      "input": {
        "inline": {
          "hex": "0000803f0000803f0000803f0000803f0000803f0000803f0000803f0000803f0000803f0000803f0000803f0000803f0000803f0000803f0000803f0000803f"
        }
      }
    },
    {
      "cmd": "buffer_upload",
      "resource": "stage",
      "offset": 0,
      "size": 256,
      "input": {
        "resource": "src",
        "offset": 0,
        "size": 256
      }
    },
    {
      "cmd": "buffer_copy",
      "src": "src",
      "dst": "dst",
      "src_offset": 0,
      "dst_offset": 0,
      "size": 256
    },
    {
      "cmd": "log",
      "message": "all-commands corpus: textures"
    },
    {
      "cmd": "texture_upload",
      "resource": "img_a",
      "level": 0,
      "offset": [
        0,
        0,
        0
      ],
      "size": [
        16,
        4,
        1
      ],
      "storage": "float4",
      "input": {
        "inline": {
          "hex": "00000000000000000000003f0000803f8988883d000000000000003f0000803f8988083e000000000000003f0000803fcdcc4c3e000000000000003f0000803f8988883e000000000000003f0000803fabaaaa3e000000000000003f0000803fcdcccc3e000000000000003f0000803fefeeee3e000000000000003f0000803f8988083f000000000000003f0000803f9a99193f000000000000003f0000803fabaa2a3f000000000000003f0000803fbcbb3b3f000000000000003f0000803fcdcc4c3f000000000000003f0000803fdedd5d3f000000000000003f0000803fefee6e3f000000000000003f0000803f0000803f000000000000003f0000803f00000000abaaaa3e0000003f0000803f8988883dabaaaa3e0000003f0000803f8988083eabaaaa3e0000003f0000803fcdcc4c3eabaaaa3e0000003f0000803f8988883eabaaaa3e0000003f0000803fabaaaa3eabaaaa3e0000003f0000803fcdcccc3eabaaaa3e0000003f0000803fefeeee3eabaaaa3e0000003f0000803f8988083fabaaaa3e0000003f0000803f9a99193fabaaaa3e0000003f0000803fabaa2a3fabaaaa3e0000003f0000803fbcbb3b3fabaaaa3e0000003f0000803fcdcc4c3fabaaaa3e0000003f0000803fdedd5d3fabaaaa3e0000003f0000803fefee6e3fabaaaa3e0000003f0000803f0000803fabaaaa3e0000003f0000803f00000000abaa2a3f0000003f0000803f8988883dabaa2a3f0000003f0000803f8988083eabaa2a3f0000003f0000803fcdcc4c3eabaa2a3f0000003f0000803f8988883eabaa2a3f0000003f0000803fabaaaa3eabaa2a3f0000003f0000803fcdcccc3eabaa2a3f0000003f0000803fefeeee3eabaa2a3f0000003f0000803f8988083fabaa2a3f0000003f0000803f9a99193fabaa2a3f0000003f0000803fabaa2a3fabaa2a3f0000003f0000803fbcbb3b3fabaa2a3f0000003f0000803fcdcc4c3fabaa2a3f0000003f0000803fdedd5d3fabaa2a3f0000003f0000803fefee6e3fabaa2a3f0000003f0000803f0000803fabaa2a3f0000003f0000803f000000000000803f0000003f0000803f8988883d0000803f0000003f0000803f8988083e0000803f0000003f0000803fcdcc4c3e0000803f0000003f0000803f8988883e0000803f0000003f0000803fabaaaa3e0000803f0000003f0000803fcdcccc3e0000803f0000003f0000803fefeeee3e0000803f0000003f0000803f8988083f0000803f0000003f0000803f9a99193f0000803f0000003f0000803fabaa2a3f0000803f0000003f0000803fbcbb3b3f0000803f0000003f0000803fcdcc4c3f0000803f0000003f0000803fdedd5d3f0000803f0000003f0000803fefee6e3f0000803f0000003f0000803f0000803f0000803f0000003f0000803f"
        }
      }
    },
    {
      "cmd": "texture_copy",
      "storage": "float4",
      "src": "img_a",
      "dst": "img_b",
      "src_level": 0,
      "dst_level": 0,
      "size": [
        16,
        4,
        1
      ],
      "src_offset": [
        0,
        0,
        0
      ],
      "dst_offset": [
        0,
        0,
        0
      ]
    },
    {
      "cmd": "texture_download",
      "resource": "img_b",
      "level": 0,
      "offset": [
        0,
        0,
        0
      ],
      "size": [
        16,
        4,
        1
      ],
      "storage": "float4",
      "output": {
        "discard": true
      }
    },
    {
      "cmd": "buffer_to_texture_copy",
      "buffer": "bytes",
      "buffer_offset": 0,
      "texture": "img_b",
      "storage": "float4",
      "level": 0,
      "size": [
        16,
        4,
        1
      ],
      "offset": [
        0,
        0,
        0
      ]
    },
    {
      "cmd": "texture_to_buffer_copy",
      "buffer": "img_bytes",
      "buffer_offset": 0,
      "texture": "img_b",
      "storage": "float4",
      "level": 0,
      "size": [
        16,
        4,
        1
      ],
      "offset": [
        0,
        0,
        0
      ]
    },
    {
      "cmd": "buffer_download",
      "resource": "img_bytes",
      "offset": 0,
      "size": 1024,
      "output": {
        "discard": true
      },
      "verify": {
        "kind": "copy",
        "source": "bytes"
      }
    },
    {
      "cmd": "texture_to_buffer_copy",
      "buffer": "img_bytes2",
      "buffer_offset": 0,
      "texture": "img_a",
      "storage": "float4",
      "level": 0,
      "size": [
        16,
        4,
        1
      ],
      "offset": [
        0,
        0,
        0
      ]
    },
    {
      "cmd": "buffer_download",
      "resource": "img_bytes2",
      "offset": 0,
      "size": 1024,
      "output": {
        "discard": true
      },
      "verify": {
        "kind": "copy",
        "source": "bytes"
      }
    },
    {
      "cmd": "log",
      "message": "all-commands corpus: bindless array"
    },
    {
      "cmd": "bindless_array_update",
      "resource": "heap",
      "mode": "multiple",
      "modifications": [
        {
          "slot": 0,
          "kind": "buffer",
          "op": "emplace",
          "resource": "src",
          "offset": 0,
          "size": 256
        },
        {
          "slot": 1,
          "kind": "texture2d",
          "op": "emplace",
          "resource": "img_a",
          "sampler": {
            "filter": "linear_linear",
            "address": "repeat"
          }
        },
        {
          "slot": 2,
          "kind": "buffer",
          "op": "emplace",
          "resource": "dst",
          "offset": 0,
          "size": 256
        }
      ]
    },
    {
      "cmd": "bindless_array_update",
      "resource": "heap",
      "mode": "multiple",
      "modifications": [
        {
          "slot": 1,
          "op": "remove"
        }
      ]
    },
    {
      "cmd": "log",
      "message": "all-commands corpus: dispatches"
    },
    {
      "cmd": "native_dispatch",
      "shader": "scale",
      "grid": [
        1,
        1,
        1
      ],
      "bindings": [
        {
          "index": 0,
          "resource": "src",
          "usage": "read"
        },
        {
          "index": 1,
          "resource": "check",
          "usage": "write"
        }
      ],
      "uniforms": [
        {
          "type": "float32",
          "value": 2.0
        },
        {
          "type": "float32",
          "value": 1.0
        }
      ]
    },
    {
      "cmd": "custom_command",
      "uuid": "native_shader_dispatch",
      "shader": "scale",
      "dispatch": [
        64,
        1,
        1
      ],
      "bindings": [
        {
          "register": 0,
          "space": 0,
          "resource": "src",
          "usage": "read"
        },
        {
          "index": 1,
          "resource": "check2",
          "usage": "write"
        )luisa_embedded"
     R"luisa_embedded(}
      ],
      "uniforms": [
        {
          "type": "float32",
          "value": 3.0
        },
        {
          "type": "float32",
          "value": 0.0
        }
      ]
    },
    {
      "cmd": "shader_dispatch",
      "shader": "scale_buffer",
      "arguments": [
        {
          "kind": "buffer",
          "resource": "src"
        },
        {
          "kind": "buffer",
          "resource": "check3"
        },
        {
          "kind": "uniform",
          "type": "float32",
          "value": 4.0
        },
        {
          "kind": "uniform",
          "type": "float32",
          "value": 0.0
        }
      ],
      "batched": [
        [
          64,
          1,
          1
        ]
      ]
    },
    {
      "cmd": "log",
      "message": "all-commands corpus: acceleration structures"
    },
    {
      "cmd": "mesh_build",
      "resource": "mesh0",
      "request": "prefer_update",
      "vertex_buffer": "vert",
      "vertex_buffer_offset": 0,
      "vertex_buffer_size": 48,
      "vertex_stride": 16,
      "triangle_buffer": "tri",
      "triangle_buffer_offset": 0,
      "triangle_buffer_size": 12
    },
    {
      "cmd": "procedural_primitive_build",
      "resource": "prim0",
      "request": "prefer_update",
      "aabb_buffer": "aabbs",
      "aabb_buffer_offset": 0,
      "aabb_buffer_size": 24
    },
    {
      "cmd": "accel_build",
      "resource": "as",
      "instance_count": 1,
      "request": "force_build",
      "update_instance_buffer_only": false,
      "modifications": [
        {
          "index": 0,
          "user_id": 7,
          "visibility": 255,
          "opaque": true,
          "transform": [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0
          ],
          "primitive": "mesh0"
        }
      ]
    },
    {
      "cmd": "synchronize",
      "label": "builds"
    },
    {
      "cmd": "log",
      "message": "all-commands corpus: verification"
    },
    {
      "cmd": "buffer_download",
      "resource": "check",
      "output": {
        "discard": true
      },
      "verify": {
        "kind": "linear",
        "source": "src",
        "k": 2.0,
        "c": 1.0
      }
    },
    {
      "cmd": "buffer_download",
      "resource": "check2",
      "output": {
        "discard": true
      },
      "verify": {
        "kind": "linear",
        "source": "src",
        "k": 3.0,
        "c": 0.0
      }
    },
    {
      "cmd": "buffer_download",
      "resource": "check3",
      "output": {
        "discard": true
      },
      "verify": {
        "kind": "linear",
        "source": "src",
        "k": 4.0,
        "c": 0.0
      }
    },
    {
      "cmd": "buffer_download",
      "resource": "dst",
      "output": {
        "discard": true
      },
      "verify": {
        "kind": "linear",
        "source": "src",
        "k": 1.0,
        "c": 0.0
      }
    }
  ]
}
)luisa_embedded"},
};

inline constexpr size_t embedded_file_count = std::size(kEmbeddedFiles);

// The embedded copy of `path` (relative to `examples/compute/`), or an empty
// view when the corpus does not carry it.
[[nodiscard]] inline std::string_view embedded_file(std::string_view path) noexcept {
    for (auto &&file : kEmbeddedFiles) {
        if (path == file.path) { return file.contents; }
    }
    return {};
}

// ---------------------------------------------------------------------------
// self-contained shader variants (no #include, so no include directory)
// ---------------------------------------------------------------------------

// The same `dst[i] = src[i] * k + c` as `shaders/scale.hlsl`, with
// `luisa_native_shader_transform` inlined.
inline constexpr std::string_view kEmbeddedDefaultHlslSource =
    R"luisa_embedded(StructuredBuffer<float> src : register(t0);
RWStructuredBuffer<float> dst : register(u0);
cbuffer Uniforms : register(b0) { float k; float c; };
[numthreads(64, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    dst[tid.x] = src[tid.x] * k + c;
}
)luisa_embedded";

// The same maths as `shaders/scale.glsl`, helper inlined.
inline constexpr std::string_view kEmbeddedDefaultGlslSource =
    R"luisa_embedded(#version 450
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) readonly buffer A { float a[]; } src;
layout(set = 0, binding = 1) buffer B { float b[]; } dst;
layout(push_constant) uniform Push { float k; float c; } uniforms;
void main() {
    uint i = gl_GlobalInvocationID.x;
    dst.b[i] = src.a[i] * uniforms.k + uniforms.c;
}
)luisa_embedded";

// The same maths as `shaders/scale.cuda`, helper inlined.
inline constexpr std::string_view kEmbeddedDefaultCudaSource =
    R"luisa_embedded(extern "C" __global__ void scale(const float *src, float *dst, float k, float c) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    dst[i] = src[i] * k + c;
}
)luisa_embedded";

// ---------------------------------------------------------------------------
// the default dispatch document
// ---------------------------------------------------------------------------

// Used when the command line carries shader sources but no dispatch document
// ("no JSON, only shader paths"): the old upload -> native dispatch -> verified
// readback pipeline, expressed as data. The example fills in the shader entry
// from the variant matching the selected backend.
inline constexpr std::string_view kEmbeddedDefaultDocument =
    R"luisa_embedded({
  "version": 1,
  "mode": {
    "type": "offline",
    "frames": 1
  },
  "config": {
    "default_language": "hlsl",
    "output_dir": "native_shader_output"
  },
  "shaders": [],
  "resources": [
    {
      "name": "src",
      "type": "buffer",
      "element": "float",
      "count": 64,
      "input": {
        "inline": {
          "hex": "000000000000803f0000004000004040000080400000a0400000c0400000e0400000004100001041000020410000304100004041000050410000604100007041000080410000884100009041000098410000a0410000a8410000b0410000b8410000c0410000c8410000d0410000d8410000e0410000e8410000f0410000f84100000042000004420000084200000c4200001042000014420000184200001c4200002042000024420000284200002c4200003042000034420000384200003c4200004042000044420000484200004c4200005042000054420000584200005c4200006042000064420000684200006c4200007042000074420000784200007c42"
        }
      }
    },
    {
      "name": "dst",
      "type": "buffer",
      "element": "float",
      "count": 64
    }
  ],
  "workflow": [
    {
      "cmd": "log",
      "message": "embedded default workflow: upload, dispatch, read back"
    },
    {
      "cmd": "native_dispatch",
      "shader": "scale",
      "grid": [
        1,
        1,
        1
      ],
      "bindings": [
        {
          "index": 0,
          "resource": "src",
          "usage": "read"
        },
        {
          "index": 1,
          "resource": "dst",
          "usage": "write"
        }
      ],
      "uniforms": [
        {
          "type": "float32",
          "value": 2.0
        },
        {
          "type": "float32",
          "value": 1.0
        }
      ]
    },
    {
      "cmd": "buffer_download",
      "resource": "dst",
      "output": {
        "file": "dst.bin",
        "format": "raw",
        "overwrite": true
      },
      "verify": {
        "kind": "linear",
        "source": "src",
        "k": 2.0,
        "c": 1.0
      }
    }
  ]
}
)luisa_embedded";

}// namespace luisa::native_shader
