/**
 * LuisaCompute NTC (RTX Neural Texture Compression)
 * Pure GPU BCn (BC1, BC3, BC4, BC5, BC7) Block Compression Implementation.
 */

#include "bcn_encoder.h"
#include <libntc/buffer_copy.h>
#include <luisa/core/logging.h>
#include <fstream>
#include <cstring>

namespace luisa::ntc {

#pragma pack(push, 1)
struct DDS_PIXELFORMAT {
    uint32_t dwSize{32};
    uint32_t dwFlags{0x4}; // DDPF_FOURCC
    uint32_t dwFourCC{0};
    uint32_t dwRGBBitCount{0};
    uint32_t dwRBitMask{0};
    uint32_t dwGBitMask{0};
    uint32_t dwBBitMask{0};
    uint32_t dwABitMask{0};
};

struct DDS_HEADER {
    uint32_t dwSize{124};
    uint32_t dwFlags{0x1007}; // DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT
    uint32_t dwHeight{0};
    uint32_t dwWidth{0};
    uint32_t dwPitchOrLinearSize{0};
    uint32_t dwDepth{0};
    uint32_t dwMipMapCount{1};
    uint32_t dwReserved1[11]{};
    DDS_PIXELFORMAT ddspf{};
    uint32_t dwCaps{0x1000}; // DDSCAPS_TEXTURE
    uint32_t dwCaps2{0};
    uint32_t dwCaps3{0};
    uint32_t dwCaps4{0};
    uint32_t dwReserved2{0};
};

struct DDS_HEADER_DXT10 {
    uint32_t dxgiFormat{98}; // DXGI_FORMAT_BC7_UNORM
    uint32_t resourceDimension{3}; // D3D10_RESOURCE_DIMENSION_TEXTURE2D
    uint32_t miscFlag{0};
    uint32_t arraySize{1};
    uint32_t miscFlags2{0};
};
#pragma pack(pop)

BCnEncoder::BCnEncoder(Device &dev, Stream &st)
    : device(dev), stream(st) {
    _compile_shaders();
}

void BCnEncoder::_compile_shaders() {
    if (_compiled) return;

    // BC1 Kernel (2 uints per block)
    Kernel2D bc1_kernel = [](BufferFloat input, BufferUInt output,
                             UInt width, UInt height, UInt channel_offset) {
        set_block_size(8, 8, 1);
        UInt2 block_id = dispatch_id().xy();
        UInt num_blocks_x = (width + 3u) / 4u;
        UInt num_blocks_y = (height + 3u) / 4u;

        $if (block_id.x < num_blocks_x & block_id.y < num_blocks_y) {
            Float3 min_rgb = make_float3(1.0f);
            Float3 max_rgb = make_float3(0.0f);

            std::array<Var<float3>, 16> colors{};
            for (uint py = 0; py < 4; ++py) {
                for (uint px = 0; px < 4; ++px) {
                    uint p_idx = py * 4 + px;
                    UInt x = min(block_id.x * 4u + px, width - 1u);
                    UInt y = min(block_id.y * 4u + py, height - 1u);
                    UInt in_base = (y * width + x) * 16u + channel_offset;
                    Float3 c = make_float3(clamp(input.read(in_base + 0u), 0.0f, 1.0f),
                                           clamp(input.read(in_base + 1u), 0.0f, 1.0f),
                                           clamp(input.read(in_base + 2u), 0.0f, 1.0f));
                    colors[p_idx] = c;
                    min_rgb = min(min_rgb, c);
                    max_rgb = max(max_rgb, c);
                }
            }

            UInt r0 = cast<uint>(clamp(round(max_rgb.x * 31.0f), 0.0f, 31.0f));
            UInt g0 = cast<uint>(clamp(round(max_rgb.y * 63.0f), 0.0f, 63.0f));
            UInt b0 = cast<uint>(clamp(round(max_rgb.z * 31.0f), 0.0f, 31.0f));
            UInt c0 = (r0 << 11u) | (g0 << 5u) | b0;

            UInt r1 = cast<uint>(clamp(round(min_rgb.x * 31.0f), 0.0f, 31.0f));
            UInt g1 = cast<uint>(clamp(round(min_rgb.y * 63.0f), 0.0f, 63.0f));
            UInt b1 = cast<uint>(clamp(round(min_rgb.z * 31.0f), 0.0f, 31.0f));
            UInt c1 = (r1 << 11u) | (g1 << 5u) | b1;

            $if (c0 <= c1) {
                $if (c0 == c1) {
                    $if (c0 > 0u) { c1 = c0 - 1u; }
                    $else { c0 = 1u; c1 = 0u; };
                }
                $else {
                    UInt tmp = c0; c0 = c1; c1 = tmp;
                };
            };

            Float3 rgb0 = make_float3(cast<float>((c0 >> 11u) & 31u) / 31.0f,
                                      cast<float>((c0 >> 5u) & 63u) / 63.0f,
                                      cast<float>(c0 & 31u) / 31.0f);
            Float3 rgb1 = make_float3(cast<float>((c1 >> 11u) & 31u) / 31.0f,
                                      cast<float>((c1 >> 5u) & 63u) / 63.0f,
                                      cast<float>(c1 & 31u) / 31.0f);
            Float3 rgb2 = (rgb0 * 2.0f + rgb1) * (1.0f / 3.0f);
            Float3 rgb3 = (rgb0 + rgb1 * 2.0f) * (1.0f / 3.0f);

            UInt indices = 0u;
            for (uint p = 0; p < 16; ++p) {
                Float3 col = colors[p];
                Float d0 = dot(col - rgb0, col - rgb0);
                Float d1 = dot(col - rgb1, col - rgb1);
                Float d2 = dot(col - rgb2, col - rgb2);
                Float d3 = dot(col - rgb3, col - rgb3);

                Float min_d = d0;
                UInt best_i = 0u;
                $if (d1 < min_d) { min_d = d1; best_i = 1u; };
                $if (d2 < min_d) { min_d = d2; best_i = 2u; };
                $if (d3 < min_d) { min_d = d3; best_i = 3u; };

                indices = indices | (best_i << (p * 2u));
            }

            UInt block_idx = block_id.y * num_blocks_x + block_id.x;
            output.write(block_idx * 2u + 0u, c0 | (c1 << 16u));
            output.write(block_idx * 2u + 1u, indices);
        };
    };

    // BC4 Kernel (2 uints per block)
    Kernel2D bc4_kernel = [](BufferFloat input, BufferUInt output,
                             UInt width, UInt height, UInt channel_offset) {
        set_block_size(8, 8, 1);
        UInt2 block_id = dispatch_id().xy();
        UInt num_blocks_x = (width + 3u) / 4u;
        UInt num_blocks_y = (height + 3u) / 4u;

        $if (block_id.x < num_blocks_x & block_id.y < num_blocks_y) {
            Float min_v = 1.0f;
            Float max_v = 0.0f;

            std::array<Var<float>, 16> vals{};
            for (uint py = 0; py < 4; ++py) {
                for (uint px = 0; px < 4; ++px) {
                    uint p_idx = py * 4 + px;
                    UInt x = min(block_id.x * 4u + px, width - 1u);
                    UInt y = min(block_id.y * 4u + py, height - 1u);
                    UInt in_base = (y * width + x) * 16u + channel_offset;
                    Float v = clamp(input.read(in_base), 0.0f, 1.0f);
                    vals[p_idx] = v;
                    min_v = min(min_v, v);
                    max_v = max(max_v, v);
                }
            }

            UInt ep0 = cast<uint>(clamp(round(max_v * 255.0f), 0.0f, 255.0f));
            UInt ep1 = cast<uint>(clamp(round(min_v * 255.0f), 0.0f, 255.0f));

            $if (ep0 <= ep1) {
                $if (ep0 == ep1) {
                    $if (ep0 > 0u) { ep1 = ep0 - 1u; }
                    $else { ep0 = 1u; ep1 = 0u; };
                }
                $else {
                    UInt tmp = ep0; ep0 = ep1; ep1 = tmp;
                };
            };

            Float f0 = cast<float>(ep0);
            Float f1 = cast<float>(ep1);
            std::array<Var<float>, 8> pal{};
            pal[0] = f0;
            pal[1] = f1;
            pal[2] = (f0 * 6.0f + f1 * 1.0f) * (1.0f / 7.0f);
            pal[3] = (f0 * 5.0f + f1 * 2.0f) * (1.0f / 7.0f);
            pal[4] = (f0 * 4.0f + f1 * 3.0f) * (1.0f / 7.0f);
            pal[5] = (f0 * 3.0f + f1 * 4.0f) * (1.0f / 7.0f);
            pal[6] = (f0 * 2.0f + f1 * 5.0f) * (1.0f / 7.0f);
            pal[7] = (f0 * 1.0f + f1 * 6.0f) * (1.0f / 7.0f);

            UInt low_indices = 0u;
            UInt high_indices = 0u;

            for (uint p = 0; p < 16; ++p) {
                Float v = vals[p] * 255.0f;
                Float min_d = abs(v - pal[0]);
                UInt best_i = 0u;
                for (uint i = 1; i < 8; ++i) {
                    Float d = abs(v - pal[i]);
                    $if (d < min_d) { min_d = d; best_i = i; };
                }

                if (p < 5) {
                    low_indices = low_indices | (best_i << (p * 3u));
                } else if (p == 5) {
                    low_indices = low_indices | ((best_i & 1u) << 15u);
                    high_indices = high_indices | ((best_i >> 1u) & 3u);
                } else {
                    high_indices = high_indices | (best_i << (p * 3u - 16u));
                }
            }

            UInt word0 = ep0 | (ep1 << 8u) | ((low_indices & 0xFFFFu) << 16u);
            UInt word1 = high_indices;
            UInt block_idx = block_id.y * num_blocks_x + block_id.x;
            output.write(block_idx * 2u + 0u, word0);
            output.write(block_idx * 2u + 1u, word1);
        };
    };

    // BC5 Kernel (4 uints per block)
    Kernel2D bc5_kernel = [](BufferFloat input, BufferUInt output,
                             UInt width, UInt height, UInt ch0, UInt ch1) {
        set_block_size(8, 8, 1);
        UInt2 block_id = dispatch_id().xy();
        UInt num_blocks_x = (width + 3u) / 4u;
        UInt num_blocks_y = (height + 3u) / 4u;

        $if (block_id.x < num_blocks_x & block_id.y < num_blocks_y) {
            auto compress_ch = [&](UInt ch_offset) {
                Float min_v = 1.0f;
                Float max_v = 0.0f;
                std::array<Var<float>, 16> vals{};
                for (uint py = 0; py < 4; ++py) {
                    for (uint px = 0; px < 4; ++px) {
                        uint p_idx = py * 4 + px;
                        UInt x = min(block_id.x * 4u + px, width - 1u);
                        UInt y = min(block_id.y * 4u + py, height - 1u);
                        UInt in_base = (y * width + x) * 16u + ch_offset;
                        Float v = clamp(input.read(in_base), 0.0f, 1.0f);
                        vals[p_idx] = v;
                        min_v = min(min_v, v);
                        max_v = max(max_v, v);
                    }
                }
                UInt ep0 = cast<uint>(clamp(round(max_v * 255.0f), 0.0f, 255.0f));
                UInt ep1 = cast<uint>(clamp(round(min_v * 255.0f), 0.0f, 255.0f));
                $if (ep0 <= ep1) {
                    $if (ep0 == ep1) {
                        $if (ep0 > 0u) { ep1 = ep0 - 1u; }
                        $else { ep0 = 1u; ep1 = 0u; };
                    }
                    $else {
                        UInt tmp = ep0; ep0 = ep1; ep1 = tmp;
                    };
                };
                Float f0 = cast<float>(ep0);
                Float f1 = cast<float>(ep1);
                std::array<Var<float>, 8> pal{};
                pal[0] = f0;
                pal[1] = f1;
                pal[2] = (f0 * 6.0f + f1 * 1.0f) * (1.0f / 7.0f);
                pal[3] = (f0 * 5.0f + f1 * 2.0f) * (1.0f / 7.0f);
                pal[4] = (f0 * 4.0f + f1 * 3.0f) * (1.0f / 7.0f);
                pal[5] = (f0 * 3.0f + f1 * 4.0f) * (1.0f / 7.0f);
                pal[6] = (f0 * 2.0f + f1 * 5.0f) * (1.0f / 7.0f);
                pal[7] = (f0 * 1.0f + f1 * 6.0f) * (1.0f / 7.0f);

                UInt low_indices = 0u;
                UInt high_indices = 0u;
                for (uint p = 0; p < 16; ++p) {
                    Float v = vals[p] * 255.0f;
                    Float min_d = abs(v - pal[0]);
                    UInt best_i = 0u;
                    for (uint i = 1; i < 8; ++i) {
                        Float d = abs(v - pal[i]);
                        $if (d < min_d) { min_d = d; best_i = i; };
                    }
                    if (p < 5) {
                        low_indices = low_indices | (best_i << (p * 3u));
                    } else if (p == 5) {
                        low_indices = low_indices | ((best_i & 1u) << 15u);
                        high_indices = high_indices | ((best_i >> 1u) & 3u);
                    } else {
                        high_indices = high_indices | (best_i << (p * 3u - 16u));
                    }
                }
                return make_uint2(ep0 | (ep1 << 8u) | ((low_indices & 0xFFFFu) << 16u), high_indices);
            };

            UInt2 r_block = compress_ch(ch0);
            UInt2 g_block = compress_ch(ch1);
            UInt block_idx = block_id.y * num_blocks_x + block_id.x;
            output.write(block_idx * 4u + 0u, r_block.x);
            output.write(block_idx * 4u + 1u, r_block.y);
            output.write(block_idx * 4u + 2u, g_block.x);
            output.write(block_idx * 4u + 3u, g_block.y);
        };
    };

    // BC3 Kernel (4 uints per block: BC4 Alpha + BC1 RGB)
    Kernel2D bc3_kernel = [](BufferFloat input, BufferUInt output,
                             UInt width, UInt height, UInt ch_rgb, UInt ch_a) {
        set_block_size(8, 8, 1);
        UInt2 block_id = dispatch_id().xy();
        UInt num_blocks_x = (width + 3u) / 4u;
        UInt num_blocks_y = (height + 3u) / 4u;

        $if (block_id.x < num_blocks_x & block_id.y < num_blocks_y) {
            // 1. Compress Alpha (BC4)
            Float min_a = 1.0f;
            Float max_a = 0.0f;
            std::array<Var<float>, 16> alphas{};
            for (uint py = 0; py < 4; ++py) {
                for (uint px = 0; px < 4; ++px) {
                    uint p_idx = py * 4 + px;
                    UInt x = min(block_id.x * 4u + px, width - 1u);
                    UInt y = min(block_id.y * 4u + py, height - 1u);
                    UInt in_base = (y * width + x) * 16u + ch_a;
                    Float a = clamp(input.read(in_base), 0.0f, 1.0f);
                    alphas[p_idx] = a;
                    min_a = min(min_a, a);
                    max_a = max(max_a, a);
                }
            }

            UInt a_ep0 = cast<uint>(clamp(round(max_a * 255.0f), 0.0f, 255.0f));
            UInt a_ep1 = cast<uint>(clamp(round(min_a * 255.0f), 0.0f, 255.0f));
            $if (a_ep0 <= a_ep1) {
                $if (a_ep0 == a_ep1) {
                    $if (a_ep0 > 0u) { a_ep1 = a_ep0 - 1u; }
                    $else { a_ep0 = 1u; a_ep1 = 0u; };
                }
                $else {
                    UInt tmp = a_ep0; a_ep0 = a_ep1; a_ep1 = tmp;
                };
            };

            Float af0 = cast<float>(a_ep0);
            Float af1 = cast<float>(a_ep1);
            std::array<Var<float>, 8> a_pal{};
            a_pal[0] = af0;
            a_pal[1] = af1;
            a_pal[2] = (af0 * 6.0f + af1 * 1.0f) * (1.0f / 7.0f);
            a_pal[3] = (af0 * 5.0f + af1 * 2.0f) * (1.0f / 7.0f);
            a_pal[4] = (af0 * 4.0f + af1 * 3.0f) * (1.0f / 7.0f);
            a_pal[5] = (af0 * 3.0f + af1 * 4.0f) * (1.0f / 7.0f);
            a_pal[6] = (af0 * 2.0f + af1 * 5.0f) * (1.0f / 7.0f);
            a_pal[7] = (af0 * 1.0f + af1 * 6.0f) * (1.0f / 7.0f);

            UInt a_low = 0u;
            UInt a_high = 0u;
            for (uint p = 0; p < 16; ++p) {
                Float v = alphas[p] * 255.0f;
                Float min_d = abs(v - a_pal[0]);
                UInt best_i = 0u;
                for (uint i = 1; i < 8; ++i) {
                    Float d = abs(v - a_pal[i]);
                    $if (d < min_d) { min_d = d; best_i = i; };
                }
                if (p < 5) {
                    a_low = a_low | (best_i << (p * 3u));
                } else if (p == 5) {
                    a_low = a_low | ((best_i & 1u) << 15u);
                    a_high = a_high | ((best_i >> 1u) & 3u);
                } else {
                    a_high = a_high | (best_i << (p * 3u - 16u));
                }
            }
            UInt a_word0 = a_ep0 | (a_ep1 << 8u) | ((a_low & 0xFFFFu) << 16u);
            UInt a_word1 = a_high;

            // 2. Compress RGB (BC1)
            Float3 min_rgb = make_float3(1.0f);
            Float3 max_rgb = make_float3(0.0f);
            std::array<Var<float3>, 16> colors{};
            for (uint py = 0; py < 4; ++py) {
                for (uint px = 0; px < 4; ++px) {
                    uint p_idx = py * 4 + px;
                    UInt x = min(block_id.x * 4u + px, width - 1u);
                    UInt y = min(block_id.y * 4u + py, height - 1u);
                    UInt in_base = (y * width + x) * 16u + ch_rgb;
                    Float3 c = make_float3(clamp(input.read(in_base + 0u), 0.0f, 1.0f),
                                           clamp(input.read(in_base + 1u), 0.0f, 1.0f),
                                           clamp(input.read(in_base + 2u), 0.0f, 1.0f));
                    colors[p_idx] = c;
                    min_rgb = min(min_rgb, c);
                    max_rgb = max(max_rgb, c);
                }
            }

            UInt r0 = cast<uint>(clamp(round(max_rgb.x * 31.0f), 0.0f, 31.0f));
            UInt g0 = cast<uint>(clamp(round(max_rgb.y * 63.0f), 0.0f, 63.0f));
            UInt b0 = cast<uint>(clamp(round(max_rgb.z * 31.0f), 0.0f, 31.0f));
            UInt c0 = (r0 << 11u) | (g0 << 5u) | b0;

            UInt r1 = cast<uint>(clamp(round(min_rgb.x * 31.0f), 0.0f, 31.0f));
            UInt g1 = cast<uint>(clamp(round(min_rgb.y * 63.0f), 0.0f, 63.0f));
            UInt b1 = cast<uint>(clamp(round(min_rgb.z * 31.0f), 0.0f, 31.0f));
            UInt c1 = (r1 << 11u) | (g1 << 5u) | b1;

            $if (c0 <= c1) {
                $if (c0 == c1) {
                    $if (c0 > 0u) { c1 = c0 - 1u; }
                    $else { c0 = 1u; c1 = 0u; };
                }
                $else {
                    UInt tmp = c0; c0 = c1; c1 = tmp;
                };
            };

            Float3 rgb0 = make_float3(cast<float>((c0 >> 11u) & 31u) / 31.0f,
                                      cast<float>((c0 >> 5u) & 63u) / 63.0f,
                                      cast<float>(c0 & 31u) / 31.0f);
            Float3 rgb1 = make_float3(cast<float>((c1 >> 11u) & 31u) / 31.0f,
                                      cast<float>((c1 >> 5u) & 63u) / 63.0f,
                                      cast<float>(c1 & 31u) / 31.0f);
            Float3 rgb2 = (rgb0 * 2.0f + rgb1) * (1.0f / 3.0f);
            Float3 rgb3 = (rgb0 + rgb1 * 2.0f) * (1.0f / 3.0f);

            UInt rgb_indices = 0u;
            for (uint p = 0; p < 16; ++p) {
                Float3 col = colors[p];
                Float d0 = dot(col - rgb0, col - rgb0);
                Float d1 = dot(col - rgb1, col - rgb1);
                Float d2 = dot(col - rgb2, col - rgb2);
                Float d3 = dot(col - rgb3, col - rgb3);

                Float min_d = d0;
                UInt best_i = 0u;
                $if (d1 < min_d) { min_d = d1; best_i = 1u; };
                $if (d2 < min_d) { min_d = d2; best_i = 2u; };
                $if (d3 < min_d) { min_d = d3; best_i = 3u; };

                rgb_indices = rgb_indices | (best_i << (p * 2u));
            }
            UInt c_word0 = c0 | (c1 << 16u);
            UInt c_word1 = rgb_indices;

            UInt block_idx = block_id.y * num_blocks_x + block_id.x;
            output.write(block_idx * 4u + 0u, a_word0);
            output.write(block_idx * 4u + 1u, a_word1);
            output.write(block_idx * 4u + 2u, c_word0);
            output.write(block_idx * 4u + 3u, c_word1);
        };
    };

    // BC7 Mode 6 Kernel (4 uints per block)
    Kernel2D bc7_kernel = [](BufferFloat input, BufferUInt output,
                             UInt width, UInt height, UInt ch_r, UInt ch_g) {
        set_block_size(8, 8, 1);
        UInt2 block_id = dispatch_id().xy();
        UInt num_blocks_x = (width + 3u) / 4u;
        UInt num_blocks_y = (height + 3u) / 4u;

        $if (block_id.x < num_blocks_x & block_id.y < num_blocks_y) {
            Float3 min_rgb = make_float3(1.0f);
            Float3 max_rgb = make_float3(0.0f);
            std::array<Var<float3>, 16> colors{};

            for (uint py = 0; py < 4; ++py) {
                for (uint px = 0; px < 4; ++px) {
                    uint p_idx = py * 4 + px;
                    UInt x = min(block_id.x * 4u + px, width - 1u);
                    UInt y = min(block_id.y * 4u + py, height - 1u);
                    UInt in_base = (y * width + x) * 16u + ch_r;
                    Float3 c = make_float3(clamp(input.read(in_base + 0u), 0.0f, 1.0f),
                                           clamp(input.read(in_base + 1u), 0.0f, 1.0f),
                                           clamp(input.read(in_base + 2u), 0.0f, 1.0f));
                    colors[p_idx] = c;
                    min_rgb = min(min_rgb, c);
                    max_rgb = max(max_rgb, c);
                }
            }

            UInt r0_7 = cast<uint>(clamp(round(min_rgb.x * 127.0f), 0.0f, 127.0f));
            UInt g0_7 = cast<uint>(clamp(round(min_rgb.y * 127.0f), 0.0f, 127.0f));
            UInt b0_7 = cast<uint>(clamp(round(min_rgb.z * 127.0f), 0.0f, 127.0f));
            UInt a0_7 = 127u;

            UInt r1_7 = cast<uint>(clamp(round(max_rgb.x * 127.0f), 0.0f, 127.0f));
            UInt g1_7 = cast<uint>(clamp(round(max_rgb.y * 127.0f), 0.0f, 127.0f));
            UInt b1_7 = cast<uint>(clamp(round(max_rgb.z * 127.0f), 0.0f, 127.0f));
            UInt a1_7 = 127u;

            Float3 rgb0 = make_float3(cast<float>(r0_7) / 127.0f, cast<float>(g0_7) / 127.0f, cast<float>(b0_7) / 127.0f);
            Float3 rgb1 = make_float3(cast<float>(r1_7) / 127.0f, cast<float>(g1_7) / 127.0f, cast<float>(b1_7) / 127.0f);

            std::array<Var<uint>, 16> indices{};
            for (uint p = 0; p < 16; ++p) {
                Float3 col = colors[p];
                Float min_d = 1e9f;
                UInt best_i = 0u;
                for (uint i = 0; i < 16; ++i) {
                    Float t = cast<float>(i) * (1.0f / 15.0f);
                    Float3 pal_col = rgb0 * (1.0f - t) + rgb1 * t;
                    Float d = dot(col - pal_col, col - pal_col);
                    $if (d < min_d) { min_d = d; best_i = i; };
                }
                indices[p] = best_i;
            }

            // BC7 Mode 6 anchor index 0 MSB must be 0 (< 8). If not, swap endpoints and invert indices.
            $if ((indices[0] & 8u) != 0u) {
                UInt tr = r0_7; r0_7 = r1_7; r1_7 = tr;
                UInt tg = g0_7; g0_7 = g1_7; g1_7 = tg;
                UInt tb = b0_7; b0_7 = b1_7; b1_7 = tb;
                UInt ta = a0_7; a0_7 = a1_7; a1_7 = ta;
                for (uint p = 0; p < 16; ++p) {
                    indices[p] = indices[p] ^ 15u;
                }
            };

            UInt w0 = 0x40u | (r0_7 << 7u) | (r1_7 << 14u) | (g0_7 << 21u) | ((g1_7 & 0xFu) << 28u);
            UInt w1 = (g1_7 >> 4u) | (b0_7 << 3u) | (b1_7 << 10u) | (a0_7 << 17u) | (a1_7 << 24u) | (1u << 31u);
            UInt w2 = 1u | ((indices[0] & 7u) << 1u);
            for (uint p = 1; p < 8; ++p) {
                w2 = w2 | (indices[p] << (p * 4u));
            }
            UInt w3 = 0u;
            for (uint p = 8; p < 16; ++p) {
                w3 = w3 | (indices[p] << ((p - 8u) * 4u));
            }

            UInt block_idx = block_id.y * num_blocks_x + block_id.x;
            output.write(block_idx * 4u + 0u, w0);
            output.write(block_idx * 4u + 1u, w1);
            output.write(block_idx * 4u + 2u, w2);
            output.write(block_idx * 4u + 3u, w3);
        };
    };

    _compressBC1Kernel = device.compile(bc1_kernel);
    _compressBC3Kernel = device.compile(bc3_kernel);
    _compressBC4Kernel = device.compile(bc4_kernel);
    _compressBC5Kernel = device.compile(bc5_kernel);
    _compressBC7Kernel = device.compile(bc7_kernel);
    stream << synchronize();
    _compiled = true;
}

Buffer<uint> BCnEncoder::encode_to_gpu_buffer(const Buffer<float> &input_float_buffer,
                                              uint32_t width, uint32_t height,
                                              uint32_t channel_offset, uint32_t num_channels,
                                              BlockCompressedFormat format) {
    _compile_shaders();
    uint32_t num_blocks_x = (width + 3u) / 4u;
    uint32_t num_blocks_y = (height + 3u) / 4u;
    size_t total_blocks = static_cast<size_t>(num_blocks_x) * num_blocks_y;

    if (format == BlockCompressedFormat::BC1) {
        Buffer<uint> out_buf = device.create_buffer<uint>(total_blocks * 2);
        stream << _compressBC1Kernel(input_float_buffer, out_buf, width, height, channel_offset).dispatch(num_blocks_x, num_blocks_y)
               << synchronize();
        return out_buf;
    } else if (format == BlockCompressedFormat::BC3) {
        Buffer<uint> out_buf = device.create_buffer<uint>(total_blocks * 4);
        stream << _compressBC3Kernel(input_float_buffer, out_buf, width, height, channel_offset, channel_offset + 3u).dispatch(num_blocks_x, num_blocks_y)
               << synchronize();
        return out_buf;
    } else if (format == BlockCompressedFormat::BC4) {
        Buffer<uint> out_buf = device.create_buffer<uint>(total_blocks * 2);
        stream << _compressBC4Kernel(input_float_buffer, out_buf, width, height, channel_offset).dispatch(num_blocks_x, num_blocks_y)
               << synchronize();
        return out_buf;
    } else if (format == BlockCompressedFormat::BC5) {
        Buffer<uint> out_buf = device.create_buffer<uint>(total_blocks * 4);
        stream << _compressBC5Kernel(input_float_buffer, out_buf, width, height, channel_offset, channel_offset + 1).dispatch(num_blocks_x, num_blocks_y)
               << synchronize();
        return out_buf;
    } else { // BC7 or fallback
        Buffer<uint> out_buf = device.create_buffer<uint>(total_blocks * 4);
        stream << _compressBC7Kernel(input_float_buffer, out_buf, width, height, channel_offset, channel_offset + 1).dispatch(num_blocks_x, num_blocks_y)
               << synchronize();
        return out_buf;
    }
}

bool BCnEncoder::encode_and_save_dds(const Buffer<float> &input_float_buffer,
                                     uint32_t width, uint32_t height,
                                     uint32_t channel_offset, uint32_t num_channels,
                                     BlockCompressedFormat format,
                                     const std::filesystem::path &out_path) {
    Buffer<uint> gpu_bcn = encode_to_gpu_buffer(input_float_buffer, width, height,
                                                channel_offset, num_channels, format);
    std::vector<uint32_t> host_bcn(gpu_bcn.size());
    stream << copy_from_device(gpu_bcn, host_bcn.data()) << synchronize();

    std::ofstream dds_file(out_path, std::ios::binary);
    if (!dds_file.is_open()) {
        LUISA_ERROR_WITH_LOCATION("Failed to open output DDS file: {}", out_path.string());
        return false;
    }

    uint32_t magic = 0x20534444; // 'DDS '
    dds_file.write(reinterpret_cast<const char *>(&magic), sizeof(magic));

    DDS_HEADER hdr{};
    hdr.dwFlags |= 0x00080000; // DDSD_LINEARSIZE
    hdr.dwWidth = width;
    hdr.dwHeight = height;
    hdr.dwPitchOrLinearSize = static_cast<uint32_t>(host_bcn.size() * sizeof(uint32_t));

    bool is_bc7 = (format == BlockCompressedFormat::BC7);
    if (format == BlockCompressedFormat::BC1) {
        hdr.ddspf.dwFourCC = 0x31545844; // 'DXT1'
    } else if (format == BlockCompressedFormat::BC3) {
        hdr.ddspf.dwFourCC = 0x35545844; // 'DXT5'
    } else if (format == BlockCompressedFormat::BC4) {
        hdr.ddspf.dwFourCC = 0x55344342; // 'BC4U'
    } else if (format == BlockCompressedFormat::BC5) {
        hdr.ddspf.dwFourCC = 0x55354342; // 'BC5U'
    } else if (is_bc7) {
        hdr.ddspf.dwFourCC = 0x30315844; // 'DX10'
    }

    dds_file.write(reinterpret_cast<const char *>(&hdr), sizeof(hdr));

    if (is_bc7) {
        DDS_HEADER_DXT10 dxt10{};
        dds_file.write(reinterpret_cast<const char *>(&dxt10), sizeof(dxt10));
    }

    dds_file.write(reinterpret_cast<const char *>(host_bcn.data()), host_bcn.size() * sizeof(uint32_t));
    return true;
}

} // namespace luisa::ntc
