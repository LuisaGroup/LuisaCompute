/**
 * LuisaCompute NTC (RTX Neural Texture Compression)
 * High-performance GPU Inference Engine Implementation.
 */

#include "infer_engine.h"
#include <luisa/core/logging.h>
#include <chrono>

namespace luisa::ntc {

InferEngine::InferEngine(Device &dev, Stream &st)
    : device(dev), stream(st) {}

bool InferEngine::load_container(const NtexContainer &container) {
    const MLP *mlp = container.find_mlp(InferenceWeightType::GenericInt8);
    if (!mlp) {
        mlp = container.find_mlp(InferenceWeightType::GenericFP8);
    }
    if (!mlp || mlp->layers.size() != NTC_MLP_LAYERS) {
        LUISA_ERROR_WITH_LOCATION("Invalid MLP in container: must have 4 layers.");
        return false;
    }

    _gridScale = container.latentShape.gridSizeScale;
    _numFeatures = container.latentShape.numFeatures;
    _numLatentLayers = FeatureGridMath::get_num_layers(_numFeatures);
    _numNeuralMips = static_cast<uint32_t>(container.latents.size());
    _containerWidth = container.width;
    _containerHeight = container.height;
    _colorMips = container.colorMips;

    // 1. Unpack Weights, Scales, and Biases
    std::vector<int> hostWeights(NTC_TOTAL_WEIGHTS, 0);
    std::vector<float> hostScales(NTC_TOTAL_BIASES, 1.0f);
    std::vector<int> hostBiases(NTC_TOTAL_BIASES, 0);

    size_t weightOffset = 0;
    size_t scaleBiasOffset = 0;

    for (size_t layer = 0; layer < NTC_MLP_LAYERS; ++layer) {
        const auto &l = mlp->layers[layer];
        auto wdata = container.get_view_data(l.weightView);
        auto bdata = container.get_view_data(l.biasView);

        size_t num_w = l.inputChannels * l.outputChannels;
        if (l.weightType == MlpDataType::Int8) {
            const auto *ptr = reinterpret_cast<const int8_t *>(wdata.data());
            for (size_t i = 0; i < num_w && i < wdata.size(); ++i) {
                hostWeights[weightOffset + i] = static_cast<int>(ptr[i]);
            }
        } else if (l.weightType == MlpDataType::FloatE4M3) {
            const auto *ptr = reinterpret_cast<const uint8_t *>(wdata.data());
            for (size_t i = 0; i < num_w && i < wdata.size(); ++i) {
                float f = fp8_e4m3_to_float(ptr[i]);
                hostWeights[weightOffset + i] = static_cast<int>(std::round(f * 127.0f));
            }
        }

        if (l.scaleView) {
            auto sdata = container.get_view_data(*l.scaleView);
            const auto *sptr = reinterpret_cast<const float *>(sdata.data());
            for (size_t i = 0; i < l.outputChannels && i * sizeof(float) < sdata.size(); ++i) {
                hostScales[scaleBiasOffset + i] = sptr[i];
            }
        }

        if (l.biasType == MlpDataType::Int32) {
            const auto *bptr = reinterpret_cast<const int32_t *>(bdata.data());
            for (size_t i = 0; i < l.outputChannels && i * sizeof(int32_t) < bdata.size(); ++i) {
                hostBiases[scaleBiasOffset + i] = bptr[i];
            }
        } else if (l.biasType == MlpDataType::Float16) {
            const auto *bptr = reinterpret_cast<const uint16_t *>(bdata.data());
            for (size_t i = 0; i < l.outputChannels && i * sizeof(uint16_t) < bdata.size(); ++i) {
                float f = fp16_to_float(bptr[i]);
                hostBiases[scaleBiasOffset + i] = static_cast<int>(std::round(f));
            }
        }

        weightOffset += num_w;
        scaleBiasOffset += l.outputChannels;
    }

    _weightBuffer = device.create_buffer<int>(hostWeights.size());
    _scaleBuffer = device.create_buffer<float>(hostScales.size());
    _biasBuffer = device.create_buffer<int>(hostBiases.size());

    stream << _weightBuffer.copy_from(hostWeights.data())
           << _scaleBuffer.copy_from(hostScales.data())
           << _biasBuffer.copy_from(hostBiases.data());

    // 2. Unpack Latents
    std::vector<uint32_t> allLatents;
    std::vector<uint32_t> mipOffsets(_numNeuralMips * 4, 0);
    std::vector<uint32_t> mipDims(_numNeuralMips * 2, 0);

    for (uint32_t mip = 0; mip < _numNeuralMips; ++mip) {
        const auto &latImg = container.latents[mip];
        mipDims[mip * 2] = latImg.width;
        mipDims[mip * 2 + 1] = latImg.height;

        for (uint32_t layer = 0; layer < latImg.layerViews.size(); ++layer) {
            uint32_t v_idx = latImg.layerViews[layer];
            auto vdata = container.decompress_view_data(v_idx);
            size_t num_pixels = static_cast<size_t>(latImg.width) * latImg.height;

            mipOffsets[mip * 4 + layer] = static_cast<uint32_t>(allLatents.size());
            const auto *u16_ptr = reinterpret_cast<const uint16_t *>(vdata.data());

            for (size_t p = 0; p < num_pixels; ++p) {
                uint16_t val = (p * sizeof(uint16_t) < vdata.size()) ? u16_ptr[p] : 0;
                allLatents.push_back(static_cast<uint32_t>(val));
            }
        }
    }

    _latentBuffer = device.create_buffer<uint>(std::max<size_t>(allLatents.size(), 1));
    _latentMipOffsets = device.create_buffer<uint>(std::max<size_t>(mipOffsets.size(), 1));
    _latentMipDims = device.create_buffer<uint>(std::max<size_t>(mipDims.size(), 1));

    if (!allLatents.empty()) {
        stream << _latentBuffer.copy_from(allLatents.data())
               << _latentMipOffsets.copy_from(mipOffsets.data())
               << _latentMipDims.copy_from(mipDims.data());
    }

    // 3. Compile high-throughput DSL Kernel
    Kernel2D decompress_kernel = [&](BufferInt weights, BufferFloat scales, BufferInt biases,
                                     BufferUInt latents, BufferUInt mip_offsets, BufferUInt mip_dims,
                                     BufferFloat out_buffer,
                                     UInt neuralLod, Float positionLod, Float positionScale,
                                     UInt width, UInt height) {
        set_block_size(16, 16, 1);
        UInt2 coord = dispatch_id().xy();
        $if (coord.x < width & coord.y < height) {
            Float2 uv = (make_float2(cast<float>(coord.x), cast<float>(coord.y)) + 0.5f) / make_float2(cast<float>(width), cast<float>(height));

            // Network input vector: 48 values
            Local<int> net_in{48};
            $for(i, 48u) {
                net_in.write(i, 0);
            };

            // High-res Latents (0..15)
            UInt grid_w0 = mip_dims.read(neuralLod * 2u);
            UInt grid_h0 = mip_dims.read(neuralLod * 2u + 1u);
            Float fx0 = uv.x * cast<float>(grid_w0) - 0.5f;
            Float fy0 = uv.y * cast<float>(grid_h0) - 0.5f;
            Int xb0 = cast<int>(floor(fx0));
            Int yb0 = cast<int>(floor(fy0));
            Float dx0 = fx0 - cast<float>(xb0);
            Float dy0 = fy0 - cast<float>(yb0);
            Float w00_0 = (1.0f - dx0) * (1.0f - dy0);
            Float w10_0 = dx0 * (1.0f - dy0);
            Float w01_0 = (1.0f - dx0) * dy0;
            Float w11_0 = dx0 * dy0;

            Int gw0_i = cast<int>(grid_w0);
            Int gh0_i = cast<int>(grid_h0);
            Int x0_0 = (xb0 % gw0_i + gw0_i) % gw0_i;
            Int y0_0 = (yb0 % gh0_i + gh0_i) % gh0_i;
            Int x1_0 = (x0_0 + 1) % gw0_i;
            Int y1_0 = (y0_0 + 1) % gh0_i;

            UInt p00_0 = cast<uint>(y0_0) * grid_w0 + cast<uint>(x0_0);
            UInt p10_0 = cast<uint>(y0_0) * grid_w0 + cast<uint>(x1_0);
            UInt p01_0 = cast<uint>(y1_0) * grid_w0 + cast<uint>(x0_0);
            UInt p11_0 = cast<uint>(y1_0) * grid_w0 + cast<uint>(x1_0);

            uint32_t active_latent_layers = _numLatentLayers;
            for (uint layer = 0; layer < 4; ++layer) {
                if (layer < active_latent_layers) {
                    UInt base_off = mip_offsets.read(neuralLod * 4u + layer);
                    UInt v00 = latents.read(base_off + p00_0);
                    UInt v10 = latents.read(base_off + p10_0);
                    UInt v01 = latents.read(base_off + p01_0);
                    UInt v11 = latents.read(base_off + p11_0);

                    for (uint k = 0; k < 4; ++k) {
                        UInt shift = k * 4u;
                        Float c00 = cast<float>((v00 >> shift) & 0xFu) * (1.0f / 15.0f);
                        Float c10 = cast<float>((v10 >> shift) & 0xFu) * (1.0f / 15.0f);
                        Float c01 = cast<float>((v01 >> shift) & 0xFu) * (1.0f / 15.0f);
                        Float c11 = cast<float>((v11 >> shift) & 0xFu) * (1.0f / 15.0f);
                        Float val = c00 * w00_0 + c10 * w10_0 + c01 * w01_0 + c11 * w11_0;
                        Float scaled = val * (2.0f * NTC_INPUT_SCALE) - NTC_INPUT_SCALE - 0.5f;
                        net_in.write(layer * 4u + k, cast<int>(floor(scaled)));
                    }
                }
            }

            // Low-res Latents (16..31)
            UInt lod1 = neuralLod + 1u;
            UInt grid_w1 = mip_dims.read(lod1 * 2u);
            UInt grid_h1 = mip_dims.read(lod1 * 2u + 1u);
            Float fx1 = uv.x * cast<float>(grid_w1) - 0.5f;
            Float fy1 = uv.y * cast<float>(grid_h1) - 0.5f;
            Int xb1 = cast<int>(floor(fx1));
            Int yb1 = cast<int>(floor(fy1));
            Float dx1 = fx1 - cast<float>(xb1);
            Float dy1 = fy1 - cast<float>(yb1);
            Float w00_1 = (1.0f - dx1) * (1.0f - dy1);
            Float w10_1 = dx1 * (1.0f - dy1);
            Float w01_1 = (1.0f - dx1) * dy1;
            Float w11_1 = dx1 * dy1;

            Int gw1_i = cast<int>(grid_w1);
            Int gh1_i = cast<int>(grid_h1);
            Int x0_1 = (xb1 % gw1_i + gw1_i) % gw1_i;
            Int y0_1 = (yb1 % gh1_i + gh1_i) % gh1_i;
            Int x1_1 = (x0_1 + 1) % gw1_i;
            Int y1_1 = (y0_1 + 1) % gh1_i;

            UInt p00_1 = cast<uint>(y0_1) * grid_w1 + cast<uint>(x0_1);
            UInt p10_1 = cast<uint>(y0_1) * grid_w1 + cast<uint>(x1_1);
            UInt p01_1 = cast<uint>(y1_1) * grid_w1 + cast<uint>(x0_1);
            UInt p11_1 = cast<uint>(y1_1) * grid_w1 + cast<uint>(x1_1);

            for (uint layer = 0; layer < 4; ++layer) {
                if (layer < active_latent_layers) {
                    UInt base_off = mip_offsets.read(lod1 * 4u + layer);
                    UInt v00 = latents.read(base_off + p00_1);
                    UInt v10 = latents.read(base_off + p10_1);
                    UInt v01 = latents.read(base_off + p01_1);
                    UInt v11 = latents.read(base_off + p11_1);

                    for (uint k = 0; k < 4; ++k) {
                        UInt shift = k * 4u;
                        Float c00 = cast<float>((v00 >> shift) & 0xFu) * (1.0f / 15.0f);
                        Float c10 = cast<float>((v10 >> shift) & 0xFu) * (1.0f / 15.0f);
                        Float c01 = cast<float>((v01 >> shift) & 0xFu) * (1.0f / 15.0f);
                        Float c11 = cast<float>((v11 >> shift) & 0xFu) * (1.0f / 15.0f);
                        Float val = c00 * w00_1 + c10 * w10_1 + c01 * w01_1 + c11 * w11_1;
                        Float scaled = val * (2.0f * NTC_INPUT_SCALE) - NTC_INPUT_SCALE - 0.5f;
                        net_in.write(16u + layer * 4u + k, cast<int>(floor(scaled)));
                    }
                }
            }

            // Positional Encoding (32..43)
            Float2 posf = make_float2(cast<float>(coord.x), cast<float>(coord.y)) * positionScale;
            for (uint wave = 0; wave < 3; ++wave) {
                Float fx = fract(posf.x);
                Float fy = fract(posf.y);
                Float fx_shift = fract(posf.x + 0.25f);
                Float fy_shift = fract(posf.y + 0.25f);

                Float4 enc = make_float4(fx * 2.0f - 1.0f, fy * 2.0f - 1.0f,
                                         fx_shift * 2.0f - 1.0f, fy_shift * 2.0f - 1.0f);
                net_in.write(32u + wave * 4u + 0u, cast<int>(enc.x * NTC_INPUT_SCALE));
                net_in.write(32u + wave * 4u + 1u, cast<int>(enc.y * NTC_INPUT_SCALE));
                net_in.write(32u + wave * 4u + 2u, cast<int>(enc.z * NTC_INPUT_SCALE));
                net_in.write(32u + wave * 4u + 3u, cast<int>(enc.w * NTC_INPUT_SCALE));
                posf = posf * 2.0f;
            }

            // LOD parameters (44..47)
            Int ilod = cast<int>(positionLod * NTC_INPUT_SCALE);
            net_in.write(44u, ilod);
            net_in.write(45u, ilod);
            net_in.write(46u, 0);
            net_in.write(47u, 0);

            // MLP Layer 0: 48 -> 64
            Local<int> hid0{64};
            $for(c, 64u) {
                Var<int> acc = biases.read(c);
                $for(r, 48u) {
                    acc += net_in.read(r) * weights.read(c * 48u + r);
                };
                Float res = cast<float>(acc) * scales.read(c);
                Float act = min(res, 3.0f) * clamp(res * 0.333333f + 0.5f, 0.0f, 1.0f);
                hid0.write(c, cast<int>(round(act * 80.0f) - 113.0f));
            };

            // MLP Layer 1: 64 -> 48
            Local<int> hid1{48};
            $for(c, 48u) {
                Var<int> acc = biases.read(64u + c);
                $for(r, 64u) {
                    acc += hid0.read(r) * weights.read(3072u + c * 64u + r);
                };
                Float res = cast<float>(acc) * scales.read(64u + c);
                Float act = min(res, 3.0f) * clamp(res * 0.333333f + 0.5f, 0.0f, 1.0f);
                hid1.write(c, cast<int>(round(act * 80.0f) - 113.0f));
            };

            // MLP Layer 2: 48 -> 32
            Local<int> hid2{32};
            $for(c, 32u) {
                Var<int> acc = biases.read(112u + c);
                $for(r, 48u) {
                    acc += hid1.read(r) * weights.read(6144u + c * 48u + r);
                };
                Float res = cast<float>(acc) * scales.read(112u + c);
                Float act = min(res, 3.0f) * clamp(res * 0.333333f + 0.5f, 0.0f, 1.0f);
                hid2.write(c, cast<int>(round(act * 80.0f) - 113.0f));
            };

            // MLP Layer 3: 32 -> 16
            UInt pixel_idx = (coord.y * width + coord.x) * 16u;
            $for(c, 16u) {
                Var<int> acc = biases.read(144u + c);
                $for(r, 32u) {
                    acc += hid2.read(r) * weights.read(7680u + c * 32u + r);
                };
                Float out_val = cast<float>(acc) * scales.read(144u + c);
                out_buffer.write(pixel_idx + c, out_val);
            };
        };
    };

    _decompressKernel = device.compile(decompress_kernel);
    stream << synchronize();
    _initialized = true;
    return true;
}

Buffer<float> InferEngine::decompress_mip_device(uint32_t color_mip, uint32_t width, uint32_t height) {
    if (!_initialized) {
        LUISA_ERROR_WITH_LOCATION("InferEngine not initialized.");
        return {};
    }

    size_t total_elements = static_cast<size_t>(width) * height * NTC_MLP_OUTPUT_CHANNELS;
    Buffer<float> outBuffer = device.create_buffer<float>(total_elements);

    uint32_t neural_lod = 0;
    float pos_lod = 0.0f;
    float pos_scale = 1.0f;
    if (color_mip < _colorMips.size()) {
        neural_lod = static_cast<uint32_t>(_colorMips[color_mip].latentMip);
        pos_lod = _colorMips[color_mip].positionLod;
        pos_scale = _colorMips[color_mip].positionScale;
    } else {
        neural_lod = static_cast<uint32_t>(FeatureGridMath::lod_to_neural_lod(color_mip, _gridScale, _numNeuralMips));
        uint32_t mw = std::max(1u, _containerWidth >> color_mip);
        uint32_t mh = std::max(1u, _containerHeight >> color_mip);
        uint32_t gw = FeatureGridMath::get_grid_dimension(_containerWidth, neural_lod, _gridScale);
        uint32_t gh = FeatureGridMath::get_grid_dimension(_containerHeight, neural_lod, _gridScale);
        float widthScale = static_cast<float>(gw) / static_cast<float>(mw);
        float heightScale = static_cast<float>(gh) / static_cast<float>(mh);
        pos_scale = 0.5f * std::max(widthScale, heightScale);
        pos_lod = std::max(-1.0f, std::min(1.0f, 0.25f * std::log2(pos_scale)));
    }

    stream << _decompressKernel(_weightBuffer, _scaleBuffer, _biasBuffer,
                                _latentBuffer, _latentMipOffsets, _latentMipDims,
                                outBuffer,
                                static_cast<uint32_t>(neural_lod), pos_lod, pos_scale,
                                width, height).dispatch(width, height)
           << synchronize();

    return outBuffer;
}

std::vector<DecompressedTexture> InferEngine::decompress_all_textures(const NtexContainer &container, uint32_t color_mip) {
    uint32_t mip_w = std::max(1u, container.width >> color_mip);
    uint32_t mip_h = std::max(1u, container.height >> color_mip);

    Buffer<float> out_gpu = decompress_mip_device(color_mip, mip_w, mip_h);

    size_t total_pixels = static_cast<size_t>(mip_w) * mip_h;
    std::vector<float> host_pixels(total_pixels * NTC_MLP_OUTPUT_CHANNELS);
    stream << out_gpu.copy_to(host_pixels.data()) << synchronize();

    std::vector<DecompressedTexture> results;
    results.reserve(container.textures.size());

    for (const auto &tex : container.textures) {
        DecompressedTexture dtex;
        dtex.name = tex.name;
        dtex.width = mip_w;
        dtex.height = mip_h;
        dtex.channels = tex.numChannels;
        dtex.format = tex.channelFormat.value_or(ChannelFormat::UNORM8);
        dtex.colorSpace = tex.rgbColorSpace.value_or(ColorSpace::Linear);
        dtex.bcFormat = tex.bcFormat.value_or(BlockCompressedFormat::None);

        dtex.pixelBytes.resize(total_pixels * tex.numChannels);

        for (size_t p = 0; p < total_pixels; ++p) {
            for (uint32_t ch = 0; ch < tex.numChannels; ++ch) {
                uint32_t src_ch = tex.firstChannel + ch;
                float val = host_pixels[p * NTC_MLP_OUTPUT_CHANNELS + src_ch];
                uint8_t byte_val = static_cast<uint8_t>(std::clamp(std::round(val * 255.0f), 0.0f, 255.0f));
                dtex.pixelBytes[p * tex.numChannels + ch] = byte_val;
            }
        }
        results.push_back(std::move(dtex));
    }

    return results;
}

double InferEngine::benchmark_throughput(uint32_t color_mip, uint32_t width, uint32_t height, uint32_t iterations) {
    if (!_initialized) return 0.0;

    size_t total_elements = static_cast<size_t>(width) * height * NTC_MLP_OUTPUT_CHANNELS;
    Buffer<float> outBuffer = device.create_buffer<float>(total_elements);

    uint32_t neural_lod = 0;
    float pos_lod = 0.0f;
    float pos_scale = 1.0f;
    if (color_mip < _colorMips.size()) {
        neural_lod = static_cast<uint32_t>(_colorMips[color_mip].latentMip);
        pos_lod = _colorMips[color_mip].positionLod;
        pos_scale = _colorMips[color_mip].positionScale;
    } else {
        neural_lod = static_cast<uint32_t>(FeatureGridMath::lod_to_neural_lod(color_mip, _gridScale, _numNeuralMips));
        uint32_t mw = std::max(1u, _containerWidth >> color_mip);
        uint32_t mh = std::max(1u, _containerHeight >> color_mip);
        uint32_t gw = FeatureGridMath::get_grid_dimension(_containerWidth, neural_lod, _gridScale);
        uint32_t gh = FeatureGridMath::get_grid_dimension(_containerHeight, neural_lod, _gridScale);
        float widthScale = static_cast<float>(gw) / static_cast<float>(mw);
        float heightScale = static_cast<float>(gh) / static_cast<float>(mh);
        pos_scale = 0.5f * std::max(widthScale, heightScale);
        pos_lod = std::max(-1.0f, std::min(1.0f, 0.25f * std::log2(pos_scale)));
    }

    // Warm-up
    for (int i = 0; i < 5; ++i) {
        stream << _decompressKernel(_weightBuffer, _scaleBuffer, _biasBuffer,
                                    _latentBuffer, _latentMipOffsets, _latentMipDims,
                                    outBuffer,
                                    static_cast<uint32_t>(neural_lod), pos_lod, pos_scale,
                                    width, height).dispatch(width, height);
    }
    stream << synchronize();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < iterations; ++i) {
        stream << _decompressKernel(_weightBuffer, _scaleBuffer, _biasBuffer,
                                    _latentBuffer, _latentMipOffsets, _latentMipDims,
                                    outBuffer,
                                    static_cast<uint32_t>(neural_lod), pos_lod, pos_scale,
                                    width, height).dispatch(width, height);
    }
    stream << synchronize();
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed_sec = std::chrono::duration<double>(t1 - t0).count();
    double total_texels = static_cast<double>(width) * height * iterations;
    double mtexels_per_sec = (total_texels / elapsed_sec) * 1e-6;

    return mtexels_per_sec;
}

} // namespace luisa::ntc
