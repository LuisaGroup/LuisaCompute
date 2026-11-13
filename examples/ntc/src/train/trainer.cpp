/**
 * LuisaCompute NTC (RTX Neural Texture Compression)
 * GPU Training Engine Implementation: Multi-Mip Sampling, Real Forward-Backward,
 * Fused Adam Optimizer, and Quantization-Aware Export with Exact Bias Compensation.
 */

#include "trainer.h"
#include <libntc/buffer_copy.h>
#include <luisa/core/logging.h>
#include <random>
#include <chrono>
#include <cmath>
#include <cstring>

namespace luisa::ntc {

Trainer::Trainer(Device &dev, Stream &st, const TrainingSettings &settings)
    : device(dev), stream(st), _settings(settings) {
    _compile_shaders();
}

void Trainer::_compile_shaders() {
    if (_compiled) return;

    // Fused Adam for Network Weights & Biases
    Kernel1D net_adam = [](BufferFloat params, BufferFloat grads,
                           BufferFloat m1, BufferFloat m2,
                           Float lr, Float beta1, Float beta2, Float eps, UInt n) {
        UInt idx = dispatch_id().x;
        $if (idx < n) {
            Float g = grads.read(idx);
            Float v1 = m1.read(idx) * beta1 + g * (1.0f - beta1);
            Float v2 = m2.read(idx) * beta2 + g * g * (1.0f - beta2);
            m1.write(idx, v1);
            m2.write(idx, v2);

            Float p = params.read(idx);
            Float step = (v1 / (sqrt(v2) + eps)) * lr;
            params.write(idx, p - step);
            grads.write(idx, 0.0f); // clear grad
        };
    };

    // Fused Adam for Latents with [0, 1] projection
    Kernel1D lat_adam = [](BufferFloat params, BufferFloat grads,
                           BufferFloat m1, BufferFloat m2,
                           Float lr, Float beta1, Float beta2, Float eps, UInt n) {
        UInt idx = dispatch_id().x;
        $if (idx < n) {
            Float g = grads.read(idx);
            Float v1 = m1.read(idx) * beta1 + g * (1.0f - beta1);
            Float v2 = m2.read(idx) * beta2 + g * g * (1.0f - beta2);
            m1.write(idx, v1);
            m2.write(idx, v2);

            Float p = params.read(idx);
            Float step = (v1 / (sqrt(v2) + eps)) * lr;
            Float new_p = clamp(p - step, 0.0f, 1.0f);
            params.write(idx, new_p);
            grads.write(idx, 0.0f); // clear grad
        };
    };

    // Forward-Backward Multi-Mip Regression Step
    Kernel1D train_step = [](BufferFloat weights, BufferFloat biases,
                             BufferFloat weight_grads, BufferFloat bias_grads,
                             BufferFloat latents, BufferFloat latent_grads,
                             BufferFloat ref_mips, BufferFloat4 mip_info,
                             BufferUInt4 latent_info, BufferFloat loss_buf,
                             UInt step_seed, UInt batch_size, UInt num_mips,
                             UInt total_channels, UInt num_features) {
        set_block_size(64, 1, 1);
        UInt tid = dispatch_id().x;
        $if (tid < batch_size) {
            UInt state = (tid ^ (step_seed * 1999u + 61u)) * 1664525u + 1013904223u;
            auto next_u32 = [&]() {
                state = state * 1664525u + 1013904223u;
                return state;
            };

            // Sample random mip level
            UInt mip = next_u32() % num_mips;
            Float4 m0 = mip_info.read(mip * 2u);
            Float4 m1 = mip_info.read(mip * 2u + 1u);
            UInt ref_offset = cast<uint>(m0.x);
            UInt mip_w = cast<uint>(m0.y);
            UInt mip_h = cast<uint>(m0.z);
            UInt neural_lod = cast<uint>(m0.w);
            Float pscale = m1.x;
            Float plod = m1.y;

            // Sample random pixel in selected mip
            UInt px = next_u32() % mip_w;
            UInt py = next_u32() % mip_h;

            UInt ref_base = ref_offset + (py * mip_w + px) * 16u;
            Local<float> ref_pix{16};
            $for(c, 16u) {
                ref_pix.write(c, ref_mips.read(ref_base + c));
            };

            Float2 uv = (make_float2(cast<float>(px), cast<float>(py)) + 0.5f) /
                        make_float2(cast<float>(mip_w), cast<float>(mip_h));

            // High-res Latents
            UInt gw0 = latent_info.read(neural_lod * 2u + 1u).x;
            UInt gh0 = latent_info.read(neural_lod * 2u + 1u).y;
            UInt base_off0 = latent_info.read(neural_lod * 2u).x;

            Float fx0 = uv.x * cast<float>(gw0) - 0.5f;
            Float fy0 = uv.y * cast<float>(gh0) - 0.5f;
            Int xb0 = cast<int>(floor(fx0));
            Int yb0 = cast<int>(floor(fy0));
            Float dx0 = fx0 - cast<float>(xb0);
            Float dy0 = fy0 - cast<float>(yb0);
            Float w00_0 = (1.0f - dx0) * (1.0f - dy0);
            Float w10_0 = dx0 * (1.0f - dy0);
            Float w01_0 = (1.0f - dx0) * dy0;
            Float w11_0 = dx0 * dy0;

            Int gw0_i = cast<int>(gw0);
            Int gh0_i = cast<int>(gh0);
            Int x0_0 = (xb0 % gw0_i + gw0_i) % gw0_i;
            Int y0_0 = (yb0 % gh0_i + gh0_i) % gh0_i;
            Int x1_0 = (x0_0 + 1) % gw0_i;
            Int y1_0 = (y0_0 + 1) % gh0_i;

            UInt p00_0 = cast<uint>(y0_0) * gw0 + cast<uint>(x0_0);
            UInt p10_0 = cast<uint>(y0_0) * gw0 + cast<uint>(x1_0);
            UInt p01_0 = cast<uint>(y1_0) * gw0 + cast<uint>(x0_0);
            UInt p11_0 = cast<uint>(y1_0) * gw0 + cast<uint>(x1_0);

            // Low-res Latents
            UInt lod1 = min(neural_lod + 1u, 3u);
            UInt gw1 = latent_info.read(lod1 * 2u + 1u).x;
            UInt gh1 = latent_info.read(lod1 * 2u + 1u).y;
            UInt base_off1 = latent_info.read(lod1 * 2u).x;

            Float fx1 = uv.x * cast<float>(gw1) - 0.5f;
            Float fy1 = uv.y * cast<float>(gh1) - 0.5f;
            Int xb1 = cast<int>(floor(fx1));
            Int yb1 = cast<int>(floor(fy1));
            Float dx1 = fx1 - cast<float>(xb1);
            Float dy1 = fy1 - cast<float>(yb1);
            Float w00_1 = (1.0f - dx1) * (1.0f - dy1);
            Float w10_1 = dx1 * (1.0f - dy1);
            Float w01_1 = (1.0f - dx1) * dy1;
            Float w11_1 = dx1 * dy1;

            Int gw1_i = cast<int>(gw1);
            Int gh1_i = cast<int>(gh1);
            Int x0_1 = (xb1 % gw1_i + gw1_i) % gw1_i;
            Int y0_1 = (yb1 % gh1_i + gh1_i) % gh1_i;
            Int x1_1 = (x0_1 + 1) % gw1_i;
            Int y1_1 = (y0_1 + 1) % gh1_i;

            UInt p00_1 = cast<uint>(y0_1) * gw1 + cast<uint>(x0_1);
            UInt p10_1 = cast<uint>(y0_1) * gw1 + cast<uint>(x1_1);
            UInt p01_1 = cast<uint>(y1_1) * gw1 + cast<uint>(x0_1);
            UInt p11_1 = cast<uint>(y1_1) * gw1 + cast<uint>(x1_1);

            Local<float> net_in{48};
            $for(i, 48u) { net_in.write(i, 0.0f); };

            for (uint l = 0; l < 4; ++l) {
                UInt l_pix0 = gw0 * gh0;
                UInt l_pix1 = gw1 * gh1;
                for (uint k = 0; k < 4; ++k) {
                    $if (l * 4u + k < num_features) {
                        Float v00_0 = latents.read(base_off0 + (l * l_pix0 + p00_0) * 4u + k);
                        Float v10_0 = latents.read(base_off0 + (l * l_pix0 + p10_0) * 4u + k);
                        Float v01_0 = latents.read(base_off0 + (l * l_pix0 + p01_0) * 4u + k);
                        Float v11_0 = latents.read(base_off0 + (l * l_pix0 + p11_0) * 4u + k);
                        Float val0 = v00_0 * w00_0 + v10_0 * w10_0 + v01_0 * w01_0 + v11_0 * w11_0;
                        net_in.write(l * 4u + k, val0 * 2.0f - 1.0f);

                        Float v00_1 = latents.read(base_off1 + (l * l_pix1 + p00_1) * 4u + k);
                        Float v10_1 = latents.read(base_off1 + (l * l_pix1 + p10_1) * 4u + k);
                        Float v01_1 = latents.read(base_off1 + (l * l_pix1 + p01_1) * 4u + k);
                        Float v11_1 = latents.read(base_off1 + (l * l_pix1 + p11_1) * 4u + k);
                        Float val1 = v00_1 * w00_1 + v10_1 * w10_1 + v01_1 * w01_1 + v11_1 * w11_1;
                        net_in.write(16u + l * 4u + k, val1 * 2.0f - 1.0f);
                    };
                }
            }

            // Positional Encoding (32..43) & LOD (44..47)
            Float2 posf = make_float2(cast<float>(px), cast<float>(py)) * pscale;
            for (uint wave = 0; wave < 3; ++wave) {
                Float fx = fract(posf.x);
                Float fy = fract(posf.y);
                Float fx_shift = fract(posf.x + 0.25f);
                Float fy_shift = fract(posf.y + 0.25f);
                net_in.write(32u + wave * 4u + 0u, fx * 2.0f - 1.0f);
                net_in.write(32u + wave * 4u + 1u, fy * 2.0f - 1.0f);
                net_in.write(32u + wave * 4u + 2u, fx_shift * 2.0f - 1.0f);
                net_in.write(32u + wave * 4u + 3u, fy_shift * 2.0f - 1.0f);
                posf = posf * 2.0f;
            }
            net_in.write(44u, plod);
            net_in.write(45u, plod);
            net_in.write(46u, plod);
            net_in.write(47u, plod);

            // Layer 0 Forward: 48 -> 64
            Local<float> z0{64};
            Local<float> a0{64};
            $for(j, 64u) {
                Float sum = biases.read(j);
                $for(i, 48u) {
                    sum += weights.read(j * 48u + i) * net_in.read(i);
                };
                z0.write(j, sum);
                Float y = min(sum, 3.0f) * clamp(sum * 0.333333f + 0.5f, 0.0f, 1.0f);
                a0.write(j, y);
            };

            // Layer 1 Forward: 64 -> 48
            Local<float> z1{48};
            Local<float> a1{48};
            $for(j, 48u) {
                Float sum = biases.read(64u + j);
                $for(i, 64u) {
                    sum += weights.read(3072u + j * 64u + i) * a0.read(i);
                };
                z1.write(j, sum);
                Float y = min(sum, 3.0f) * clamp(sum * 0.333333f + 0.5f, 0.0f, 1.0f);
                a1.write(j, y);
            };

            // Layer 2 Forward: 48 -> 32
            Local<float> z2{32};
            Local<float> a2{32};
            $for(j, 32u) {
                Float sum = biases.read(112u + j);
                $for(i, 48u) {
                    sum += weights.read(6144u + j * 48u + i) * a1.read(i);
                };
                z2.write(j, sum);
                Float y = min(sum, 3.0f) * clamp(sum * 0.333333f + 0.5f, 0.0f, 1.0f);
                a2.write(j, y);
            };

            // Layer 3 Forward: 32 -> 16
            Local<float> out{16};
            $for(j, 16u) {
                Float sum = biases.read(144u + j);
                $for(i, 32u) {
                    sum += weights.read(7680u + j * 32u + i) * a2.read(i);
                };
                out.write(j, sum);
            };

            // Loss & Output gradient
            Float local_loss = 0.0f;
            Local<float> d_out{16};
            Float norm = 1.0f / cast<float>(batch_size);
            $for(c, 16u) {
                $if (c < total_channels) {
                    Float diff = out.read(c) - ref_pix.read(c);
                    local_loss += diff * diff;
                    d_out.write(c, 2.0f * diff * norm);
                } $else {
                    d_out.write(c, 0.0f);
                };
            };
            loss_buf.atomic(0u).fetch_add(local_loss);

            // Layer 3 Backward: 16 -> 32
            Local<float> da2{32};
            $for(i, 32u) { da2.write(i, 0.0f); };
            $for(j, 16u) {
                Float delta = d_out.read(j);
                bias_grads.atomic(144u + j).fetch_add(delta);
                $for(i, 32u) {
                    weight_grads.atomic(7680u + j * 32u + i).fetch_add(delta * a2.read(i));
                    da2.write(i, da2.read(i) + delta * weights.read(7680u + j * 32u + i));
                };
            };

            // Layer 2 Backward: 32 -> 48
            Local<float> da1{48};
            $for(i, 48u) { da1.write(i, 0.0f); };
            $for(j, 32u) {
                Float z = z2.read(j);
                Float dy_dz = def(0.0f);
                $if (z > -1.5f & z < 1.5f) {
                    dy_dz = (2.0f / 3.0f) * z + 0.5f;
                } $elif (z >= 1.5f & z <= 3.0f) {
                    dy_dz = 1.0f;
                };
                Float delta = da2.read(j) * dy_dz;
                bias_grads.atomic(112u + j).fetch_add(delta);
                $for(i, 48u) {
                    weight_grads.atomic(6144u + j * 48u + i).fetch_add(delta * a1.read(i));
                    da1.write(i, da1.read(i) + delta * weights.read(6144u + j * 48u + i));
                };
            };

            // Layer 1 Backward: 48 -> 64
            Local<float> da0{64};
            $for(i, 64u) { da0.write(i, 0.0f); };
            $for(j, 48u) {
                Float z = z1.read(j);
                Float dy_dz = def(0.0f);
                $if (z > -1.5f & z < 1.5f) {
                    dy_dz = (2.0f / 3.0f) * z + 0.5f;
                } $elif (z >= 1.5f & z <= 3.0f) {
                    dy_dz = 1.0f;
                };
                Float delta = da1.read(j) * dy_dz;
                bias_grads.atomic(64u + j).fetch_add(delta);
                $for(i, 64u) {
                    weight_grads.atomic(3072u + j * 64u + i).fetch_add(delta * a0.read(i));
                    da0.write(i, da0.read(i) + delta * weights.read(3072u + j * 64u + i));
                };
            };

            // Layer 0 Backward: 64 -> 48
            Local<float> d_net_in{48};
            $for(i, 48u) { d_net_in.write(i, 0.0f); };
            $for(j, 64u) {
                Float z = z0.read(j);
                Float dy_dz = def(0.0f);
                $if (z > -1.5f & z < 1.5f) {
                    dy_dz = (2.0f / 3.0f) * z + 0.5f;
                } $elif (z >= 1.5f & z <= 3.0f) {
                    dy_dz = 1.0f;
                };
                Float delta = da0.read(j) * dy_dz;
                bias_grads.atomic(j).fetch_add(delta);
                $for(i, 48u) {
                    weight_grads.atomic(j * 48u + i).fetch_add(delta * net_in.read(i));
                    d_net_in.write(i, d_net_in.read(i) + delta * weights.read(j * 48u + i));
                };
            };

            // Latents Backward
            for (uint l = 0; l < 4; ++l) {
                UInt l_pix0 = gw0 * gh0;
                UInt l_pix1 = gw1 * gh1;
                for (uint k = 0; k < 4; ++k) {
                    $if (l * 4u + k < num_features) {
                        Float gf0 = d_net_in.read(l * 4u + k) * 2.0f;
                        latent_grads.atomic(base_off0 + (l * l_pix0 + p00_0) * 4u + k).fetch_add(gf0 * w00_0);
                        latent_grads.atomic(base_off0 + (l * l_pix0 + p10_0) * 4u + k).fetch_add(gf0 * w10_0);
                        latent_grads.atomic(base_off0 + (l * l_pix0 + p01_0) * 4u + k).fetch_add(gf0 * w01_0);
                        latent_grads.atomic(base_off0 + (l * l_pix0 + p11_0) * 4u + k).fetch_add(gf0 * w11_0);

                        Float gf1 = d_net_in.read(16u + l * 4u + k) * 2.0f;
                        latent_grads.atomic(base_off1 + (l * l_pix1 + p00_1) * 4u + k).fetch_add(gf1 * w00_1);
                        latent_grads.atomic(base_off1 + (l * l_pix1 + p10_1) * 4u + k).fetch_add(gf1 * w10_1);
                        latent_grads.atomic(base_off1 + (l * l_pix1 + p01_1) * 4u + k).fetch_add(gf1 * w01_1);
                        latent_grads.atomic(base_off1 + (l * l_pix1 + p11_1) * 4u + k).fetch_add(gf1 * w11_1);
                    };
                }
            }
        };
    };

    _networkAdamKernel = device.compile(net_adam);
    _latentAdamKernel = device.compile(lat_adam);
    _trainStepKernel = device.compile(train_step);
    stream << synchronize();
    _compiled = true;
}

void Trainer::_init_parameters() {
    std::mt19937 rng(_settings.randomSeed);

    std::vector<float> hostWeights(NTC_TOTAL_WEIGHTS);
    std::vector<float> hostBiases(NTC_TOTAL_BIASES, 0.0f);

    auto init_layer = [&](size_t offset, size_t in_c, size_t out_c) {
        float bound = std::sqrt(6.0f / static_cast<float>(in_c));
        std::uniform_real_distribution<float> dist(-bound, bound);
        for (size_t i = 0; i < in_c * out_c; ++i) {
            hostWeights[offset + i] = dist(rng);
        }
    };

    init_layer(0, 48, 64);
    init_layer(3072, 64, 48);
    init_layer(6144, 48, 32);
    init_layer(7680, 32, 16);

    _networkWeights = device.create_buffer<float>(NTC_TOTAL_WEIGHTS);
    _networkBiases = device.create_buffer<float>(NTC_TOTAL_BIASES);
    _networkWeightGrads = device.create_buffer<float>(NTC_TOTAL_WEIGHTS);
    _networkBiasGrads = device.create_buffer<float>(NTC_TOTAL_BIASES);
    _networkWeightM1 = device.create_buffer<float>(NTC_TOTAL_WEIGHTS);
    _networkWeightM2 = device.create_buffer<float>(NTC_TOTAL_WEIGHTS);
    _networkBiasM1 = device.create_buffer<float>(NTC_TOTAL_BIASES);
    _networkBiasM2 = device.create_buffer<float>(NTC_TOTAL_BIASES);

    std::vector<float> zeros_w(NTC_TOTAL_WEIGHTS, 0.0f);
    std::vector<float> zeros_b(NTC_TOTAL_BIASES, 0.0f);

    stream << copy_to_device(_networkWeights, hostWeights.data())
           << copy_to_device(_networkBiases, hostBiases.data())
           << copy_to_device(_networkWeightGrads, zeros_w.data())
           << copy_to_device(_networkBiasGrads, zeros_b.data())
           << copy_to_device(_networkWeightM1, zeros_w.data())
           << copy_to_device(_networkWeightM2, zeros_w.data())
           << copy_to_device(_networkBiasM1, zeros_b.data())
           << copy_to_device(_networkBiasM2, zeros_b.data());
}

bool Trainer::train(const MaterialManifest &manifest,
                    NtexContainer &out_container,
                    std::function<void(const TrainingProgress &)> progress_callback) {
    _init_parameters();

    int num_neural_mips = FeatureGridMath::calculate_num_neural_mips(manifest.width, manifest.height, _settings.gridScale);
    int num_layers = FeatureGridMath::get_num_layers(_settings.numFeatures);

    // Compute total latent floats across all mips and layers
    size_t total_latents = 0;
    std::vector<size_t> mip_latent_offsets(num_neural_mips);
    std::vector<uint> host_latent_info(num_neural_mips * 8, 0u);

    for (int mip = 0; mip < num_neural_mips; ++mip) {
        mip_latent_offsets[mip] = total_latents;
        int mw = FeatureGridMath::get_grid_dimension(manifest.width, mip, _settings.gridScale);
        int mh = FeatureGridMath::get_grid_dimension(manifest.height, mip, _settings.gridScale);

        host_latent_info[mip * 8 + 0] = static_cast<uint>(total_latents);
        host_latent_info[mip * 8 + 4] = static_cast<uint>(mw);
        host_latent_info[mip * 8 + 5] = static_cast<uint>(mh);

        total_latents += static_cast<size_t>(mw) * mh * num_layers * 4;
    }

    _latentMipInfoBuffer = device.create_buffer<uint4>(num_neural_mips * 2);
    stream << copy_to_device(_latentMipInfoBuffer, host_latent_info.data());

    std::mt19937 rng(_settings.randomSeed + 42);
    std::uniform_real_distribution<float> lat_dist(0.48f, 0.52f);
    std::vector<float> host_latents(total_latents);
    for (size_t i = 0; i < total_latents; ++i) {
        host_latents[i] = lat_dist(rng);
    }

    _latentFloats = device.create_buffer<float>(total_latents);
    _latentGrads = device.create_buffer<float>(total_latents);
    _latentM1 = device.create_buffer<float>(total_latents);
    _latentM2 = device.create_buffer<float>(total_latents);

    std::vector<float> zeros_lat(total_latents, 0.0f);
    stream << copy_to_device(_latentFloats, host_latents.data())
           << copy_to_device(_latentGrads, zeros_lat.data())
           << copy_to_device(_latentM1, zeros_lat.data())
           << copy_to_device(_latentM2, zeros_lat.data());

    // Flatten and upload reference mips
    size_t total_ref_floats = 0;
    for (const auto &mip_vec : manifest.mipLevelData) {
        total_ref_floats += mip_vec.size();
    }
    std::vector<float> host_ref(total_ref_floats);
    std::vector<float4> host_mip_info(manifest.numMips * 2);

    size_t curr_ref_offset = 0;
    for (uint32_t mip = 0; mip < manifest.numMips; ++mip) {
        std::memcpy(host_ref.data() + curr_ref_offset,
                    manifest.mipLevelData[mip].data(),
                    manifest.mipLevelData[mip].size() * sizeof(float));

        uint32_t mw = std::max(1u, manifest.width >> mip);
        uint32_t mh = std::max(1u, manifest.height >> mip);
        uint32_t nl = FeatureGridMath::lod_to_neural_lod(mip, _settings.gridScale, num_neural_mips);

        uint32_t gw = FeatureGridMath::get_grid_dimension(manifest.width, nl, _settings.gridScale);
        uint32_t gh = FeatureGridMath::get_grid_dimension(manifest.height, nl, _settings.gridScale);
        float ws = static_cast<float>(gw) / static_cast<float>(mw);
        float hs = static_cast<float>(gh) / static_cast<float>(mh);
        float pscale = 0.5f * std::max(ws, hs);
        float plod = std::clamp(0.25f * std::log2(pscale), -1.0f, 1.0f);

        host_mip_info[mip * 2 + 0] = make_float4(static_cast<float>(curr_ref_offset),
                                                 static_cast<float>(mw),
                                                 static_cast<float>(mh),
                                                 static_cast<float>(nl));
        host_mip_info[mip * 2 + 1] = make_float4(pscale, plod, 0.0f, 0.0f);

        curr_ref_offset += manifest.mipLevelData[mip].size();
    }

    _referenceMips = device.create_buffer<float>(std::max<size_t>(total_ref_floats, 1));
    _mipInfoBuffer = device.create_buffer<float4>(host_mip_info.size());
    _lossBuffer = device.create_buffer<float>(1);

    stream << copy_to_device(_referenceMips, host_ref.data())
           << copy_to_device(_mipInfoBuffer, host_mip_info.data())
           << synchronize();

    LUISA_INFO("Starting Real GPU NTC Training: {} steps, {} channels, resolution {}x{}",
               _settings.trainingSteps, manifest.totalChannels, manifest.width, manifest.height);

    auto t_start = std::chrono::high_resolution_clock::now();
    uint32_t batch_size = _settings.kPixelsPerBatch * 1024u;
    if (batch_size < 1024u) batch_size = 1024u;
    float zero_loss = 0.0f;

    for (uint32_t step = 1; step <= _settings.trainingSteps; ++step) {
        float progress = static_cast<float>(step) / static_cast<float>(_settings.trainingSteps);
        float lr_mult = 0.5f * (1.0f + std::cos(progress * 3.14159265f));
        float net_lr = _settings.networkLearningRate * (0.01f + 0.99f * lr_mult);
        float grid_lr = _settings.gridLearningRate * (0.01f + 0.99f * lr_mult);

        bool report = (step % _settings.stepsPerIteration == 0 || step == _settings.trainingSteps);
        if (report) {
            stream << copy_to_device(_lossBuffer, &zero_loss);
        }

        uint32_t step_seed = _settings.randomSeed + step * 71u;

        // 1. Forward-Backward Pass with real analytical gradients
        stream << _trainStepKernel(_networkWeights, _networkBiases,
                                  _networkWeightGrads, _networkBiasGrads,
                                  _latentFloats, _latentGrads,
                                  _referenceMips, _mipInfoBuffer,
                                  _latentMipInfoBuffer, _lossBuffer,
                                  step_seed, batch_size, manifest.numMips,
                                  manifest.totalChannels, _settings.numFeatures).dispatch(batch_size);

        // 2. Fused Adam Optimizer Updates
        stream << _networkAdamKernel(_networkWeights, _networkWeightGrads,
                                    _networkWeightM1, _networkWeightM2,
                                    net_lr, 0.9f, 0.999f, 1e-8f, NTC_TOTAL_WEIGHTS).dispatch(NTC_TOTAL_WEIGHTS)
               << _networkAdamKernel(_networkBiases, _networkBiasGrads,
                                    _networkBiasM1, _networkBiasM2,
                                    net_lr, 0.9f, 0.999f, 1e-8f, NTC_TOTAL_BIASES).dispatch(NTC_TOTAL_BIASES)
               << _latentAdamKernel(_latentFloats, _latentGrads,
                                   _latentM1, _latentM2,
                                   grid_lr, 0.9f, 0.999f, 1e-8f, static_cast<uint32_t>(total_latents)).dispatch(total_latents);

        if (report) {
            float host_loss_sum = 0.0f;
            stream << copy_from_device(_lossBuffer, &host_loss_sum) << synchronize();

            float mse = host_loss_sum / static_cast<float>(batch_size * std::max(1u, manifest.totalChannels));
            float psnr = -10.0f * std::log10(std::max(mse, 1e-7f));

            auto t_now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(t_now - t_start).count();

            TrainingProgress prog{};
            prog.currentStep = step;
            prog.totalSteps = _settings.trainingSteps;
            prog.currentLoss = mse;
            prog.currentPsnr = psnr;
            prog.elapsedTimeSec = elapsed;

            LUISA_INFO("Step {}/{}: Real MSE = {:.6f}, PSNR = {:.2f} dB, Elapsed = {:.2f}s",
                       step, _settings.trainingSteps, mse, psnr, elapsed);

            if (progress_callback) {
                progress_callback(prog);
            }
        }
    }

    // Export quantized container
    out_container = export_container(manifest);
    return true;
}

NtexContainer Trainer::export_container(const MaterialManifest &manifest) {
    NtexContainer container;
    container.schemaVersion = NTC_SCHEMA_VERSION;
    container.width = manifest.width;
    container.height = manifest.height;
    container.numChannels = (manifest.totalChannels + 1) & ~1; // NTC requires even number of channels
    container.numColorMips = manifest.numMips;

    container.latentShape.gridSizeScale = _settings.gridScale;
    container.latentShape.numFeatures = _settings.numFeatures;

    // Read back trained FP32 parameters from GPU
    std::vector<float> trainedWeights(NTC_TOTAL_WEIGHTS);
    std::vector<float> trainedBiases(NTC_TOTAL_BIASES);
    std::vector<float> trainedLatents(_latentFloats.size());

    stream << copy_from_device(_networkWeights, trainedWeights.data())
           << copy_from_device(_networkBiases, trainedBiases.data())
           << copy_from_device(_latentFloats, trainedLatents.data())
           << synchronize();

    // 1. Quantize INT8 MLP
    MLP mlpInt8;
    mlpInt8.activation = ActivationType::HGELUClamp;
    mlpInt8.weightLayout = MatrixLayout::RowMajor;

    // 2. Quantize FP8 MLP (hybrid FP8 E4M3 for layers 0-2 and INT8 for layer 3)
    MLP mlpFP8;
    mlpFP8.activation = ActivationType::HGELUClamp;
    mlpFP8.weightLayout = MatrixLayout::RowMajor;

    size_t w_off = 0;
    size_t b_off = 0;

    std::array<uint32_t, 4> in_dims = {48, 64, 48, 32};
    std::array<uint32_t, 4> out_dims = {64, 48, 32, 16};

    for (size_t layer = 0; layer < NTC_MLP_LAYERS; ++layer) {
        uint32_t in_c = in_dims[layer];
        uint32_t out_c = out_dims[layer];
        size_t layer_weights_count = in_c * out_c;

        std::vector<int8_t> int8Weights(layer_weights_count);
        std::vector<float> floatScales(out_c);
        std::vector<int32_t> int32Biases(out_c);

        for (uint32_t j = 0; j < out_c; ++j) {
            float max_abs = 0.0001f;
            for (uint32_t i = 0; i < in_c; ++i) {
                max_abs = std::max(max_abs, std::abs(trainedWeights[w_off + j * in_c + i]));
            }
            float limit = max_abs;
            float scale = limit / 127.0f;
            float inv_scale = 127.0f / limit;

            int integerWeightSum = 0;
            for (uint32_t i = 0; i < in_c; ++i) {
                float q = std::round(trainedWeights[w_off + j * in_c + i] * inv_scale);
                int8_t qi = static_cast<int8_t>(std::clamp(q, -127.0f, 127.0f));
                int8Weights[j * in_c + i] = qi;
                integerWeightSum += qi;
            }

            float layerScale = scale;
            float layerBias = trainedBiases[b_off + j];

            if (layer == 0) {
                layerScale /= NTC_INPUT_SCALE;
            } else {
                layerScale *= (1.0f / 3.0f);
                layerBias -= static_cast<float>(integerWeightSum * (-113)) * layerScale;
            }

            int32_t integerLayerBias = (layerScale > 1e-12f)
                ? static_cast<int32_t>(std::round(layerBias / layerScale))
                : 0;

            floatScales[j] = layerScale;
            int32Biases[j] = integerLayerBias;
        }

        MLPLayer l_int8;
        l_int8.inputChannels = in_c;
        l_int8.outputChannels = out_c;
        l_int8.weightType = MlpDataType::Int8;
        l_int8.scaleType = MlpDataType::Float32;
        l_int8.biasType = MlpDataType::Int32;
        l_int8.weightView = container.allocate_view_and_append_data(int8Weights.data(), int8Weights.size());
        l_int8.scaleView = container.allocate_view_and_append_data(floatScales.data(), floatScales.size() * sizeof(float));
        l_int8.biasView = container.allocate_view_and_append_data(int32Biases.data(), int32Biases.size() * sizeof(int32_t));
        mlpInt8.layers.push_back(l_int8);

        // FP8 layer
        MLPLayer l_fp8;
        l_fp8.inputChannels = in_c;
        l_fp8.outputChannels = out_c;

        if (layer == NTC_MLP_LAYERS - 1) {
            // Layer 3 uses INT8 for high precision
            l_fp8.weightType = MlpDataType::Int8;
            l_fp8.scaleType = MlpDataType::Float32;
            l_fp8.biasType = MlpDataType::Int32;
            l_fp8.weightView = container.allocate_view_and_append_data(int8Weights.data(), int8Weights.size());
            l_fp8.scaleView = container.allocate_view_and_append_data(floatScales.data(), floatScales.size() * sizeof(float));
            l_fp8.biasView = container.allocate_view_and_append_data(int32Biases.data(), int32Biases.size() * sizeof(int32_t));
        } else {
            // Layers 0-2 use FP8 E4M3
            std::vector<uint8_t> fp8Weights(layer_weights_count);
            for (size_t i = 0; i < layer_weights_count; ++i) {
                fp8Weights[i] = float_to_fp8_e4m3(trainedWeights[w_off + i]);
            }
            std::vector<uint16_t> fp16Biases(out_c);
            for (size_t c = 0; c < out_c; ++c) {
                fp16Biases[c] = float_to_fp16(trainedBiases[b_off + c]);
            }
            l_fp8.weightType = MlpDataType::FloatE4M3;
            l_fp8.biasType = MlpDataType::Float16;
            l_fp8.weightView = container.allocate_view_and_append_data(fp8Weights.data(), fp8Weights.size());
            l_fp8.biasView = container.allocate_view_and_append_data(fp16Biases.data(), fp16Biases.size() * sizeof(uint16_t));
        }
        mlpFP8.layers.push_back(l_fp8);

        w_off += layer_weights_count;
        b_off += out_c;
    }

    container.mlpVersions.push_back(mlpInt8);
    container.mlpVersions.push_back(mlpFP8);

    // 3. Populate Textures & Channels
    for (const auto &mtex : manifest.textures) {
        TextureDesc td;
        td.name = mtex.name;
        td.firstChannel = mtex.firstChannel;
        td.numChannels = mtex.channels;
        td.channelFormat = ChannelFormat::UNORM8;
        td.rgbColorSpace = mtex.isSRGB ? ColorSpace::sRGB : ColorSpace::Linear;
        td.bcFormat = mtex.bcFormat;
        container.textures.push_back(td);
    }

    for (size_t ch = 0; ch < container.numChannels; ++ch) {
        ChannelDesc cd;
        cd.colorSpace = ColorSpace::Linear;
        if (ch < manifest.totalChannels) {
            for (const auto &tex : manifest.textures) {
                if (ch >= tex.firstChannel && ch < tex.firstChannel + tex.channels) {
                    if (tex.isSRGB && (ch - tex.firstChannel) < 3) {
                        cd.colorSpace = ColorSpace::sRGB;
                    }
                }
            }
        }
        container.channels.push_back(cd);
    }

    // 4. Quantize and pack Latents (BGRA4)
    int num_neural_mips = FeatureGridMath::calculate_num_neural_mips(manifest.width, manifest.height, _settings.gridScale);
    int num_layers = FeatureGridMath::get_num_layers(_settings.numFeatures);

    size_t lat_offset = 0;
    for (int mip = 0; mip < num_neural_mips; ++mip) {
        uint32_t mw = FeatureGridMath::get_grid_dimension(manifest.width, mip, _settings.gridScale);
        uint32_t mh = FeatureGridMath::get_grid_dimension(manifest.height, mip, _settings.gridScale);

        LatentImageDesc lid;
        lid.width = mw;
        lid.height = mh;

        for (int l = 0; l < num_layers; ++l) {
            size_t num_pixels = static_cast<size_t>(mw) * mh;
            std::vector<uint16_t> packedLayer(num_pixels);

            for (size_t p = 0; p < num_pixels; ++p) {
                size_t p_off = lat_offset + (l * num_pixels + p) * 4;
                float f0 = trainedLatents[p_off + 0];
                float f1 = trainedLatents[p_off + 1];
                float f2 = trainedLatents[p_off + 2];
                float f3 = trainedLatents[p_off + 3];
                packedLayer[p] = FeatureGridStorage::pack_bgra4(f0, f1, f2, f3);
            }

            uint32_t view_idx = container.allocate_view_and_append_data(
                packedLayer.data(), packedLayer.size() * sizeof(uint16_t), CompressionType::None);
            lid.layerViews.push_back(view_idx);
        }

        container.latents.push_back(lid);
        lat_offset += static_cast<size_t>(mw) * mh * num_layers * 4;
    }

    // 5. Populate Color Mips with exact PositionScale and PositionLod
    for (uint32_t mipLevel = 0; mipLevel < manifest.numMips; ++mipLevel) {
        ColorMipDesc cmd;
        cmd.width = std::max(1u, manifest.width >> mipLevel);
        cmd.height = std::max(1u, manifest.height >> mipLevel);
        cmd.latentMip = FeatureGridMath::lod_to_neural_lod(mipLevel, _settings.gridScale, num_neural_mips);

        uint32_t gw = FeatureGridMath::get_grid_dimension(manifest.width, cmd.latentMip, _settings.gridScale);
        uint32_t gh = FeatureGridMath::get_grid_dimension(manifest.height, cmd.latentMip, _settings.gridScale);
        float widthScale = static_cast<float>(gw) / static_cast<float>(cmd.width);
        float heightScale = static_cast<float>(gh) / static_cast<float>(cmd.height);
        cmd.positionScale = 0.5f * std::max(widthScale, heightScale);
        cmd.positionLod = std::clamp(0.25f * std::log2(cmd.positionScale), -1.0f, 1.0f);
        container.colorMips.push_back(cmd);
    }

    return container;
}

} // namespace luisa::ntc
