/**
 * LuisaCompute NTC (RTX Neural Texture Compression)
 * GPU Training Engine: Multi-Mip Sampling, Forward-Backward Regression, Fused Adam & QAT.
 */

#pragma once

#include <libntc/common.h>
#include "ntex_container.h"
#include "feature_grid.h"
#include "manifest.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <vector>
#include <memory>

namespace luisa::ntc {

struct TrainingSettings {
    uint32_t trainingSteps{10000};
    uint32_t stepsPerIteration{1000};
    float networkLearningRate{0.005f};
    float gridLearningRate{0.1f};
    uint32_t kPixelsPerBatch{64};
    uint32_t numFeatures{16};
    uint32_t gridScale{4};
    uint32_t randomSeed{1337};
    bool useFP8{true};
    bool generateMips{true};
};

struct TrainingProgress {
    uint32_t currentStep{0};
    uint32_t totalSteps{0};
    float currentLoss{0.0f};
    float currentPsnr{0.0f};
    double elapsedTimeSec{0.0};
};

class Trainer {
public:
    Device &device;
    Stream &stream;

private:
    TrainingSettings _settings;

    // Ground-truth textures on GPU
    Buffer<float> _referenceMips; // Flattened reference pixels for all mips
    std::vector<uint32_t> _mipOffsets;
    std::vector<uint32_t> _mipDims;

    // Network parameters (FP32 training state)
    Buffer<float> _networkWeights;      // 8192 floats
    Buffer<float> _networkBiases;       // 160 floats
    Buffer<float> _networkWeightGrads;  // 8192 floats
    Buffer<float> _networkBiasGrads;    // 160 floats
    Buffer<float> _networkWeightM1;     // 8192 floats
    Buffer<float> _networkWeightM2;     // 8192 floats
    Buffer<float> _networkBiasM1;       // 160 floats
    Buffer<float> _networkBiasM2;       // 160 floats

    // Latent grids (FP32 training state)
    Buffer<float> _latentFloats;        // All latents in [0, 1]
    Buffer<float> _latentGrads;         // Latent gradients (accumulated via atomics)
    Buffer<float> _latentM1;            // Latent Adam M1
    Buffer<float> _latentM2;            // Latent Adam M2

    // GPU Buffers for Mips and Loss
    Buffer<float4> _mipInfoBuffer;
    Buffer<uint4> _latentMipInfoBuffer;
    Buffer<float> _lossBuffer;

    // Shaders
    Shader1D<Buffer<float>, Buffer<float>, Buffer<float>, Buffer<float>,
             Buffer<float>, Buffer<float>, Buffer<float>, Buffer<float4>,
             Buffer<uint4>, Buffer<float>,
             uint, uint, uint, uint, uint> _trainStepKernel;
    Shader1D<Buffer<float>, Buffer<float>, Buffer<float>, Buffer<float>, float, float, float, float, uint> _networkAdamKernel;
    Shader1D<Buffer<float>, Buffer<float>, Buffer<float>, Buffer<float>, float, float, float, float, uint> _latentAdamKernel;

    bool _compiled{false};

    void _init_parameters();
    void _compile_shaders();

public:
    Trainer(Device &dev, Stream &st, const TrainingSettings &settings = {});

    // Train on a loaded material manifest
    bool train(const MaterialManifest &manifest,
               NtexContainer &out_container,
               std::function<void(const TrainingProgress &)> progress_callback = nullptr);

    // Build quantized NTEX container from trained state
    NtexContainer export_container(const MaterialManifest &manifest);
};

} // namespace luisa::ntc
