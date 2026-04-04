#include <Resource/FeatureCheck.h>
#include <DXRuntime/Device.h>
#include <luisa/core/logging.h>

namespace lc::dx {

void FeatureCheck::check(Device const *device) {
    if (!device || !device->device) {
        return;
    }

    auto d3d_device = device->device.Get();

    // Check Resource Binding Tier for bindless support (Tier 3 required for full bindless)
    {
        D3D12_FEATURE_DATA_D3D12_OPTIONS options = {};
        HRESULT hr = d3d_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options));
        if (SUCCEEDED(hr)) {
            _flags.bindless_binding_supported = (options.ResourceBindingTier >= D3D12_RESOURCE_BINDING_TIER_3);
        } else {
            _flags.bindless_binding_supported = false;
        }
    }

    // Check Ray-Tracing support (Tier 1.0 for accel structure, Tier 1.1 for ray query)
    {
        D3D12_FEATURE_DATA_D3D12_OPTIONS5 options5 = {};
        HRESULT hr = d3d_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &options5, sizeof(options5));
        if (SUCCEEDED(hr)) {
            _raytracing_tier = options5.RaytracingTier;
            _flags.raytracing_supported = (options5.RaytracingTier >= D3D12_RAYTRACING_TIER_1_0) && (options5.RaytracingTier >= D3D12_RAYTRACING_TIER_1_1);
        } else {
            _raytracing_tier = D3D12_RAYTRACING_TIER_NOT_SUPPORTED;
            _flags.raytracing_supported = false;
        }
    }

    // Check Wave/Warp operations support
    {
        D3D12_FEATURE_DATA_D3D12_OPTIONS1 options1 = {};
        HRESULT hr = d3d_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &options1, sizeof(options1));
        if (SUCCEEDED(hr)) {
            _wave_lane_count_min = options1.WaveLaneCountMin;
            _wave_lane_count_max = options1.WaveLaneCountMax;
            _flags.wave_operation_supported = (options1.WaveLaneCountMax > 0);
        } else {
            _wave_lane_count_min = 0;
            _wave_lane_count_max = 0;
            _flags.wave_operation_supported = false;
        }
    }

    // Check Enhanced Barriers support
    {
        D3D12_FEATURE_DATA_D3D12_OPTIONS12 options12 = {};
        HRESULT hr = d3d_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS12, &options12, sizeof(options12));
        if (SUCCEEDED(hr)) {
            _flags.enhanced_barriers_supported = options12.EnhancedBarriersSupported;
        } else {
            _flags.enhanced_barriers_supported = false;
        }
    }

    // Check Mesh Shader support
    {
        D3D12_FEATURE_DATA_D3D12_OPTIONS7 options7 = {};
        HRESULT hr = d3d_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &options7, sizeof(options7));
        if (SUCCEEDED(hr)) {
            _flags.mesh_shader_supported = (options7.MeshShaderTier >= D3D12_MESH_SHADER_TIER_1);
        } else {
            _flags.mesh_shader_supported = false;
        }
    }

    // Check Cooperative Vector support (requires experimental features enabled)
    /* {
        D3D12_FEATURE_DATA_D3D12_OPTIONS_EXPERIMENTAL options_exp = {};
        HRESULT hr = d3d_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS_EXPERIMENTAL,
                                                     &options_exp,
                                                     sizeof(options_exp));
        if (SUCCEEDED(hr)) {
            _cooperative_vector_tier = options_exp.CooperativeVectorTier;
            _flags.cooperative_vector_supported = (options_exp.CooperativeVectorTier >= D3D12_COOPERATIVE_VECTOR_TIER_1_0);
        } else {
            // This is expected if experimental features are not enabled
            _cooperative_vector_tier = D3D12_COOPERATIVE_VECTOR_TIER_NOT_SUPPORTED;
            _flags.cooperative_vector_supported = false;
        }
    } */

    // Check Work Graph support (D3D12_OPTIONS21)
    {
        D3D12_FEATURE_DATA_D3D12_OPTIONS21 options21 = {};
        HRESULT hr = d3d_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS21, &options21, sizeof(options21));
        if (SUCCEEDED(hr)) {
            _work_graphs_tier = options21.WorkGraphsTier;
            _flags.work_graph_supported = (options21.WorkGraphsTier >= D3D12_WORK_GRAPHS_TIER_1_0);
        } else {
            _work_graphs_tier = D3D12_WORK_GRAPHS_TIER_NOT_SUPPORTED;
            _flags.work_graph_supported = false;
        }
    }
}

}// namespace lc::dx
