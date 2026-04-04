#pragma once
#include <d3dx12.h>

namespace lc::dx {

// Forward declaration
class Device;

/// FeatureCheck class for querying DirectX 12 GPU capabilities
/// This class provides a centralized way to check hardware feature support
/// including bindless binding, ray-tracing, wave operations, and cooperative vectors.
class FeatureCheck {
public:
    struct FeatureFlags {
        // Resource binding tier 3 is required for bindless (full tier 3 support)
        bool bindless_binding_supported : 1;
        // Ray-query support (Tier 1.1 or higher)
        bool raytracing_supported : 1;
        // Wave/Warp operations support (WaveLaneCountMax > 0)
        bool wave_operation_supported : 1;
        // Cooperative vector support (Tier 1.0 or higher, requires experimental features)
        bool cooperative_vector_supported : 1;
        // Enhanced barriers support (D3D12_OPTIONS12)
        bool enhanced_barriers_supported : 1;
        // Mesh shader support (D3D12_OPTIONS7)
        bool mesh_shader_supported : 1;
        // Work graph support (D3D12_OPTIONS21)
        bool work_graph_supported : 1;
        // Optional: additional flags for future expansion
    };

private:
    FeatureFlags _flags{};
    uint _wave_lane_count_min{0};
    uint _wave_lane_count_max{0};
    D3D12_RAYTRACING_TIER _raytracing_tier{D3D12_RAYTRACING_TIER_NOT_SUPPORTED};
    // D3D12_COOPERATIVE_VECTOR_TIER _cooperative_vector_tier{D3D12_COOPERATIVE_VECTOR_TIER_NOT_SUPPORTED};
    D3D12_WORK_GRAPHS_TIER _work_graphs_tier{D3D12_WORK_GRAPHS_TIER_NOT_SUPPORTED};

public:
    /// Constructor - initializes and queries all feature support from the device
    void check(Device const *device);

    /// Default constructor - all features marked as unsupported
    FeatureCheck() = default;

    /// Check if bindless binding is supported (Resource Binding Tier 3)
    [[nodiscard]] bool bindless_binding_supported() const noexcept {
        return _flags.bindless_binding_supported;
    }

    /// Check if ray-query is supported (requires Raytracing Tier 1.1)
    [[nodiscard]] bool raytracing_supported() const noexcept {
        return _flags.raytracing_supported;
    }

    /// Check if wave/warp operations are supported
    [[nodiscard]] bool wave_operation_supported() const noexcept {
        return _flags.wave_operation_supported;
    }

    /// Check if cooperative vector is supported
    [[nodiscard]] bool cooperative_vector_supported() const noexcept {
        return _flags.cooperative_vector_supported;
    }

    /// Check if enhanced barriers are supported
    [[nodiscard]] bool enhanced_barriers_supported() const noexcept {
        return _flags.enhanced_barriers_supported;
    }

    /// Check if mesh shader is supported
    [[nodiscard]] bool mesh_shader_supported() const noexcept {
        return _flags.mesh_shader_supported;
    }

    /// Check if work graph is supported
    [[nodiscard]] bool work_graph_supported() const noexcept {
        return _flags.work_graph_supported;
    }

    /// Get minimum wave lane count
    [[nodiscard]] uint wave_lane_count_min() const noexcept {
        return _wave_lane_count_min;
    }

    /// Get maximum wave lane count
    [[nodiscard]] uint wave_lane_count_max() const noexcept {
        return _wave_lane_count_max;
    }

    /// Get ray-tracing tier level
    [[nodiscard]] D3D12_RAYTRACING_TIER raytracing_tier() const noexcept {
        return _raytracing_tier;
    }

    /// Get cooperative vector tier level
    // [[nodiscard]] D3D12_COOPERATIVE_VECTOR_TIER cooperative_vector_tier() const noexcept {
    //     return _cooperative_vector_tier;
    // }

    /// Get work graphs tier level
    [[nodiscard]] D3D12_WORK_GRAPHS_TIER work_graphs_tier() const noexcept {
        return _work_graphs_tier;
    }

    /// Get all feature flags as a struct
    [[nodiscard]] FeatureFlags &flags() noexcept {
        return _flags;
    }
};

}// namespace lc::dx
