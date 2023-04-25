//
// Created by Mike Smith on 2021/7/22.
//

#pragma once

#import <vector>
#import <Metal/Metal.h>

#import <core/stl.h>
#import <rtx/accel.h>
#import <backends/metal/metal_mesh.h>

namespace luisa::compute::metal {

class MetalDevice;
class MetalStream;

class MetalAccel {

public:
    struct Resource {
        id<MTLResource> handle;
        [[nodiscard]] bool operator==(const Resource &rhs) const noexcept { return handle == rhs.handle; }
    };
    struct ResourceHash {
        [[nodiscard]] uint64_t operator()(const Resource &r) const noexcept { return [r.handle hash]; }
    };

private:
    id<MTLComputePipelineState> _update_shader;
    id<MTLAccelerationStructure> _handle{nullptr};
    id<MTLBuffer> _instance_buffer{nullptr};
    id<MTLBuffer> _update_buffer{nullptr};
    MTLInstanceAccelerationStructureDescriptor *_descriptor{nullptr};
    size_t _update_scratch_size{};
    luisa::vector<const MetalMesh *> _meshes;
    NSMutableArray<id<MTLAccelerationStructure>> *_mesh_handles;
    luisa::unordered_set<Resource, ResourceHash, std::equal_to<>> _resources;

public:
    MetalAccel(id<MTLComputePipelineState> update_shader, AccelUsageHint hint) noexcept;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] id<MTLCommandBuffer> build(
        MetalStream *stream, id<MTLCommandBuffer> command_buffer, uint instance_count,
        AccelBuildRequest request, luisa::span<const AccelBuildCommand::Modification> mods) noexcept;
    [[nodiscard]] auto instance_buffer() const noexcept { return _instance_buffer; }
    [[nodiscard]] auto descriptor() const noexcept { return _descriptor; }
    [[nodiscard]] auto &resources() const noexcept { return _resources; }
};

}// namespace luisa::compute::metal
