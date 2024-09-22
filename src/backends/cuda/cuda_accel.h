#pragma once

#include <cuda.h>

#include <luisa/runtime/rtx/accel.h>
#include "cuda_primitive.h"
#include "optix_api.h"

namespace luisa::compute::cuda {

class CUDAMesh;
class CUDAMotionInstance;
class CUDAHeap;
class CUDADevice;
class CUDAStream;
class CUDACommandEncoder;

/**
 * @brief Acceleration structure of CUDA
 * 
 */
class CUDAAccel {

public:
    /**
     * @brief Binding struct of device
     * 
     */
    struct alignas(16) Binding {
        optix::TraversableHandle handle;
        CUdeviceptr instances;
    };

private:
    AccelOption _option;
    bool _requires_rebuild{true};
    mutable spin_mutex _mutex;
    optix::TraversableHandle _handle{};
    CUdeviceptr _instance_buffer{};
    size_t _instance_buffer_size{};
    CUdeviceptr _bvh_buffer{};
    size_t _bvh_buffer_size{};
    size_t _update_buffer_size{};
    luisa::vector<const CUDAPrimitiveBase *> _primitives;
    luisa::vector<uint64_t> _prim_handles;
    luisa::unordered_map<const CUDAMotionInstance *, const CUDAPrimitive *> _motion_instance_to_primitive;
    luisa::string _name;

private:
    void _build(CUDACommandEncoder &encoder) noexcept;
    void _update(CUDACommandEncoder &encoder) noexcept;

public:
    explicit CUDAAccel(const AccelOption &option) noexcept;
    ~CUDAAccel() noexcept;
    void build(CUDACommandEncoder &encoder, AccelBuildCommand *command) noexcept;
    [[nodiscard]] optix::TraversableHandle handle() const noexcept;
    [[nodiscard]] CUdeviceptr instance_buffer() const noexcept;
    [[nodiscard]] Binding binding() const noexcept;
    [[nodiscard]] auto pointer_to_handle() const noexcept { return &_handle; }
    void set_name(luisa::string &&name) noexcept;
};

}// namespace luisa::compute::cuda

