#pragma once

#include <rtx/accel.h>
#include <rtx/mesh.h>
#include <py/managed_collector.h>
#include <py/py_stream.h>
#include <core/stl/unordered_map.h>

namespace luisa::compute {

struct MeshUpdateCmd {
    AccelOption option;
    uint64_t vertex_buffer;
    size_t vertex_buffer_offset;
    size_t vertex_buffer_size;
    size_t vertex_stride;
    uint64_t triangle_buffer;
    size_t triangle_buffer_offset;
    size_t triangle_buffer_size;
};

class ManagedAccel final {
    struct Data : public vstd::IOperatorNewBase {
        ManagedCollector collector;
        Accel accel;
        vstd::vector<std::pair<uint64, MeshUpdateCmd>> meshes;
        luisa::unordered_map<uint64, MeshUpdateCmd> requireUpdateMesh;
        vstd::vector<uint64> meshDisposeList;
        Data(Accel &&accel);
    };
    vstd::unique_ptr<Data> data;

public:
    Accel &GetAccel() noexcept { return data->accel; }
    ManagedAccel(Accel &&accel) noexcept;
    ManagedAccel(ManagedAccel &&) noexcept = default;
    ManagedAccel(ManagedAccel const &) = delete;
    ~ManagedAccel() noexcept;
    ManagedAccel &operator=(ManagedAccel &&) = default;
    ManagedAccel &operator=(ManagedAccel const&) = delete;
    ManagedAccel() = delete;
    void emplace(MeshUpdateCmd const &mesh, float4x4 const &transform, bool visible, bool opaque) noexcept;
    void pop_back() noexcept;
    void set(size_t idx, MeshUpdateCmd const &mesh, float4x4 const &transform, bool visible, bool opaque) noexcept;
    void update(PyStream &stream) noexcept;
};

}// namespace luisa::compute
