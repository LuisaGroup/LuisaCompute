#pragma once

#include <hip/hip_runtime.h>

#include <luisa/core/spin_mutex.h>
#include <luisa/core/stl.h>
#include <luisa/runtime/rhi/command.h>
#include "../common/resource_tracker.h"

namespace luisa::compute::hip {

class HIPCommandEncoder;

class HIPBindlessArray {

public:
    // ABI contract with hip_codegen_llvm_impl_type.cpp:
    // { i64, i64, i64, i64, i64, i64, i64, i64, i64 } = 72 bytes per slot
    struct Slot {
        uint64_t buffer;
        size_t size;
        uint64_t tex2d;
        uint64_t tex2d_levels;
        uint64_t tex2d_size;
        uint64_t tex3d;
        uint64_t tex3d_levels;
        uint64_t tex3d_size_xy;
        uint64_t tex3d_size_z;
    };
    static_assert(sizeof(Slot) == 72u);

    struct Binding {
        hipDeviceptr_t slots;
        size_t capacity;
    };

private:
    struct TextureSlot {
        hipDeviceptr_t handle_table{};
        luisa::vector<hipTextureObject_t> objects;
    };

    hipDeviceptr_t _handle{};
    size_t _capacity{};
    luisa::vector<Slot> _host_slots;
    luisa::vector<TextureSlot> _tex2d_slots;
    luisa::vector<TextureSlot> _tex3d_slots;
    ResourceTracker _texture_tracker;
    ResourceTracker _texture_handle_table_tracker;
    luisa::string _name;
    spin_mutex _mutex;

public:
    explicit HIPBindlessArray(size_t capacity) noexcept;
    ~HIPBindlessArray() noexcept;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto binding() const noexcept { return Binding{_handle, _capacity}; }
    void update(HIPCommandEncoder &encoder, BindlessArrayUpdateCommand *cmd) noexcept;
    void set_name(luisa::string &&name) noexcept;
};

}// namespace luisa::compute::hip
