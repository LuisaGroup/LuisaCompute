#pragma once

#include <core/stl.h>
#include <runtime/sampler.h>
#include <vector>
#include <core/logging.h>

using namespace luisa::compute;

namespace lc::ispc {

class ISPCBindlessArray {
public:
    struct DeviceData {
        uint64_t* buffer;
        uint64_t* tex2d;
        uint64_t* tex3d;
        uint32_t* tex2dSize;
        uint32_t* tex3dSize;
    };
private:
    DeviceData data;
    size_t size;
    luisa::vector<uint64_t> bufferVector;
    luisa::vector<uint64_t> bufferAddressVector;
    luisa::vector<uint64_t> tex2dVector;
    luisa::vector<uint64_t> tex3dVector;
    luisa::vector<uint32_t> tex2dSizeVector;
    luisa::vector<uint32_t> tex3dSizeVector;
public:
    explicit ISPCBindlessArray(size_t size) noexcept;
    void emplace_buffer(size_t index, uint64_t buffer, size_t offset) noexcept;
    void emplace_tex2d(size_t index, uint64_t buffer, Sampler sampler) noexcept;
    void emplace_tex3d(size_t index, uint64_t buffer, Sampler sampler) noexcept;
    void remove_buffer(size_t index) noexcept;
    void remove_tex2d(size_t index) noexcept;
    void remove_tex3d(size_t index) noexcept;
    [[nodiscard]] bool uses_buffer(uint64_t handle) const noexcept;
    [[nodiscard]] bool uses_texture(uint64_t handle) const noexcept;
    [[nodiscard]] auto getDeviceData() const noexcept { return data; }
};

}