#include <backends/ispc/runtime/ispc_bindless_array.h>
#include <backends/ispc/ISPCTest/Types.h>

namespace lc::ispc {

ISPCBindlessArray::ISPCBindlessArray(size_t size) noexcept : 
    size(size), bufferVector(size, 0), bufferAddressVector(size, 0), tex2dVector(size, 0), tex3dVector(size, 0),
    tex2dSizeVector(size * 2, 0), tex3dSizeVector(size * 3, 0) {
    data.buffer = bufferAddressVector.data();
    data.tex2d = tex2dVector.data();
    data.tex3d = tex3dVector.data();
    data.tex2dSize = tex2dSizeVector.data();
    data.tex3dSize = tex3dSizeVector.data();
}

void ISPCBindlessArray::emplace_buffer(size_t index, uint64_t buffer, size_t offset) noexcept {
    bufferVector[index] = buffer;
    bufferAddressVector[index] = buffer + offset;
}

void ISPCBindlessArray::emplace_tex2d(size_t index, uint64_t tex, Sampler sampler) noexcept {
    tex2dVector[index] = tex;
    tex2dSizeVector[index * 2 + 0] = reinterpret_cast<Texture2D*>(tex)->width;
    tex2dSizeVector[index * 2 + 1] = reinterpret_cast<Texture2D*>(tex)->height;
}

void ISPCBindlessArray::emplace_tex3d(size_t index, uint64_t tex, Sampler sampler) noexcept {

}

void ISPCBindlessArray::remove_buffer(size_t index) noexcept {
    bufferVector[index] = 0;
}

void ISPCBindlessArray::remove_tex2d(size_t index) noexcept {
    tex2dVector[index] = 0;
}

void ISPCBindlessArray::remove_tex3d(size_t index) noexcept {
    tex3dVector[index] = 0;
}

[[nodiscard]] bool ISPCBindlessArray::uses_buffer(uint64_t handle) const noexcept {
    for(auto& item: bufferVector)
        if(item == handle) return true;
    return false;
}

[[nodiscard]] bool ISPCBindlessArray::uses_texture(uint64_t handle) const noexcept {
    for(auto& item: tex2dVector)
        if(item == handle) return true;
    for(auto& item: tex3dVector)
        if(item == handle) return true;
    return false;
}


}