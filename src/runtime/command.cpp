//
// Created by Mike Smith on 2021/3/3.
//

#include <core/logging.h>
#include <runtime/command.h>

namespace luisa::compute {

void KernelArgumentEncoder::Deleter::operator()(KernelArgumentEncoder::Data *ptr) const noexcept {
    _recycle(ptr);
}

std::vector<std::unique_ptr<KernelArgumentEncoder::Data>> &KernelArgumentEncoder::_available_blocks() noexcept {
    static thread_local std::vector<std::unique_ptr<Data>> blocks;
    return blocks;
}

KernelArgumentEncoder::Storage KernelArgumentEncoder::_allocate() noexcept {
    auto &&blocks = _available_blocks();
    if (blocks.empty()) { return Storage{new Data}; }
    auto data = blocks.back().release();
    blocks.pop_back();
    return Storage{data};
}

void KernelArgumentEncoder::_recycle(Data *storage) noexcept {
    _available_blocks().emplace_back(storage);
}

std::span<const KernelArgumentEncoder::Argument> KernelArgumentEncoder::arguments() const noexcept {
    return _arguments;
}

void KernelArgumentEncoder::encode_buffer(uint64_t handle, size_t offset_bytes) noexcept {
    _arguments.emplace_back(BufferArgument{handle, offset_bytes});
}

void KernelArgumentEncoder::encode_uniform(const void *data, size_t size, size_t alignment) noexcept {
    auto aligned_ptr = reinterpret_cast<std::byte *>((reinterpret_cast<uint64_t>(_ptr) + alignment - 1u) / alignment * alignment);
    if (aligned_ptr + size > _storage->data() + _storage->size()) {
        LUISA_ERROR_WITH_LOCATION("Size limit of uniform data exceeded.");
    }
    std::memmove(aligned_ptr, data, size);
    _arguments.emplace_back(UniformArgument{std::span{aligned_ptr, size}});
    _ptr = aligned_ptr + size;
}

std::span<const std::byte> KernelArgumentEncoder::uniform_data() const noexcept {
    return {_storage->data(), _ptr};
}

}// namespace luisa::compute
