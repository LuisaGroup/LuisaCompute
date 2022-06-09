//
// Created by Mike Smith on 2022/2/11.
//

#include <backends/llvm/llvm_bindless_array.h>

namespace luisa::compute::llvm {

LLVMBindlessArray::LLVMBindlessArray(size_t capacity) noexcept
    : _slots(capacity), _items(capacity) {}

void LLVMBindlessArray::emplace_buffer(size_t index, const void *buffer, size_t offset) noexcept {
    remove_buffer(index);
    _slots[index].buffer = buffer;
    _slots[index].buffer_offset = offset;
    _tracker.retain(reinterpret_cast<uint64_t>(buffer));
    _dirty.mark(index);
}

void LLVMBindlessArray::emplace_tex2d(size_t index, const LLVMTexture *tex, Sampler sampler) noexcept {
    remove_tex2d(index);
    _slots[index].tex2d = tex;
    _slots[index].sampler2d = sampler;
    _tracker.retain(reinterpret_cast<uint64_t>(tex));
    _dirty.mark(index);
}

void LLVMBindlessArray::emplace_tex3d(size_t index, const LLVMTexture *tex, Sampler sampler) noexcept {
    remove_tex3d(index);
    _slots[index].tex3d = tex;
    _slots[index].sampler3d = sampler;
    _tracker.retain(reinterpret_cast<uint64_t>(tex));
    _dirty.mark(index);
}

void LLVMBindlessArray::remove_buffer(size_t index) noexcept {
    if (auto &buffer = _slots[index].buffer) {
        _tracker.release(reinterpret_cast<uint64_t>(buffer));
        buffer = nullptr;
    }
}

void LLVMBindlessArray::remove_tex2d(size_t index) noexcept {
    if (auto &texture = _slots[index].tex2d) {
        _tracker.release(reinterpret_cast<uint64_t>(texture));
        texture = nullptr;
    }
}

void LLVMBindlessArray::remove_tex3d(size_t index) noexcept {
    if (auto &texture = _slots[index].tex3d) {
        _tracker.release(reinterpret_cast<uint64_t>(texture));
        texture = nullptr;
    }
}

void LLVMBindlessArray::update(ThreadPool &pool) noexcept {
    auto s = luisa::span{_slots};//.subspan(_dirty.offset(), _dirty.size());
    pool.async([this, offset = 0u,//offset = _dirty.offset(),
                slots = luisa::vector<Slot>{s.cbegin(), s.cend()}] {
        for (auto i = 0u; i < slots.size(); i++) {
            auto slot = slots[i];
            if (auto b = static_cast<const std::byte *>(slot.buffer)) {
                _items[i + offset].buffer = b + slot.buffer_offset;
            }
            if (auto t = slot.tex2d) {
                _items[i + offset].tex2d = t;
                _items[i + offset].sampler2d = slot.sampler2d.code();
            }
            if (auto t = slot.tex3d) {
                _items[i + offset].tex3d = t;
                _items[i + offset].sampler3d = slot.sampler3d.code();
            }
        }
    });
    _dirty.clear();
    _tracker.commit();
}

bool LLVMBindlessArray::uses_resource(uint64_t handle) const noexcept {
    return _tracker.uses(handle);
}

}// namespace luisa::compute::llvm
