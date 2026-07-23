//
// Created by Mike Smith on 2022/2/11.
//

#include "fallback_bindless_array.h"
#include "fallback_buffer.h"
#include <luisa/core/logging.h>

namespace luisa::compute::fallback {

FallbackBindlessArray::FallbackBindlessArray(size_t capacity) noexcept
    : _slots(capacity) {}

void FallbackBindlessArray::update(luisa::unique_ptr<BindlessArrayUpdateCommand> cmd) noexcept {
    for (auto m : cmd->modifications()) {
        auto &slot = _slots[m.slot];
        switch (m.buffer.op) {
            case BindlessArrayUpdateCommand::Modification::Operation::EMPLACE: {
                auto view = reinterpret_cast<FallbackBuffer *>(m.buffer.handle)->view_with_offset(m.buffer.offset_bytes);
                auto size =
                    m.buffer.size_bytes ==
                            BindlessArrayUpdateCommand::ModifiedBuffer::whole_buffer_size ?
                        view.size :
                        m.buffer.size_bytes;
                LUISA_ASSERT(size > 0u && size <= view.size,
                             "Bindless buffer view [{}, {}) exceeds buffer size {}.",
                             m.buffer.offset_bytes,
                             m.buffer.offset_bytes + size,
                             m.buffer.offset_bytes + view.size);
                slot.buffer = view.ptr;
                slot.set_buffer_size(size);
                break;
            }
            case BindlessArrayUpdateCommand::Modification::Operation::REMOVE: {
                slot.buffer = nullptr;
                slot.set_buffer_size(0);
                break;
            }
            default: break;
        }
        switch (m.tex2d.op) {
            case BindlessArrayUpdateCommand::Modification::Operation::EMPLACE: {
                slot.tex2d = reinterpret_cast<FallbackTexture *>(m.tex2d.handle);
                slot.set_sampler_2d(m.tex2d.sampler);
                break;
            }
            case BindlessArrayUpdateCommand::Modification::Operation::REMOVE: {
                slot.tex2d = nullptr;
                slot.set_sampler_2d(0);
                break;
            }
            default: break;
        }
        switch (m.tex3d.op) {
            case BindlessArrayUpdateCommand::Modification::Operation::EMPLACE: {
                slot.tex3d = reinterpret_cast<FallbackTexture *>(m.tex3d.handle);
                slot.set_sampler_3d(m.tex3d.sampler);
                break;
            }
            case BindlessArrayUpdateCommand::Modification::Operation::REMOVE: {
                slot.tex3d = nullptr;
                slot.set_sampler_3d(0);
                break;
            }
            default: break;
        }
    }
}

}// namespace luisa::compute::fallback
