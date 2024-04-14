#pragma once
#include "./../types/vec.hpp"

namespace luisa::shader {
struct [[builtin("indirect_buffer")]] IndirectBuffer {
    [[callop("INDIRECT_SET_DISPATCH_COUNT")]] void set_dispatch_count(uint32 count);
    [[callop("INDIRECT_SET_DISPATCH_KERNEL")]]  void set_kernel(uint32 offset, uint3 block_size, uint3 dispatch_size, uint32 kernel_id) const noexcept;
};
}