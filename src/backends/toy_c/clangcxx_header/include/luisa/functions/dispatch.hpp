#pragma once
#include "./../attributes.hpp"
#include "./../type_traits.hpp"
#include "./../types/vec.hpp"

namespace luisa::shader {

[[expr("dispatch_id")]] extern uint3 dispatch_id();
[[expr("block_id")]] extern uint3 block_id();
[[expr("thread_id")]] extern uint3 thread_id();
[[expr("dispatch_size")]] extern uint3 dispatch_size();
[[expr("kernel_id")]] extern uint32 kernel_id();
[[expr("object_id")]] extern uint32 object_id();

[[callop("SYNCHRONIZE_BLOCK")]] extern void sync_block();

// raster
[[callop("RASTER_DISCARD")]] extern void discard();
}// namespace luisa::shader