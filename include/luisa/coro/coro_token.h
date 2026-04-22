//
// Created by Mike on 2024/5/10.
//

#pragma once

namespace luisa::compute::coroutine {
using CoroToken = unsigned int;
constexpr CoroToken coro_token_entry = 0u;
constexpr CoroToken coro_token_terminal = 0x8000'0000u;
constexpr CoroToken coro_token_valid_mask = 0x7fff'ffffu;
}// namespace luisa::compute::coroutine
