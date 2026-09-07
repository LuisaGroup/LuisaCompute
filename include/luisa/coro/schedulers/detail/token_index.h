#pragma once

#include <algorithm>
#include <limits>
#include <utility>

#include <luisa/core/logging.h>
#include <luisa/core/stl/vector.h>
#include <luisa/dsl/constant.h>
#include <luisa/dsl/coro_frame.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/sugar.h>

namespace luisa::compute::coro::detail {

/// Build a device callable that maps materialized coroutine trigger tokens to
/// dense scheduler queue indices. Default DSL-generated tokens are already the
/// dense sequence [1, subroutine_count), so that path is O(1). Explicit sparse
/// tokens use a sorted constant table and a device-side binary search, keeping
/// both AST size and lookup cost independent of the numeric token range.
template<typename... Args>
[[nodiscard]] Callable<uint(uint)> make_coro_token_index_callable(
    const Coroutine<void(Args...)> &coro) noexcept {
    auto count = coro.subroutine_count();
    LUISA_ASSERT(count != 0u &&
                     count <= std::numeric_limits<uint>::max(),
                 "Coroutine subroutine count ({}) is outside the supported uint range.",
                 count);

    luisa::vector<std::pair<uint, uint>> token_indices;
    token_indices.reserve(count);
    auto dense = true;
    for (auto i = 0u; i < count; i++) {
        auto token = coro.trigger_token(i);
        LUISA_ASSERT(
            (i == 0u && token == 0u) ||
                (i != 0u && token != 0u && token != CoroFrame::TERMINAL_TOKEN),
            "Invalid coroutine trigger token {} at subroutine {}.", token, i);
        dense &= token == i;
        token_indices.emplace_back(token, static_cast<uint>(i));
    }
    std::sort(token_indices.begin(), token_indices.end(),
              [](auto lhs, auto rhs) noexcept { return lhs.first < rhs.first; });
    for (auto i = 1u; i < token_indices.size(); i++) {
        LUISA_ASSERT(token_indices[i - 1u].first != token_indices[i].first,
                     "Duplicate coroutine trigger token {}.",
                     token_indices[i].first);
    }

    luisa::vector<uint> tokens;
    luisa::vector<uint> indices;
    tokens.reserve(count);
    indices.reserve(count);
    for (auto [token, index] : token_indices) {
        tokens.emplace_back(token);
        indices.emplace_back(index);
    }
    Constant<uint> token_table{tokens.data(), tokens.size()};
    Constant<uint> index_table{indices.data(), indices.size()};
    return Callable<uint(uint)>{
        [dense, count = static_cast<uint>(count),
         &token_table, &index_table](UInt target_token) noexcept {
            auto index = def(0u);
            $if (target_token == CoroFrame::TERMINAL_TOKEN) {
                // Queue zero represents an empty/terminated frame.
            }
            $elif (target_token == 0u) {
                // A materialized subroutine can only target a continuation or
                // terminate. Queue zero is scheduler-owned and must never be
                // emitted as a continuation token.
                unreachable("Coroutine subroutine emitted reserved entry token 0.");
            }
            $else {
                if (dense) {
                    $if (target_token < count) {
                        index = target_token;
                    }
                    $else {
                        unreachable("Coroutine subroutine emitted an unknown dense trigger token.");
                    };
                } else {
                    auto first = def(0u);
                    auto last = def(count);
                    $while (first < last) {
                        auto middle = first + (last - first) / 2u;
                        auto token = token_table[middle];
                        $if (token < target_token) {
                            first = middle + 1u;
                        }
                        $else {
                            last = middle;
                        };
                    };
                    $if (first < count) {
                        $if (token_table[first] == target_token) {
                            index = index_table[first];
                        }
                        $else {
                            unreachable("Coroutine subroutine emitted an unknown sparse trigger token.");
                        };
                    }
                    $else {
                        unreachable("Coroutine subroutine emitted an unknown sparse trigger token.");
                    };
                }
            };
            return index;
        }};
}

}// namespace luisa::compute::coro::detail
