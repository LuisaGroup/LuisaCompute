#pragma once

#include <algorithm>

#include <luisa/core/stl.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>

namespace luisa::compute::coro::radix_sort {

static constexpr uint hist_block_size = 128u;
static constexpr uint sm_count = 256u;
static constexpr uint onesweep_block_size = 128u;
static constexpr uint onesweep_item_count = 32u;
static constexpr uint warp_log = 5u;
static constexpr uint warp_size = 1u << warp_log;
static constexpr uint warp_mask = warp_size - 1u;
static constexpr uint all_mask = 0xffffffffu;
static constexpr uint bin_local_mask = 0x40000000u;
static constexpr uint bin_global_mask = 0x80000000u;
static constexpr uint bin_value_mask = 0x3fffffffu;

[[nodiscard]] static inline auto ceil_div(uint x, uint y) noexcept {
    return (x + y - 1u) / y;
}

struct temp_storage {
    Buffer<uint> bin_buffer;
    Buffer<uint> launch_count;
    Buffer<uint> hist_buffer;

    temp_storage() noexcept = default;
    temp_storage(Device &device, uint max_count, uint max_digit) noexcept
        : bin_buffer{device.create_buffer<uint>(
              ceil_div(max_count, onesweep_block_size * onesweep_item_count) * max_digit)},
          launch_count{device.create_buffer<uint>(1u)},
          hist_buffer{device.create_buffer<uint>(32u * max_digit)} {}
};

struct temp_storage_view {
    BufferView<uint> bin_buffer;
    BufferView<uint> launch_count;
    BufferView<uint> hist_buffer;

    temp_storage_view() noexcept = default;
    explicit temp_storage_view(temp_storage &storage) noexcept
        : bin_buffer{storage.bin_buffer.view()},
          launch_count{storage.launch_count.view()},
          hist_buffer{storage.hist_buffer.view()} {}
};

template<typename... Args>
class instance {

private:
    uint _digit{};
    uint _bit{};
    uint _hist_group{};
    uint _low_bit{};
    uint _high_bit{};
    uint _max_count{};
    bool _bucket_mode{};
    temp_storage_view _temp;
    luisa::vector<uint> _bit_split;

    Shader1D<Buffer<uint>, uint, uint, Args...> _hist_shader;
    Shader1D<Buffer<uint>> _accum_shader;
    Shader1D<Buffer<uint>, Buffer<uint>, uint> _copy_shader;
    Shader1D<Buffer<uint>, Buffer<uint>, Buffer<uint>, uint, Args...> _bucket_scatter_shader;
    Shader1D<Buffer<uint>, Buffer<uint>, Buffer<uint>, Buffer<uint>,
             Buffer<uint>, Buffer<uint>, Buffer<uint>, uint, uint, Args...>
        _onesweep_first_shader;
    Shader1D<Buffer<uint>, Buffer<uint>, Buffer<uint>, Buffer<uint>,
             Buffer<uint>, Buffer<uint>, Buffer<uint>, uint, uint, Args...>
        _onesweep_shader;
    Shader1D<Buffer<uint>> _clear_shader;

public:
    instance() noexcept = default;

    instance(Device &device, uint max_count, temp_storage &temp,
             Callable<uint(uint, Args...)> *get_key,
             Callable<uint(uint, Args...)> *get_value,
             Callable<uint(uint, Args...)> *get_key_from_set = nullptr,
             uint mode = 0u, uint digit_count = 128u,
             uint low_bit = 0u, uint high_bit = 31u) noexcept
        : _digit{std::max(digit_count, 1u)},
          _low_bit{low_bit},
          _high_bit{high_bit},
          _max_count{max_count},
          _bucket_mode{mode == 1u},
          _temp{temp} {

        LUISA_ASSERT(mode == 0u || mode == 1u,
                     "radix_sort mode must be 0 (radix) or 1 (bucket).");
        if (get_key_from_set == nullptr) { get_key_from_set = get_key; }

        _bit = 0u;
        while ((1u << _bit) < _digit) { _bit++; }
        LUISA_ASSERT(mode == 1u || ((1u << _bit) == _digit),
                     "radix_sort digit must be a power of two in radix mode.");

        _bit_split.clear();
        if (mode == 0u) {
            for (auto i = _low_bit; i <= _high_bit; i += _bit) {
                _bit_split.emplace_back(i);
            }
        } else {
            _bit_split.emplace_back(0u);
        }
        _hist_group = static_cast<uint>(_bit_split.size());
        auto digit = _digit;
        auto bit = _bit;
        auto hist_group = _hist_group;
        auto bit_split = _bit_split;

        Kernel1D hist_kernel = [=](
                                   BufferUInt hist_buffer, UInt item_count,
                                   UInt n, Var<Args>... args) noexcept {
            set_block_size(hist_block_size);
            Shared<uint> local_hist{digit * hist_group};
            $for (i, 0u, ceil_div(digit * hist_group, hist_block_size)) {
                auto slot = i * hist_block_size + thread_x();
                $if (slot < digit * hist_group) {
                    local_hist[slot] = 0u;
                };
            };
            sync_block();
            $for (i, 0u, item_count) {
                auto id = thread_x() + i * hist_block_size +
                          item_count * hist_block_size * block_x();
                $if (id < n) {
                    for (auto j = 0u; j < hist_group; j++) {
                        auto key = ((*get_key_from_set)(id, args...) >> bit_split[j]) &
                                   ((1u << bit) - 1u);
                        key = min(key, digit - 1u);
                        local_hist.atomic(key + j * digit).fetch_add(1u);
                    }
                };
            };
            sync_block();
            $for (i, 0u, ceil_div(digit * hist_group, hist_block_size)) {
                auto slot = i * hist_block_size + thread_x();
                $if (slot < digit * hist_group) {
                    hist_buffer.atomic(slot).fetch_add(local_hist[slot]);
                };
            };
        };

        Kernel1D accum_kernel = [=](BufferUInt hist_buffer) noexcept {
            set_block_size(32u);
            $if (thread_x() == 0u) {
                auto prefix = def(0u);
                $for (i, 0u, digit) {
                    auto slot = i + block_x() * digit;
                    auto value = hist_buffer.read(slot);
                    hist_buffer.write(slot, prefix);
                    prefix += value;
                };
            };
        };

        Kernel1D copy_kernel = [](BufferUInt src, BufferUInt dst, UInt n) noexcept {
            auto x = dispatch_x();
            $if (x < n) {
                dst.write(x, src.read(x));
            };
        };

        Kernel1D bucket_scatter_kernel = [=](
                                             BufferUInt key_out,
                                             BufferUInt value_out,
                                             BufferUInt offset,
                                             UInt n,
                                             Var<Args>... args) noexcept {
            auto id = dispatch_x();
            $if (id >= n) { $return(); };
            auto key_value = (*get_key)(id, args...);
            auto key = min(key_value, digit - 1u);
            auto rank = offset.atomic(key).fetch_add(1u);
            key_out.write(rank, key_value);
            value_out.write(rank, (*get_value)(id, args...));
        };

        auto make_onesweep_kernel = [&](bool first_pass) noexcept {
            return Kernel1D{[=](
                                BufferUInt key_in, BufferUInt key_out,
                                BufferUInt value_in, BufferUInt value_out,
                                BufferUInt launch_counter, BufferUInt hist_buffer,
                                BufferUInt bin, UInt low_bit, UInt n,
                                Var<Args>... args) noexcept {
                set_block_size(onesweep_block_size);
                set_warp_size(warp_size);

                Shared<uint> logical_block_id{1u};
                $if (thread_x() == 0u) {
                    logical_block_id[0u] = launch_counter.atomic(0u).fetch_add(1u);
                };

                Shared<uint> warp_prefix{onesweep_block_size / warp_size * digit};
                Shared<uint> block_bin{digit};
                ArrayUInt<onesweep_item_count> local_rank;
                ArrayUInt<onesweep_item_count> local_key;

                $for (i, 0u, ceil_div(onesweep_block_size / warp_size * digit, onesweep_block_size)) {
                    auto slot = i * onesweep_block_size + thread_x();
                    $if (slot < onesweep_block_size / warp_size * digit) {
                        warp_prefix[slot] = 0u;
                    };
                };
                sync_block();

                auto bid = logical_block_id[0u];
                auto lane_id = thread_x() & warp_mask;
                auto warp_id = thread_x() >> warp_log;
                auto block_offset = bid * onesweep_item_count * onesweep_block_size;
                auto warp_offset = onesweep_item_count * warp_size * warp_id;

                $for (i, 0u, onesweep_item_count) {
                    auto read_pos = block_offset + warp_offset + i * warp_size + lane_id;
                    auto key_value = def(all_mask);
                    $if (read_pos < n) {
                        if (first_pass) {
                            key_value = (*get_key)(read_pos, args...);
                        } else {
                            key_value = key_in.read(read_pos);
                        }
                    };
                    local_key[i] = key_value;
                    auto key = (key_value >> low_bit) & ((1u << bit) - 1u);
                    key = min(key, digit - 1u);

                    auto matched = def(all_mask);
                    for (auto bit_index = 0u; bit_index < bit; bit_index++) {
                        auto x = (key >> bit_index) & 1u;
                        auto y = warp_active_bit_or(x << lane_id);
                        matched = matched & (y ^ ite(x == 1u, 0u, all_mask));
                    }

                    auto prefix = popcount(matched & ((1u << lane_id) - 1u));
                    auto total = popcount(matched);
                    auto warp_pre = warp_prefix[warp_id * digit + key];
                    $if (prefix == 0u) {
                        warp_prefix[warp_id * digit + key] = warp_pre + total;
                    };
                    local_rank[i] = prefix + warp_pre;
                };
                sync_block();

                $for (i, 0u, ceil_div(digit, onesweep_block_size)) {
                    auto digit_index = i * onesweep_block_size + thread_x();
                    $if (digit_index < digit) {
                        auto digit_pre = def(0u);
                        $for (warp, 0u, ceil_div(onesweep_block_size, warp_size)) {
                            auto warp_pre = warp_prefix[warp * digit + digit_index];
                            warp_prefix[warp * digit + digit_index] = digit_pre;
                            digit_pre += warp_pre;
                        };

                        bin.volatile_write(bid * digit + digit_index,
                                           digit_pre | bin_local_mask);

                        auto ptr = cast<int>(bid) - 1;
                        auto global_pre = def(0u);
                        $while (ptr >= 0) {
                            auto read_value = def(0u);
                            $while (read_value == 0u) {
                                read_value = bin.volatile_read(ptr.cast<uint>() * digit + digit_index);
                            };
                            global_pre += read_value & bin_value_mask;
                            $if ((read_value & bin_global_mask) != 0u) {
                                $break;
                            };
                            ptr -= 1;
                        };

                        bin.volatile_write(bid * digit + digit_index,
                                           (global_pre + digit_pre) | bin_global_mask);
                        block_bin[digit_index] = global_pre + hist_buffer.read(digit_index);
                    };
                };
                sync_block();

                $for (i, 0u, onesweep_item_count) {
                    auto read_pos = block_offset + warp_offset + i * warp_size + lane_id;
                    $if (read_pos < n) {
                        auto rank = local_rank[i];
                        auto key_value = local_key[i];
                        auto key = (key_value >> low_bit) & ((1u << bit) - 1u);
                        key = min(key, digit - 1u);
                        rank += warp_prefix[warp_id * digit + key];
                        rank += block_bin[key];
                        key_out.write(rank, key_value);
                        auto value = def(0u);
                        if (first_pass) {
                            value = (*get_value)(read_pos, args...);
                        } else {
                            value = value_in.read(read_pos);
                        }
                        value_out.write(rank, value);
                    };
                };
            }};
        };

        Kernel1D clear_kernel = [](BufferUInt buffer) noexcept {
            buffer.write(dispatch_x(), 0u);
        };

        _hist_shader = device.compile(hist_kernel);
        _accum_shader = device.compile(accum_kernel);
        _copy_shader = device.compile(copy_kernel);
        _bucket_scatter_shader = device.compile(bucket_scatter_kernel);
        _onesweep_first_shader = device.compile(make_onesweep_kernel(true));
        _onesweep_shader = device.compile(make_onesweep_kernel(false));
        _clear_shader = device.compile(clear_kernel);

        LUISA_ASSERT(max_count < (1u << 30u), "radix_sort array is too large.");
    }

    void _encode_build_histogram(CommandList &commands, uint n,
                                 compute::detail::prototype_to_shader_invocation_t<Args>... args) noexcept {
        auto thread_count = hist_block_size * sm_count;
        commands << _clear_shader(_temp.hist_buffer).dispatch(_hist_group * _digit);
        if (thread_count >= n) {
            commands << _hist_shader(_temp.hist_buffer, 1u, n, args...)
                            .dispatch(ceil_div(n, hist_block_size) * hist_block_size);
        } else {
            commands << _hist_shader(_temp.hist_buffer,
                                     ceil_div(n, sm_count * hist_block_size),
                                     n, args...)
                            .dispatch(sm_count * hist_block_size);
        }
        commands << _accum_shader(_temp.hist_buffer).dispatch(_hist_group * 32u);
    }

    void build_histogram(Stream &stream, uint n,
                         compute::detail::prototype_to_shader_invocation_t<Args>... args) noexcept {
        auto commands = CommandList::create(3u);
        _encode_build_histogram(commands, n, args...);
        stream << commands.commit();
    }

    void sort(Stream &stream,
              BufferView<uint> temp_key, BufferView<uint> temp_value,
              BufferView<uint> key_out, BufferView<uint> value_out,
              uint n,
              compute::detail::prototype_to_shader_invocation_t<Args>... args) noexcept {
        LUISA_ASSERT(n <= _max_count, "radix_sort size {} exceeds capacity {}.", n, _max_count);
        auto commands = CommandList::create(3u + (_bucket_mode ? 2u : 3u * _hist_group));
        _encode_build_histogram(commands, n, args...);
        if (_bucket_mode) {
            commands << _copy_shader(_temp.hist_buffer.subview(0u, _digit),
                                     _temp.bin_buffer.subview(0u, _digit), _digit)
                            .dispatch(_digit);
            commands << _bucket_scatter_shader(key_out, value_out,
                                               _temp.bin_buffer.subview(0u, _digit),
                                               n, args...)
                            .dispatch(n);
            stream << commands.commit();
            return;
        }
        BufferView<uint> keys[2] = {temp_key, key_out};
        BufferView<uint> values[2] = {temp_value, value_out};
        auto out = _hist_group & 1u;
        for (auto i = 0u; i < _hist_group; i++) {
            commands << _clear_shader(_temp.launch_count).dispatch(1u);
            commands << _clear_shader(_temp.bin_buffer)
                            .dispatch(ceil_div(n, onesweep_block_size * onesweep_item_count) * _digit);
            if (i == 0u) {
                commands << _onesweep_first_shader(
                                keys[out ^ 1u], keys[out], values[out ^ 1u], values[out],
                                _temp.launch_count, _temp.hist_buffer.subview(i * _digit, _digit),
                                _temp.bin_buffer, _bit_split[i], n, args...)
                                .dispatch(ceil_div(n, onesweep_block_size * onesweep_item_count) *
                                          onesweep_block_size);
            } else {
                commands << _onesweep_shader(
                                keys[out ^ 1u], keys[out], values[out ^ 1u], values[out],
                                _temp.launch_count, _temp.hist_buffer.subview(i * _digit, _digit),
                                _temp.bin_buffer, _bit_split[i], n, args...)
                                .dispatch(ceil_div(n, onesweep_block_size * onesweep_item_count) *
                                          onesweep_block_size);
            }
            out ^= 1u;
        }
        LUISA_ASSERT(out == 0u, "radix_sort output buffer mismatch.");
        stream << commands.commit();
    }

    [[nodiscard]] auto sort_switch(
        Stream &stream, BufferView<uint> temp_key[2],
        BufferView<uint> temp_value[2], uint n,
        compute::detail::prototype_to_shader_invocation_t<Args>... args) noexcept {
        LUISA_ASSERT(n <= _max_count, "radix_sort size {} exceeds capacity {}.", n, _max_count);
        auto commands = CommandList::create(3u + (_bucket_mode ? 2u : 3u * _hist_group));
        _encode_build_histogram(commands, n, args...);
        if (_bucket_mode) {
            commands << _copy_shader(_temp.hist_buffer.subview(0u, _digit),
                                     _temp.bin_buffer.subview(0u, _digit), _digit)
                            .dispatch(_digit);
            commands << _bucket_scatter_shader(temp_key[1u], temp_value[1u],
                                               _temp.bin_buffer.subview(0u, _digit),
                                               n, args...)
                            .dispatch(n);
            stream << commands.commit();
            return 1u;
        }
        auto out = 1u;
        for (auto i = 0u; i < _hist_group; i++) {
            commands << _clear_shader(_temp.launch_count).dispatch(1u);
            commands << _clear_shader(_temp.bin_buffer)
                            .dispatch(ceil_div(n, onesweep_block_size * onesweep_item_count) * _digit);
            if (i == 0u) {
                commands << _onesweep_first_shader(
                                temp_key[out ^ 1u], temp_key[out],
                                temp_value[out ^ 1u], temp_value[out],
                                _temp.launch_count, _temp.hist_buffer.subview(i * _digit, _digit),
                                _temp.bin_buffer, _bit_split[i], n, args...)
                                .dispatch(ceil_div(n, onesweep_block_size * onesweep_item_count) *
                                          onesweep_block_size);
            } else {
                commands << _onesweep_shader(
                                temp_key[out ^ 1u], temp_key[out],
                                temp_value[out ^ 1u], temp_value[out],
                                _temp.launch_count, _temp.hist_buffer.subview(i * _digit, _digit),
                                _temp.bin_buffer, _bit_split[i], n, args...)
                                .dispatch(ceil_div(n, onesweep_block_size * onesweep_item_count) *
                                          onesweep_block_size);
            }
            out ^= 1u;
        }
        stream << commands.commit();
        return out ^ 1u;
    }
};

}// namespace luisa::compute::coro::radix_sort
