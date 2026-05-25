#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/runtime/stream_event.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/rhi/command_encoder.h>
#include <luisa/core/stl.h>
#include <luisa/runtime/shader.h>
#include <luisa/dsl/sugar.h>
namespace luisa::compute {
namespace radix_sort {
const uint HIST_BLOCK_SIZE = 128;
const uint SM_COUNT = 256;
const uint ONESWEEP_BLOCK_SIZE = 128;
const uint ONESWEEP_ITEM_COUNT = 32;
const uint WARP_LOG = 5;
const uint WARP_SIZE = 1 << WARP_LOG;
const uint WARP_MASK = WARP_SIZE - 1;
const uint ALL_MASK = 0xffff'ffff;
const uint BIN_LOCAL_MASK = 0x4000'0000;
const uint BIN_GLOBAL_MASK = 0x8000'0000;
const uint BIN_VAL_MASK = 0x3fff'ffff;
static inline uint ceil_div(uint x, uint y) {
    return (x + y - 1) / y;
}
struct temp_storage {
    Buffer<uint> bin_buffer;
    Buffer<uint> launch_count;
    Buffer<uint> hist_buffer;
    temp_storage(Device device, uint maxn, uint max_digit) noexcept
        : bin_buffer{device.create_buffer<uint>(
              ceil_div(maxn, ONESWEEP_BLOCK_SIZE * ONESWEEP_ITEM_COUNT) * max_digit)},
          launch_count{device.create_buffer<uint>(1u)},
          hist_buffer{device.create_buffer<uint>(32 * max_digit)} {}
    temp_storage() noexcept = default;
};
struct temp_storage_view {
    BufferView<uint> bin_buffer;
    BufferView<uint> launch_count;
    BufferView<uint> hist_buffer;
    temp_storage_view() noexcept = default;
    temp_storage_view(temp_storage &storage) noexcept
        : bin_buffer{storage.bin_buffer.view()},
          launch_count{storage.launch_count.view()},
          hist_buffer{storage.hist_buffer.view()} {}
};
template<typename... Args>
class instance {
    //2 mod: limit digit mode: search from a get_key, make single histogram and return
    // or make a histogram for bit segment, and run onesweep multiple times
    //return index intead of key
    // extra buffer we need: key_out, intermediate_key. can be passed in
public:

private:
    uint DIGIT{};
    uint BIT{};
    uint HIST_GROUP{};
    uint low_bit{};
    uint high_bit{};
    uint MAXN{};
    temp_storage_view _temp;
    luisa::vector<uint> bit_split;
    Shader1D<Buffer<uint> /*hist_buffer*/, uint /*item_count*/, uint /*n*/, Args...> hist_shader;
    Shader1D<Buffer<uint> /*hist_buffer*/> accum_shader;
    Shader1D<Buffer<uint>, Buffer<uint> /*key_out*/,
             Buffer<uint>, Buffer<uint> /*val_out*/,
             Buffer<uint> /*launch_counter*/, Buffer<uint> /*hist_buffer*/,
             Buffer<uint> /*bin*/, uint /*low_bit*/, uint /*n*/, Args...>
        onesweep_first_shader;
    Shader1D<Buffer<uint> /*key_in*/, Buffer<uint> /*key_out*/,
             Buffer<uint> /*val_in*/, Buffer<uint> /*val_out*/,
             Buffer<uint> /*launch_counter*/, Buffer<uint> /*hist_buffer*/,
             Buffer<uint> /*bin*/, uint /*low_bit*/, uint /*n*/, Args...>
        onesweep_shader;
    Shader1D<Buffer<uint> /*v*/> clear_shader;
public:
    instance() noexcept = default;
    ///initialize radix_sort
    ///@param device: device to initialize
    ///@param maxn: maximum number of elements
    ///@param get_key: callable to get key with index
    ///@param get_val: callable to get value with index
    ///@param get_key_set: callable to get key from unordered set, if nullptr, use get_key
    ///@param mode: 0 for normal radix sort, 1 for bucket sort, when mode=1, digit is the number of buckets
    ///@param digit: number of digits used for one pass of bucket sort, default 128
    ///@param low_bit: lowest bit of radix sort, will use bit in [low_bit,high_bit]
    ///@param high_bit: highest bit of radix sort
    instance(Device device, uint maxn, temp_storage &temp,
             Callable<uint(uint, Args...)> *get_key, Callable<uint(uint, Args...)> *get_val,
             Callable<uint(uint, Args...)> *get_key_set = nullptr,
             uint mode = 0, uint digit = 128, uint low_bit = 0, uint high_bit = 31) : DIGIT{digit}, low_bit{low_bit}, high_bit{high_bit}, MAXN{maxn}, _temp{temp} {
        LUISA_ASSERT(mode == 0 || mode == 1, "mode should be 0 for radix sort and 1 for bucket sort!");
        BIT = 0;
        while ((1u << BIT) < DIGIT) {
            BIT += 1;
        }
        LUISA_ASSERT(mode == 1 || ((1 << BIT) == DIGIT), "radix sort should have digit as power of 2");
        if (get_key_set == nullptr) {
            get_key_set = get_key;
        }
        bit_split.clear();
        if (mode == 0) {
            for (int i = low_bit; i <= high_bit; i += BIT) {
                bit_split.push_back(i);
            }
        } else {
            bit_split.push_back(0);
        }
        HIST_GROUP = bit_split.size();
        Kernel1D get_hist = [&](BufferUInt hist_buffer, UInt item_count, UInt n, Var<Args>... args) {
            set_block_size(HIST_BLOCK_SIZE);
            Shared<uint> local_hist{DIGIT * HIST_GROUP};
            $for (i, 0u, ceil_div(DIGIT * HIST_GROUP, HIST_BLOCK_SIZE)) {
                $if (i * HIST_BLOCK_SIZE + thread_x() < DIGIT * HIST_GROUP) {
                    local_hist[i * HIST_BLOCK_SIZE + thread_x()] = 0;
                };
            };
            sync_block();
            $for (i, 0u, item_count) {
                auto id = thread_x() + i * HIST_BLOCK_SIZE + item_count * HIST_BLOCK_SIZE * block_x();
                $if (id < n) {
                    for (auto j = 0u; j < HIST_GROUP; ++j) {
                        auto index = ((*get_key_set)(id, args...) >> bit_split[j]) & ((1 << BIT) - 1);
                        local_hist.atomic(index + j * DIGIT).fetch_add(1u);
                    }
                };
            };
            sync_block();
            $for (i, 0u, ceil_div(DIGIT * HIST_GROUP, HIST_BLOCK_SIZE)) {
                $if (i * HIST_BLOCK_SIZE + thread_x() < DIGIT * HIST_GROUP) {
                    auto cur_dig = i * HIST_BLOCK_SIZE + thread_x();
                    hist_buffer.atomic(cur_dig).fetch_add(local_hist[cur_dig]);
                };
            };
        };
        Kernel1D get_accum = [&](BufferUInt hist_buffer) {
            set_block_size(32);
            $if (thread_x() == 0) {
                auto prefix = def<uint>(0u);
                $for (i, 0u, DIGIT) {

                    auto cur_dig = i + block_x() * DIGIT;
                    auto val = hist_buffer.read(cur_dig);
                    hist_buffer.write(cur_dig, prefix);
                    prefix += val;
                };
            };
        };
        ExternalCallable<void()> thread_fence{device.backend_name() == "cuda" ?
                                                  "([] { __threadfence(); })" :
                                                  "AllMemoryBarrier"};
        ExternalCallable<uint(uint)> match_any{device.backend_name() == "cuda" ?
                                                   "([](unsigned int x) { return __match_any_sync(0xffff'ffff,x); })" :
                                                   "([](unsigned int x) { return 0u; })"};
        uint ONESWEEP_TOT_BLOCK = ceil_div(MAXN, ONESWEEP_BLOCK_SIZE * ONESWEEP_ITEM_COUNT);
        bool is_first = true;
        auto onesweep_base = [&](BufferUInt key_in, BufferUInt key_out,
                                 BufferUInt val_in, BufferUInt val_out,
                                 BufferUInt launch_counter, BufferUInt hist_buffer,
                                 BufferUInt bin, UInt low_bit, UInt n, Var<Args>... args) {
            /// break_up:
            /// block(0,(warp(id=0,item=0)(0, 1, 2, ..., WARP_SIZE) warp(0,1) warp(0,2) ... | warp(1,0) ...| warp(2,0) ... | ...) block(1) block(2) ...
            set_block_size(ONESWEEP_BLOCK_SIZE);
            Shared<uint> block_id{1};
            $if (thread_x() == 0) {
                block_id[0] = launch_counter.atomic(0u).fetch_add(1u);
            };
            Shared<uint> warp_prefix{ONESWEEP_BLOCK_SIZE / WARP_SIZE * DIGIT};
            Shared<uint> block_bin{DIGIT};
            //Shared<uint> local_rank{ONESWEEP_BLOCK_SIZE * ONESWEEP_ITEM_COUNT};
            ArrayUInt<ONESWEEP_ITEM_COUNT> local_rank;
            ArrayUInt<ONESWEEP_ITEM_COUNT> local_key;

            $for (i, 0u, ceil_div(ONESWEEP_BLOCK_SIZE / WARP_SIZE * DIGIT, ONESWEEP_BLOCK_SIZE)) {
                $if (i * ONESWEEP_BLOCK_SIZE + thread_x() < ONESWEEP_BLOCK_SIZE / WARP_SIZE * DIGIT) {
                    warp_prefix[i * ONESWEEP_BLOCK_SIZE + thread_x()] = 0u;
                };
            };
            sync_block();
            auto bid = block_id[0];
            ///get warp level prefix and rank(according to single digit)
            auto lane_id = thread_x() & WARP_MASK;
            auto warp_id = thread_x() >> WARP_LOG;
            auto block_offset = bid * ONESWEEP_ITEM_COUNT * ONESWEEP_BLOCK_SIZE;
            auto warp_offset = ONESWEEP_ITEM_COUNT * WARP_SIZE * warp_id;
            $for (i, 0u, ONESWEEP_ITEM_COUNT) {
                auto read_pos = block_offset + warp_offset + i * WARP_SIZE + lane_id;
                auto key = def<uint>(ALL_MASK);
                $if (read_pos < n) {
                    if (is_first) {
                        key = (*get_key)(read_pos, args...);
                    } else {
                        key = key_in.read(read_pos);
                    }
                };
                local_key[i] = key;
                key = (key >> low_bit) & ((1 << BIT) - 1);
                auto matched = def<uint>(0xffff'ffff);
                ///general case function
                if (device.backend_name() == "cuda") {
                    matched = match_any(key);
                } else {

                    for (auto j = 0u; j < BIT; ++j) {
                        auto x = ((key >> j) & 1);
                        auto y = warp_active_bit_or(x << lane_id);
                        matched = matched & (y ^ (ite(x == 1u, 0u, 0xffff'ffff)));
                    }
                }
                auto prefix = popcount(matched & ((1u << lane_id) - 1));
                auto total = popcount(matched);

                auto warp_pre = warp_prefix[warp_id * DIGIT + key];
                $if (prefix == 0u) {
                    warp_prefix[warp_id * DIGIT + key] = warp_pre + total;
                };
                local_rank[i] = prefix + warp_pre;
            };
            sync_block();
            //get block level prefix, the calculate global offset
            $for (i, 0u, ceil_div(DIGIT, ONESWEEP_BLOCK_SIZE)) {
                auto cur_dig = i * ONESWEEP_BLOCK_SIZE + thread_x();
                $if (cur_dig < DIGIT) {
                    auto digit_pre = def<uint>(0u);
                    $for (cur_warp, 0u, ceil_div(ONESWEEP_BLOCK_SIZE, WARP_SIZE)) {
                        auto warp_pre = warp_prefix[cur_warp * DIGIT + cur_dig];
                        warp_prefix[cur_warp * DIGIT + cur_dig] = digit_pre;
                        digit_pre += warp_pre;
                    };
                    bin.write(bid * DIGIT + cur_dig, digit_pre | BIN_LOCAL_MASK);
                    auto ptr = def<int>(bid - 1);
                    auto global_pre = def<uint>(0u);
                    $while (ptr >= 0) {
                        auto read_v = def<uint>(0u);
                        if (device.backend_name() == "cuda") {
                            $while ((read_v == 0u)) {
                                read_v = bin.read(ptr * DIGIT + cur_dig);
                                thread_fence();//flush cache
                            };
                        } else if (device.backend_name() == "metal") {
                            auto num_tries = def(0u);
                            $while ((read_v == 0u)) {
                                ExternalCallable<uint(Buffer<uint>, uint)> read_coherent{"buffer_read_coherent"};
                                read_v = read_coherent(bin, ptr * DIGIT + cur_dig);
                                num_tries += 1;
                                $if (num_tries > 10000u) {
                                    device_log("read_coherent failed: {}", ptr * DIGIT + cur_dig);
                                    $break;
                                };
                            };
                        } else {//dx cache is flushed with globallycoherent
                            read_v = bin.read(ptr * DIGIT + cur_dig);
                            $if (!warp_active_all(read_v != 0u)) {
                                $continue;
                            };
                        }
                        global_pre += (read_v & BIN_VAL_MASK);
                        $if ((read_v & BIN_GLOBAL_MASK) != 0) {
                            $break;
                        };
                        ptr -= 1;
                    };
                    bin.write(bid * DIGIT + cur_dig, (global_pre + digit_pre) | BIN_GLOBAL_MASK);
                    block_bin[cur_dig] = global_pre + hist_buffer.read(cur_dig);
                };
            };
            sync_block();
            //get final rank
            $for (i, 0u, ONESWEEP_ITEM_COUNT) {
                auto read_pos = block_offset + warp_offset + i * WARP_SIZE + lane_id;
                $if (read_pos < n) {
                    UInt warp_rank = local_rank[i];
                    auto key_v = local_key[i];
                    auto key = (key_v >> low_bit) & ((1 << BIT) - 1);
                    warp_rank += warp_prefix[warp_id * DIGIT + key];//offset between warp in a block
                    warp_rank += block_bin[key];                    //offset between block in global
                    key_out.write(warp_rank, key_v);
                    auto val = def<uint>(0u);
                    if (is_first) {
                        val = (*get_val)(read_pos, args...);
                    } else {
                        val = val_in.read(read_pos);
                    }
                    val_out.write(warp_rank, val);
                };
            };
        };
        Kernel1D onesweep_first = onesweep_base;
        is_first = false;
        Kernel1D onesweep = onesweep_base;
        Kernel1D clear = [](BufferUInt v) {
            auto x = dispatch_x();
            v.write(x, 0u);
        };
        clear_shader = device.compile(clear);
        hist_shader = device.compile(get_hist);
        accum_shader = device.compile(get_accum);
        onesweep_shader = device.compile(onesweep);
        onesweep_first_shader = device.compile(onesweep_first);
        if (maxn >= (1 << 30)) {
            LUISA_ERROR("unsupported array size!");
        }
    }
    void build_histogram(Stream &stream, uint n, detail::prototype_to_shader_invocation_t<Args>... args) {
        auto thread_count = HIST_BLOCK_SIZE * SM_COUNT;
        stream << clear_shader(_temp.hist_buffer).dispatch(HIST_GROUP * DIGIT);
        if (thread_count >= n) {
            stream << hist_shader(_temp.hist_buffer, 1, n, args...).dispatch(ceil_div(n, HIST_BLOCK_SIZE) * HIST_BLOCK_SIZE);
        } else {
            stream << hist_shader(_temp.hist_buffer, ceil_div(n, SM_COUNT * HIST_BLOCK_SIZE), n, args...)
                          .dispatch(SM_COUNT * HIST_BLOCK_SIZE);
        }
        stream << accum_shader(_temp.hist_buffer).dispatch(HIST_GROUP * 32);
    }
    ///standard sort, output to a certain array
    void sort(Stream &stream, BufferView<uint> temp_key, BufferView<uint> temp_val,
              BufferView<uint> key_out, BufferView<uint> val_out, uint n, detail::prototype_to_shader_invocation_t<Args>... args) {
        LUISA_ASSERT(n <= MAXN, "array size too large! n={},MAXN={}", n, MAXN);
        build_histogram(stream, n, args...);
        BufferView<uint> keys[2] = {temp_key, key_out};
        BufferView<uint> vals[2] = {temp_val, val_out};
        uint out = HIST_GROUP & 1;
        for (int i = 0; i < HIST_GROUP; ++i) {
            stream << clear_shader(_temp.launch_count).dispatch(1u);
            stream << clear_shader(_temp.bin_buffer).dispatch(ceil_div(n, ONESWEEP_BLOCK_SIZE * ONESWEEP_ITEM_COUNT) * DIGIT);
            if (i == 0) {
                stream << onesweep_first_shader(keys[out ^ 1], keys[out], vals[out ^ 1], vals[out], _temp.launch_count,
                                                _temp.hist_buffer.subview(i * DIGIT, DIGIT), _temp.bin_buffer, bit_split[i], n, args...)
                              .dispatch(ceil_div(n, ONESWEEP_BLOCK_SIZE * ONESWEEP_ITEM_COUNT) * ONESWEEP_BLOCK_SIZE);
            } else {
                stream << onesweep_shader(keys[out ^ 1], keys[out], vals[out ^ 1], vals[out], _temp.launch_count,
                                          _temp.hist_buffer.subview(i * DIGIT, DIGIT), _temp.bin_buffer, bit_split[i], n, args...)
                              .dispatch(ceil_div(n, ONESWEEP_BLOCK_SIZE * ONESWEEP_ITEM_COUNT) * ONESWEEP_BLOCK_SIZE);
            }
            out ^= 1;
        }
        LUISA_ASSERT(out == 0, "output buffer not match");
    }
    ///efficient sort, temp storage 0 could overlap with input, but output could be arbitrary
    uint sort_switch(Stream &stream, BufferView<uint> temp_key[2], BufferView<uint> temp_val[2], uint n, detail::prototype_to_shader_invocation_t<Args>... args) {
        LUISA_ASSERT(n <= MAXN, "array size too large! n={}", n);
        build_histogram(stream, n, args...);
        uint out = 1u;
        for (int i = 0; i < HIST_GROUP; ++i) {
            stream << clear_shader(_temp.launch_count).dispatch(1u);
            stream << clear_shader(_temp.bin_buffer).dispatch(ceil_div(n, ONESWEEP_BLOCK_SIZE * ONESWEEP_ITEM_COUNT) * DIGIT);
            if (i == 0) {
                stream << onesweep_first_shader(temp_key[out ^ 1], temp_key[out], temp_val[out ^ 1], temp_val[out], _temp.launch_count,
                                                _temp.hist_buffer.subview(i * DIGIT, DIGIT), _temp.bin_buffer, bit_split[i], n, args...)
                              .dispatch(ceil_div(n, ONESWEEP_BLOCK_SIZE * ONESWEEP_ITEM_COUNT) * ONESWEEP_BLOCK_SIZE);
            } else {
                stream << onesweep_shader(temp_key[out ^ 1], temp_key[out], temp_val[out ^ 1], temp_val[out], _temp.launch_count,
                                          _temp.hist_buffer.subview(i * DIGIT, DIGIT), _temp.bin_buffer, bit_split[i], n, args...)
                              .dispatch(ceil_div(n, ONESWEEP_BLOCK_SIZE * ONESWEEP_ITEM_COUNT) * ONESWEEP_BLOCK_SIZE);
            }

            out ^= 1;
        }
        return out ^ 1;
    }
};
}
}// namespace luisa::compute::radix_sort