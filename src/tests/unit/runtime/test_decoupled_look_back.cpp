#include "ut/ut.hpp"
#include "test_device.h"
// Decoupled look-back parallel primitive test demonstrating
// efficient parallel prefix sum (scan) algorithms using GPU warp operations.

#include <cmath>
#include <numeric>
#include <luisa/luisa-compute.h>
#include "luisa/core/basic_traits.h"
#include "luisa/core/logging.h"
#include "luisa/core/stl/string.h"
#include "luisa/dsl/resource.h"
#include "luisa/dsl/struct.h"
#include "luisa/dsl/sugar.h"
#include "luisa/dsl/func.h"
#include "luisa/dsl/var.h"
#include "luisa/dsl/builtin.h"
#include <cstddef>

namespace luisa::parallel_primitive {
using namespace luisa::compute;
// Shuffle down operation for warp-level communication
template<typename Type4Byte>
Var<Type4Byte> ShuffleDown(Var<Type4Byte> &input,
                           luisa::compute::UInt curr_lane_id,
                           luisa::compute::UInt offset,
                           luisa::compute::UInt last_lane = 32u) {
    luisa::compute::UInt src_lane = curr_lane_id + offset;
    Var<Type4Byte> result = warp_read_lane(input, src_lane);
    $if (src_lane > last_lane) {
        result = input;
    };
    return result;
};

// Get lane mask for threads with lane_id >= given value
static Callable get_lane_mask_ge = []() { return 0xFFFFFFFFu << warp_lane_id(); };
static constexpr inline bool is_power_of_two(int x) {
    return (x & (x - 1)) == 0;
}
template<size_t LOGIC_WARP_SIZE>
inline luisa::compute::UInt warp_mask(luisa::compute::UInt warp_id) {
    constexpr bool is_pow_of_two = is_power_of_two(LOGIC_WARP_SIZE);
    constexpr bool is_arch_warp = (LOGIC_WARP_SIZE == 32);

    luisa::compute::UInt member_mask = 0xFFFFFFFFu >> (32 - LOGIC_WARP_SIZE);

    if constexpr (is_pow_of_two && !is_arch_warp) {
        member_mask <<= warp_id * luisa::compute::UInt(LOGIC_WARP_SIZE);
    };

    return member_mask;
}
namespace details {
using namespace luisa::compute;

// Warp-level reduction using shuffle operations
template<typename Type4Byte, size_t LOGIC_WARP_SIZE = 32>
struct WarpReduceShfl {
    constexpr static bool IS_ARCH_WARP = (LOGIC_WARP_SIZE == 32);
    compute::UInt lane_id;
    compute::UInt warp_id;
    compute::UInt member_mask;

    WarpReduceShfl()
        : lane_id(warp_lane_id()), warp_id(IS_ARCH_WARP ? 0 : warp_lane_id() / compute::UInt(LOGIC_WARP_SIZE)), member_mask(IS_ARCH_WARP ? 0xFFFFFFFFu : warp_mask<LOGIC_WARP_SIZE>(warp_id)) {
        if constexpr (!IS_ARCH_WARP) {
            lane_id = lane_id % compute::UInt(LOGIC_WARP_SIZE);
        }
    }

    template<typename ReduceOp>
    Var<Type4Byte> Reduce(const Var<Type4Byte> &input, ReduceOp op, UInt valid_item = LOGIC_WARP_SIZE) {
        return ReduceStep(input, op, valid_item - 1);
    }

    template<bool HEAD_SEGMENT, typename FlagT, typename ReduceOp>
    Var<Type4Byte> SegmentReduce(const Var<Type4Byte> &input,
                                 const Var<FlagT> &flag,
                                 ReduceOp reduce_op,
                                 UInt valid_item = LOGIC_WARP_SIZE) {
        UInt warp_flags = compute::warp_active_bit_mask(flag == 1u).x;
        if constexpr (HEAD_SEGMENT) {
            warp_flags >>= 1;
        };

        warp_flags &= get_lane_mask_ge();

        if constexpr (!IS_ARCH_WARP) {
            warp_flags = (warp_flags & member_mask) >> (warp_id * UInt(LOGIC_WARP_SIZE));
        };

        warp_flags |= 1u << (UInt(LOGIC_WARP_SIZE) - 1u);

        UInt last_lane = compute::ctz(warp_flags);

        return ReduceStep(input, reduce_op, last_lane);
    }

private:
    template<typename ReduceOp>
    Var<Type4Byte> ReduceStep(const Var<Type4Byte> &input, ReduceOp op, UInt valid_item = LOGIC_WARP_SIZE) {
        Var<Type4Byte> result = input;

        compute::UInt offset = 1u;
        $while (offset < warp_lane_count()) {
            Var<Type4Byte> temp = ShuffleDown(result, lane_id, offset, valid_item);
            $if (lane_id + offset <= valid_item) {
                result = op(result, temp);
            };
            offset <<= 1;
        };
        return result;
    }
};
}// namespace details

template<typename Type4Byte, size_t WARP_SIZE = 32>
class WarpReduce {
public:
    WarpReduce() {
        m_shared_mem = new Shared<Type4Byte>{WARP_SIZE};
    }
    WarpReduce(Shared<Type4Byte> *shared_mem)
        : m_shared_mem(shared_mem) {
    }
    ~WarpReduce() = default;

public:
    template<typename FlagT, typename ReduceOp>
    Var<Type4Byte> TailSegmentedReduce(const Var<Type4Byte> &d_in,
                                       const Var<FlagT> &flag,
                                       ReduceOp redecu_op,
                                       compute::UInt valid_item = WARP_SIZE) {
        Var<Type4Byte> result;
        result = details::WarpReduceShfl<Type4Byte, WARP_SIZE>().template SegmentReduce<false>(d_in, flag, redecu_op, valid_item);
        return result;
    }

    template<typename FlagT>
    Var<Type4Byte> TailSegmentedSum(const Var<Type4Byte> &d_in, const Var<FlagT> &flag, compute::UInt valid_item = WARP_SIZE) {
        Var<Type4Byte> result;
        result = details::WarpReduceShfl<Type4Byte, WARP_SIZE>().template SegmentReduce<false>(d_in, flag, [](const Var<Type4Byte> &a, const Var<Type4Byte> &b) noexcept { return a + b; }, valid_item);
        return result;
    }

private:
    Shared<Type4Byte> *m_shared_mem;
};

// Swizzle scan operator for reversing operand order
template<typename ScanOp>
struct SwizzleScanOp {
    ScanOp scan_op;

public:
    explicit SwizzleScanOp(ScanOp scan_op)
        : scan_op(scan_op) {
    }

    template<typename Type>
    compute::Var<Type> operator()(const compute::Var<Type> &a, const compute::Var<Type> &b) const noexcept {
        compute::Var<Type> _a = a;
        compute::Var<Type> _b = b;
        return scan_op(_b, _a);
    }
};

template<typename T>
using SmemType = luisa::compute::Shared<T>;
template<typename T>
using SmemTypePtr = luisa::compute::Shared<T> *;

// Tile status for decoupled look-back scan
enum class ScanTileStatus : uint {
    SCAN_TILE_OBB,         // out-of-bounds
    SCAN_TILE_INVALID = 99,// not yet valid
    SCAN_TILE_PARTIAL,     // tile aggregate is available
    SCAN_TILE_INCLUSIVE,   // inclusive tile prefix is available
};

// Tile descriptor containing status and value
template<typename StatusWord, typename T>
struct TileDescriptor {
    StatusWord status;
    T value;

    TileDescriptor()
        : status(static_cast<StatusWord>(ScanTileStatus::SCAN_TILE_INVALID)), value(T(0)) {};

    TileDescriptor(const TileDescriptor &) = default;
    TileDescriptor &operator=(const TileDescriptor &) = default;
};

template<typename T>
struct ScanTileStateViewer {

    using StatusWordT = compute::uint;

    constexpr static size_t TILE_STATUS_PADDING = 32;

    compute::BufferVar<compute::uint> &d_tile_status;
    compute::BufferVar<T> &d_tile_partial;
    compute::BufferVar<T> &d_tile_inclusive;

    ScanTileStateViewer(compute::BufferVar<StatusWordT> &tile_status,
                        compute::BufferVar<T> &tile_partial,
                        compute::BufferVar<T> &tile_inclusive)
        : d_tile_status(tile_status), d_tile_partial(tile_partial), d_tile_inclusive(tile_inclusive) {};

    void SetInclusive(compute::Int tile_index, const compute::Var<T> &tile_inclusive) noexcept {
        d_tile_inclusive.volatile_write(compute::Int(TILE_STATUS_PADDING) + tile_index, tile_inclusive);
        d_tile_status.volatile_write(compute::Int(TILE_STATUS_PADDING) + tile_index,
                                     compute::def(StatusWordT(ScanTileStatus::SCAN_TILE_INCLUSIVE)));
    };

    void SetPartial(compute::Int tile_index, const compute::Var<T> &tile_partial) noexcept {
        d_tile_partial.volatile_write(compute::Int(TILE_STATUS_PADDING) + tile_index, tile_partial);
        d_tile_status.volatile_write(compute::Int(TILE_STATUS_PADDING) + tile_index,
                                     compute::def(StatusWordT(ScanTileStatus::SCAN_TILE_PARTIAL)));
    };

    template<typename DelayT>
    void WaitForValid(compute::Int tile_index, compute::Var<StatusWordT> &out_status, compute::Var<T> &out_value, DelayT delay) noexcept {
        out_status = d_tile_status.volatile_read(compute::Int(TILE_STATUS_PADDING) + tile_index);
        $while (compute::warp_active_any(out_status == StatusWordT(ScanTileStatus::SCAN_TILE_INVALID))) {
            delay();
            out_status = d_tile_status.volatile_read(compute::Int(TILE_STATUS_PADDING) + tile_index);
        };
        $if (out_status == StatusWordT(ScanTileStatus::SCAN_TILE_PARTIAL)) {
            out_value = d_tile_partial.volatile_read(compute::Int(TILE_STATUS_PADDING) + tile_index);
        }
        $else {
            out_value = d_tile_inclusive.volatile_read(compute::Int(TILE_STATUS_PADDING) + tile_index);
        };
    };
};
static void InitializeWardStatus(compute::UInt num_tile, compute::BufferVar<compute::uint> &d_tile_status) noexcept {
    compute::UInt tile_idx = compute::dispatch_id().x;
    $if (tile_idx < num_tile) {
        d_tile_status.write(compute::UInt(ScanTileStateViewer<int>::TILE_STATUS_PADDING) + tile_idx,
                            compute::def(ScanTileStateViewer<int>::StatusWordT(ScanTileStatus::SCAN_TILE_INVALID)));
    };
    $if (compute::block_id().x == 0 & compute::thread_x() < compute::UInt(ScanTileStateViewer<int>::TILE_STATUS_PADDING)) {
        d_tile_status.write(compute::thread_x(), compute::def(ScanTileStateViewer<int>::StatusWordT(ScanTileStatus::SCAN_TILE_OBB)));
    };
};

// 2D variant: tile_idx computed from dispatch_id, padding handled by first 32 threads
static void InitializeWardStatus2D(compute::UInt num_tile, compute::BufferVar<compute::uint> &d_tile_status) noexcept {
    using StatusWordT = ScanTileStateViewer<int>::StatusWordT;
    compute::UInt tile_idx = compute::dispatch_id().y * compute::dispatch_size().x + compute::dispatch_id().x;
    $if (tile_idx < 32u) {
        d_tile_status.write(tile_idx, compute::def(StatusWordT(ScanTileStatus::SCAN_TILE_OBB)));
    };
    $if (tile_idx < num_tile) {
        d_tile_status.write(32u + tile_idx, compute::def(StatusWordT(ScanTileStatus::SCAN_TILE_INVALID)));
    };
};

template<typename T>
struct TilePrefixTempStorage {
    T exclusive_prefix;
    T inclusive_prefix;
    T block_aggregate;
};

template<typename T>
struct no_delay_constructor {
    no_delay_constructor(compute::UInt) noexcept {};

    struct delay_t {
        void operator()() const noexcept {};
    };

    [[nodiscard]] delay_t operator()() const noexcept { return delay_t{}; };
};

// Decoupled look-back(warp)
// only device
template<typename T, typename ScanOpT, typename ScanTileStateT = ScanTileStateViewer<T>, typename DelayConstructorT = no_delay_constructor<T>>
class TilePrefixCallbackOp {
public:
    using WarpReduceT = WarpReduce<T, 32>;

    using StatusWordT = compute::uint;

    using TempStorageT = TilePrefixTempStorage<T>;

    // TempStorageT&                        temp_storage;
    SmemTypePtr<TempStorageT> temp_storage;
    // compute::BufferVar<ScanTileStateT>& tile_status;
    ScanTileStateT &tile_status;
    ScanOpT scan_op;
    compute::UInt tile_index;
    Var<T> exclusive_prefix;
    Var<T> inclusive_prefix;

    TilePrefixCallbackOp(ScanTileStateT &tile_state,
                         SmemTypePtr<TempStorageT> temp_storage,
                         ScanOpT scan_op,
                         compute::UInt tile_index)
        : tile_status{tile_state}, temp_storage{temp_storage}, scan_op{scan_op}, tile_index{tile_index} {};

    TilePrefixCallbackOp(ScanTileStateT &tile_state, SmemTypePtr<TempStorageT> temp_storage, ScanOpT scan_op)
        : TilePrefixCallbackOp(tile_state, temp_storage, scan_op, compute::block_x()) {};

public:
    Var<T> operator()(const Var<T> &block_aggregate) {
        $if (compute::thread_x() == 0) {
            (*temp_storage)[0].block_aggregate = block_aggregate;
            // ScanTileStateViewer::SetPartial(tile_status, tile_index, block_aggregate);
            tile_status.SetPartial(tile_index, block_aggregate);
        };

        compute::Int predecessor_idx = tile_index - compute::thread_x() - 1;
        Var<StatusWordT> predecessor_status;
        Var<T> windows_aggregate;

        // decay
        DelayConstructorT construct_delay(tile_index);
        process_windows(predecessor_idx, predecessor_status, windows_aggregate, construct_delay());

        // The exclusive tile prefix starts out as the current window aggregate
        exclusive_prefix = windows_aggregate;

        // warp(32) polling for predecessor tiles
        $while (compute::warp_active_all(predecessor_status != StatusWordT(ScanTileStatus::SCAN_TILE_INCLUSIVE))) {
            predecessor_idx -= compute::Int(32);
            process_windows(predecessor_idx, predecessor_status, windows_aggregate, construct_delay());

            exclusive_prefix = scan_op(windows_aggregate, exclusive_prefix);
        };

        $if (compute::thread_x() == 0) {
            inclusive_prefix = scan_op(exclusive_prefix, block_aggregate);
            // ScanTileStateViewer::SetInclusive(tile_status, tile_index, inclusive_prefix);
            tile_status.SetInclusive(tile_index, inclusive_prefix);
            (*temp_storage)[0].exclusive_prefix = exclusive_prefix;
            (*temp_storage)[0].inclusive_prefix = inclusive_prefix;
            // device_log("Tile {}: exclusive = {} inclusive = {} ", tile_index, exclusive_prefix, inclusive_prefix);
        };

        return exclusive_prefix;
    }

    inline compute::UInt GetTileIndex() const noexcept { return tile_index; }

    inline compute::Var<T> GetInclusivePrefix() const noexcept {
        return (*temp_storage)[0].inclusive_prefix;
    };

    inline compute::Var<T> GetExclusivePrefix() const noexcept {
        return (*temp_storage)[0].exclusive_prefix;
    };

    inline compute::Var<T> GetBlockAggregate() const noexcept {
        return (*temp_storage)[0].block_aggregate;
    };

private:
    template<typename DeLayT>
    void process_windows(compute::Int predecessor_idx, Var<StatusWordT> &predecessor_status, Var<T> &windows_aggregate, DeLayT delay) {
        Var<T> value;
        // ScanTileStateViewer::WaitForValid(tile_status, predecessor_idx, predecessor_status, value, delay);
        tile_status.WaitForValid(predecessor_idx, predecessor_status, value, delay);

        compute::UInt tail_flag = (predecessor_status == StatusWordT(ScanTileStatus::SCAN_TILE_INCLUSIVE));
        windows_aggregate =
            WarpReduceT().TailSegmentedReduce(value, tail_flag, SwizzleScanOp<ScanOpT>(scan_op));
    }
};

}// namespace luisa::parallel_primitive
#define LUISA_T_TEMPLATE() template<typename U>
#define LUISA_TILEPREFIXTEMPSTORAGE_NAME() luisa::parallel_primitive::TilePrefixTempStorage<U>
LUISA_TEMPLATE_STRUCT(LUISA_T_TEMPLATE, LUISA_TILEPREFIXTEMPSTORAGE_NAME, exclusive_prefix, inclusive_prefix, block_aggregate){};
using namespace luisa;
using namespace luisa::compute;
using namespace luisa::parallel_primitive;
using namespace boost::ut;
using namespace boost::ut::literals;

void test_decoupled_look_back(Device &device) {
    log_level_verbose();

    CommandList cmdlist;
    Stream stream = device.create_stream();

    constexpr size_t WARP_SIZE = 32;
    constexpr uint BLOCK_SIZE = 256;

    // The look-back spin-waits on the status of predecessor tiles, and there is
    // one thread block per tile. A block can only be unblocked by a block that
    // is actually running, and no backend guarantees that blocks which have not
    // been scheduled yet will ever be scheduled, so a single grid covering all
    // tiles deadlocks as soon as it exceeds what the device can hold resident
    // (on an Apple M4 that happens between 512 and 1024 blocks of 256 threads).
    // Tiles are therefore processed in resident-sized batches: inside a batch
    // the look-back spins as intended, and predecessors from earlier batches
    // are already committed because the batches are submitted in order.
    constexpr uint BATCH_BLOCK_COLS = 32u;
    constexpr uint BATCH_BLOCK_ROWS = 8u;
    constexpr uint TILES_PER_BATCH = BATCH_BLOCK_COLS * BATCH_BLOCK_ROWS;
    constexpr uint NUM_BATCHES = 400u;
    constexpr size_t NUM_TILES = static_cast<size_t>(TILES_PER_BATCH) * NUM_BATCHES;

    // the look-back walks 32 predecessors at a time and relies on the padding
    // entries in front of tile 0 to terminate, so the padding has to be
    // initialized as well
    static_assert(NUM_TILES >= ScanTileStateViewer<int>::TILE_STATUS_PADDING);

    auto scan_tile_status_buffer = device.create_buffer<uint>(WARP_SIZE + NUM_TILES);
    auto scan_tile_value_partial_buffer = device.create_buffer<int>(WARP_SIZE + NUM_TILES);
    auto scan_tile_value_inclusive_buffer = device.create_buffer<int>(WARP_SIZE + NUM_TILES);

    // one thread per tile; the first 32 threads also fill the padding entries
    constexpr uint INIT_DISPATCH_X = TILES_PER_BATCH;
    constexpr uint INIT_DISPATCH_Y = NUM_BATCHES;
    Kernel2D init_kernel = [&](BufferVar<uint> tile_status_buffer, BufferVar<int> tile_value_partial_buffer, BufferVar<int> tile_value_inclusive_buffer) noexcept {
        InitializeWardStatus2D(NUM_TILES, tile_status_buffer);
    };
    auto init_shader = device.compile(init_kernel);
    cmdlist << init_shader(scan_tile_status_buffer.view(), scan_tile_value_partial_buffer.view(), scan_tile_value_inclusive_buffer.view()).dispatch(INIT_DISPATCH_X, INIT_DISPATCH_Y);
    stream << cmdlist.commit() << synchronize();

    auto scan_op = [](const Var<int> &a, const Var<int> &b) noexcept { return a + b; };

    // one batch: BATCH_BLOCK_COLS blocks along x, BATCH_BLOCK_ROWS along y
    constexpr uint DECOUPLED_DISPATCH_X = BATCH_BLOCK_COLS * BLOCK_SIZE;
    constexpr uint DECOUPLED_DISPATCH_Y = BATCH_BLOCK_ROWS;
    Kernel2D decoupled_look_back_kernel = [&](BufferVar<uint> tile_status,
                                              BufferVar<int> tile_partial,
                                              BufferVar<int> tile_inclusive,
                                              BufferVar<int> exclusive_output,
                                              BufferVar<int> inclusive_output,
                                              UInt tile_offset) noexcept {
        luisa::compute::set_block_size(BLOCK_SIZE, 1u);
        luisa::compute::set_warp_size(WARP_SIZE);
        compute::UInt tid = compute::thread_x();
        // linearized in x-major order, matching the order blocks are launched
        // so that a tile's predecessors are always launched before it
        compute::UInt tile_idx =
            tile_offset + compute::block_id().y * BATCH_BLOCK_COLS + compute::block_id().x;

        ScanTileStateViewer<int> viewer{tile_status, tile_partial, tile_inclusive};

        using tile_prefix_op =
            TilePrefixCallbackOp<int, decltype(scan_op), ScanTileStateViewer<int>, no_delay_constructor<int>>;

        auto temp_storage = luisa::compute::Shared<TilePrefixTempStorage<int>>(1);

        tile_prefix_op prefix(viewer, &temp_storage, scan_op, tile_idx);
        compute::Int block_aggregate = 1;
        $if (tile_idx == 0) {
            $if (tid == 0) {
                viewer.SetInclusive(tile_idx, block_aggregate);
                exclusive_output.write(tile_idx, 0);
                inclusive_output.write(tile_idx, block_aggregate);
            };
        }
        $else {
            const auto warp_id = tid / luisa::compute::UInt(WARP_SIZE);

            $if (warp_id == 0) {
                Var<int> exclusive_prefix = prefix(block_aggregate);
                $if (tid == 0) {
                    Var<int> inclusive_prefix = scan_op(exclusive_prefix, block_aggregate);
                    exclusive_output.write(tile_idx, exclusive_prefix);
                    inclusive_output.write(tile_idx, inclusive_prefix);
                };
            };
        };
    };
    auto exclusive_output = device.create_buffer<int>(NUM_TILES);
    auto inclusive_output = device.create_buffer<int>(NUM_TILES);
    auto decoupled_look_back_shader = device.compile(decoupled_look_back_kernel);
    // one submission per batch, so that the batches are strictly ordered and a
    // batch only ever spins on predecessors that are resident or already done
    for (auto batch = 0u; batch < NUM_BATCHES; batch++) {
        CommandList batch_cmdlist;
        batch_cmdlist << decoupled_look_back_shader(scan_tile_status_buffer.view(),
                                                    scan_tile_value_partial_buffer.view(),
                                                    scan_tile_value_inclusive_buffer.view(),
                                                    exclusive_output.view(),
                                                    inclusive_output.view(),
                                                    batch * TILES_PER_BATCH)
                             .dispatch(DECOUPLED_DISPATCH_X, DECOUPLED_DISPATCH_Y);
        stream << batch_cmdlist.commit();
    }
    stream << synchronize();

    luisa::vector<int> exclusive_result(NUM_TILES);
    luisa::vector<int> inclusive_result(NUM_TILES);
    stream << exclusive_output.copy_to(luisa::span{exclusive_result})
           << inclusive_output.copy_to(luisa::span{inclusive_result}) << synchronize();

    luisa::vector<int> data(NUM_TILES, 1);
    luisa::vector<int> exclusive_expected(NUM_TILES);
    luisa::vector<int> inclusive_expected(NUM_TILES);
    std::exclusive_scan(data.begin(), data.end(), exclusive_expected.begin(), 0);
    std::inclusive_scan(data.begin(), data.end(), inclusive_expected.begin());

    bool exclusive_ok = std::equal(exclusive_result.begin(), exclusive_result.end(),
                                   exclusive_expected.begin());
    bool inclusive_ok = std::equal(inclusive_result.begin(), inclusive_result.end(),
                                   inclusive_expected.begin());

    if (!exclusive_ok || !inclusive_ok) {
        for (size_t i = 0; i < NUM_TILES; i++) {
            if (exclusive_result[i] != exclusive_expected[i] ||
                inclusive_result[i] != inclusive_expected[i]) {
                LUISA_INFO("Tile {}: exclusive = {}, inclusive = {} (expected: {}, {})",
                           i, exclusive_result[i], inclusive_result[i],
                           exclusive_expected[i], inclusive_expected[i]);
            }
        }
    }
    expect(exclusive_ok) << "decoupled_look_back_exclusive_scan";
    expect(inclusive_ok) << "decoupled_look_back_inclusive_scan";
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    test_decoupled_look_back(device);
}
