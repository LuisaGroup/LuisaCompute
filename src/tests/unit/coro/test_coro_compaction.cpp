#include "ut/ut.hpp"
#include <luisa/core/logging.h>
#include <luisa/coro/schedulers/compaction.h>

using namespace luisa;
using namespace luisa::compute::coro;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

/// Helper: build a buffer filled with per-instance marker values.
/// For instance i, the frame data is filled with value (i + 1) so
/// we can verify data integrity after compaction.
auto make_marker_buffer(size_t capacity, size_t stride) -> luisa::vector<uint32_t> {
    luisa::vector<uint32_t> buf(capacity * stride);
    for (size_t i = 0u; i < capacity; ++i) {
        auto base = i * stride;
        for (size_t j = 0u; j < stride; ++j) {
            buf[base + j] = static_cast<uint32_t>(i + 1u);
        }
    }
    return buf;
}

/// Verify that alive instance data is correctly preserved at the front.
void verify_compacted_data(const luisa::vector<uint32_t> &buf,
                           const luisa::vector<bool> &alive,
                           size_t stride) {
    size_t alive_idx = 0u;
    for (size_t i = 0u; i < alive.size(); ++i) {
        if (alive[i]) {
            auto base = alive_idx * stride;
            for (size_t j = 0u; j < stride; ++j) {
                expect(buf[base + j] == static_cast<uint32_t>(i + 1u))
                    << "instance " << i << " field " << j;
            }
            ++alive_idx;
        }
    }
}

} // namespace

void reg_coro_compaction() {

    "compaction_4inst_2alive_no_compact_at_threshold_0_5"_test = [] {
        // 4 instances, 2 alive -> load factor 0.5, threshold 0.5 -> no compaction
        constexpr size_t capacity = 4u;
        constexpr size_t stride = 3u; // token(1) + skip(1) + user(1)
        auto buf = make_marker_buffer(capacity, stride);
        luisa::vector<bool> alive = {true, false, true, false};

        CompactionResult result;
        compact_frame_buffer(buf, alive, stride, result, 0.5);

        expect(!result.compacted);
        expect(result.alive_count_before == 2u);
        expect(result.alive_count_after == 2u);
        expect(result.capacity == capacity);
        expect(result.load_factor_before == 0.5_d);
        expect(result.load_factor_after == 0.5_d);

        // Data should be untouched (no compaction occurred)
        for (size_t i = 0u; i < capacity; ++i) {
            auto base = i * stride;
            for (size_t j = 0u; j < stride; ++j) {
                expect(buf[base + j] == static_cast<uint32_t>(i + 1u));
            }
        }
    };

    "compaction_4inst_1alive_compact_to_front"_test = [] {
        // 4 instances, 1 alive -> load factor 0.25 < 0.5 -> compaction
        constexpr size_t capacity = 4u;
        constexpr size_t stride = 2u;
        auto buf = make_marker_buffer(capacity, stride);
        // Instance 2 is alive (values = 3 in buffer)
        luisa::vector<bool> alive = {false, false, true, false};

        CompactionResult result;
        compact_frame_buffer(buf, alive, stride, result);

        expect(result.compacted);
        expect(result.alive_count_before == 1u);
        expect(result.alive_count_after == 1u);
        expect(result.capacity == capacity);
        expect(result.load_factor_before == 0.25_d);
        expect(result.load_factor_after == 0.25_d);

        // Verify instance 2's data is now at position 0
        verify_compacted_data(buf, alive, stride);
    };

    "compaction_4inst_1alive_first_instance_preserved_identity"_test = [] {
        // Instance 0 is alive -> already at front, identity compaction
        constexpr size_t capacity = 4u;
        constexpr size_t stride = 4u;
        auto buf = make_marker_buffer(capacity, stride);
        luisa::vector<bool> alive = {true, false, false, false};

        CompactionResult result;
        compact_frame_buffer(buf, alive, stride, result);

        expect(result.compacted);
        expect(result.alive_count_before == 1u);

        // Instance 0 data should still be at position 0
        verify_compacted_data(buf, alive, stride);
    };

    "compaction_4inst_3alive_multi_compact"_test = [] {
        // 4 instances, 3 alive (instances 0,1,3) -> load 0.75, no compaction
        // BUT with threshold 0.9, it should compact
        constexpr size_t capacity = 4u;
        constexpr size_t stride = 3u;
        auto buf = make_marker_buffer(capacity, stride);
        luisa::vector<bool> alive = {true, true, false, true};

        CompactionResult result;
        compact_frame_buffer(buf, alive, stride, result, 0.9);

        expect(result.compacted);
        expect(result.alive_count_before == 3u);
        expect(result.alive_count_after == 3u);
        expect(result.load_factor_before == 0.75_d);

        // Alive instances 0,1,3 should now be at positions 0,1,2
        verify_compacted_data(buf, alive, stride);
    };

    "compaction_all_dead_empty"_test = [] {
        // All dead -> all empty at front (0 alive elements)
        constexpr size_t capacity = 8u;
        constexpr size_t stride = 2u;
        auto buf = make_marker_buffer(capacity, stride);
        luisa::vector<bool> alive(capacity, false);

        CompactionResult result;
        compact_frame_buffer(buf, alive, stride, result);

        expect(result.compacted);
        expect(result.alive_count_before == 0u);
        expect(result.alive_count_after == 0u);
        expect(result.load_factor_before == 0.0_d);
        expect(result.load_factor_after == 0.0_d);

        // No alive instances — verify no invariants broken
        expect(result.capacity == capacity);
    };

    "compaction_all_alive_no_compact_needed"_test = [] {
        // All alive -> load factor 1.0, no compaction
        constexpr size_t capacity = 5u;
        constexpr size_t stride = 2u;
        auto buf = make_marker_buffer(capacity, stride);
        luisa::vector<bool> alive(capacity, true);

        CompactionResult result;
        compact_frame_buffer(buf, alive, stride, result);

        expect(!result.compacted);
        expect(result.alive_count_before == capacity);
        expect(result.alive_count_after == capacity);
        expect(result.load_factor_before == 1.0_d);

        // Data untouched
        for (size_t i = 0u; i < capacity; ++i) {
            auto base = i * stride;
            expect(buf[base] == static_cast<uint32_t>(i + 1u));
        }
    };

    "compaction_data_integrity_large"_test = [] {
        // 16 instances, 7 alive (even indices except last) -> load ~0.4375 < 0.5
        // stride=5 to test multi-word copy
        constexpr size_t capacity = 16u;
        constexpr size_t stride = 5u;
        auto buf = make_marker_buffer(capacity, stride);
        luisa::vector<bool> alive(capacity);
        for (size_t i = 0u; i < capacity; ++i) {
            alive[i] = (i % 2u == 0u); // even indices alive
        }
        alive[14u] = false; // drop one alive so load < 0.5

        CompactionResult result;
        compact_frame_buffer(buf, alive, stride, result);

        expect(result.compacted);
        expect(result.alive_count_before == 7u);
        expect(result.alive_count_after == 7u);
        expect(result.load_factor_before < 0.5_d);

        // Verify data integrity: each alive instance's marker values preserved
        verify_compacted_data(buf, alive, stride);

        // Also verify the first 7 positions contain the alive data
        // (indices 0,2,4,6,8,10,12 -> values 1,3,5,7,9,11,13)
        for (size_t pos = 0u; pos < 7u; ++pos) {
            auto base = pos * stride;
            auto expected_val = static_cast<uint32_t>(pos * 2u + 1u);
            expect(buf[base] == expected_val)
                << "position " << pos << " should have value " << expected_val;
        }
    };

    "compaction_empty_buffer"_test = [] {
        // Zero capacity — no crash, nothing to do
        luisa::vector<uint32_t> buf;
        luisa::vector<bool> alive;

        CompactionResult result;
        compact_frame_buffer(buf, alive, 0u, result);

        expect(!result.compacted);
        expect(result.alive_count_before == 0u);
        expect(result.capacity == 0u);
        expect(result.load_factor_before == 0.0_d);
    };

    "compaction_zero_stride"_test = [] {
        // frame_stride=0 — no data to move, but still compacts logically
        constexpr size_t capacity = 4u;
        luisa::vector<uint32_t> buf;
        luisa::vector<bool> alive = {false, true, false, true};

        CompactionResult result;
        compact_frame_buffer(buf, alive, 0u, result, 0.3);

        // stride=0 -> nothing to move, just statistics
        expect(result.alive_count_before == 2u);
        expect(!result.compacted); // stride 0 bails out early
        expect(result.capacity == capacity);
    };

    "compaction_custom_thresholds"_test = [] {
        constexpr size_t capacity = 100u;
        constexpr size_t stride = 2u;
        auto buf = make_marker_buffer(capacity, stride);
        luisa::vector<bool> alive(capacity, false);
        // 30 alive out of 100 -> load_factor = 0.3
        for (size_t i = 0u; i < 30u; ++i) {
            alive[i * 3u + 1u] = true; // scatter them
        }

        // threshold 0.35 -> no compaction (0.3 < 0.35 triggers it? No, check: 0.3 >= 0.35 is false, so compaction triggers)
        // Wait, the condition is: load_factor_before >= threshold -> skip.
        // threshold 0.2 -> no compaction (0.3 >= 0.2 -> skip)
        {
            auto buf_copy = buf;
            CompactionResult r1;
            compact_frame_buffer(buf_copy, alive, stride, r1, 0.2);
            expect(!r1.compacted);
        }

        // threshold 0.35 -> compaction (0.3 < 0.35 -> compact)
        {
            auto buf_copy = buf;
            CompactionResult r2;
            compact_frame_buffer(buf_copy, alive, stride, r2, 0.35);
            expect(r2.compacted);
            verify_compacted_data(buf_copy, alive, stride);
        }

        // threshold 0.31 -> compaction (0.3 < 0.31 -> compact)
        {
            auto buf_copy = buf;
            CompactionResult r3;
            compact_frame_buffer(buf_copy, alive, stride, r3, 0.31);
            expect(r3.compacted);
        }
    };

    "compaction_preserves_total_data_of_alive_instances"_test = [] {
        // Explicitly verify every uint32_t of each alive instance is preserved
        constexpr size_t capacity = 6u;
        constexpr size_t stride = 4u;
        auto buf = make_marker_buffer(capacity, stride);

        // Overwrite specific values to make instances distinguishable
        uint32_t counter = 100u;
        for (size_t i = 0u; i < capacity; ++i) {
            auto base = i * stride;
            for (size_t j = 0u; j < stride; ++j) {
                buf[base + j] = counter++;
            }
        }

        luisa::vector<bool> alive = {false, true, false, true, false, true};

        // Save expected data for alive instances
        luisa::vector<uint32_t> expected_alive_data;
        for (size_t i = 0u; i < capacity; ++i) {
            if (alive[i]) {
                auto base = i * stride;
                for (size_t j = 0u; j < stride; ++j) {
                    expected_alive_data.push_back(buf[base + j]);
                }
            }
        }

        CompactionResult result;
        // 3 alive / 6 capacity = 0.5, need threshold > 0.5 to trigger
        compact_frame_buffer(buf, alive, stride, result, 0.55);

        expect(result.compacted);
        expect(result.alive_count_before == 3u);

        // Verify every element matches
        for (size_t pos = 0u; pos < 3u; ++pos) {
            auto base = pos * stride;
            for (size_t j = 0u; j < stride; ++j) {
                auto expected = expected_alive_data[pos * stride + j];
                expect(buf[base + j] == expected)
                    << "mismatch at pos=" << pos << " field=" << j
                    << " expected=" << expected << " got=" << buf[base + j];
            }
        }
    };
}

int main(int /*argc*/, char * /*argv*/[]) {
    reg_coro_compaction();
    return 0;
}
