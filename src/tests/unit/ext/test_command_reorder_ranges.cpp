// Test for the range tracking of the backend command reordering pass
// (src/backends/common/command_reorder_visitor.h, RangeHandle).
//
// This test covers:
// - disjoint ascending ranges beyond the old 16-range give-up limit staying in
//   one layer (no false serialization, no precision collapse)
// - write-after-write / read-after-write / write-after-read hazards on shared
//   ranges serializing into separate layers
// - read-only sharing of one range in a single layer
// - overlapping ranges merging transitively while disjoint ranges keep their
//   own (lower) layers
// - round-robin (rotating) writes keeping per-range chains independent
// - cold accesses to a previously recorded range keeping that range's fine
//   layer instead of inheriting the handle-global maximum
// - the runtime enable/disable switch (set_enabled) degenerating to strict
//   submission order
#include "ut/ut.hpp"
#include "command_reorder_visitor.h"
#include <array>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>
using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

struct FakeReorderState {
    struct Resource {
        uint64_t handle;
        bool is_buffer;
    };
    std::unordered_map<uint64_t, Usage> shader_usages;
    std::unordered_map<uint64_t, std::vector<Resource>> bindless_resources;
};

struct FakeReorderFuncTable {
    std::shared_ptr<FakeReorderState> state;
    [[nodiscard]] uint64_t canonical_buffer_handle(uint64_t handle) const noexcept {
        return handle;
    }
    [[nodiscard]] uint64_t canonical_texture_handle(uint64_t handle) const noexcept {
        return handle;
    }
    void traverse_bindless_resources(uint64_t bindless_handle, ReorderBindlessResourceVisitor visitor) const noexcept {
        if (auto iter = state->bindless_resources.find(bindless_handle);
            iter != state->bindless_resources.end()) {
            for (auto resource : iter->second) {
                visitor(resource.handle, resource.is_buffer);
            }
        }
    }
    [[nodiscard]] Usage get_usage(uint64_t shader_handle, size_t) const noexcept {
        return state->shader_usages.at(shader_handle);
    }
    void update_bindless(uint64_t, luisa::span<const BindlessArrayUpdateCommand::Modification>) const noexcept {}
    void update_bindless(uint64_t, luisa::span<const BindlessArrayUpdateCommand::BufferModification>) const noexcept {}
    void update_bindless(uint64_t, luisa::span<const BindlessArrayUpdateCommand::Texture2DModification>) const noexcept {}
    void update_bindless(uint64_t, luisa::span<const BindlessArrayUpdateCommand::Texture3DModification>) const noexcept {}
    [[nodiscard]] luisa::span<const Argument> shader_bindings(uint64_t) const noexcept {
        return {};
    }
    [[nodiscard]] luisa::span<const Argument> raster_shader_bindings(uint64_t) const noexcept {
        return {};
    }
};
static_assert(ReorderFuncTable<FakeReorderFuncTable>);

using Reorder = CommandReorderVisitor<FakeReorderFuncTable, true>;
using Range = Reorder::Range;

// BufferUploadCommand is a write of [offset, offset + size),
// BufferDownloadCommand is a read of the same range. supportConcurrentCopy is
// true, so both are tracked with range precision.
struct AccessList {
    static constexpr uint64_t buffer = 7u;
    static constexpr size_t range_size = 16u;
    std::array<std::byte, range_size> data{};
    luisa::vector<std::unique_ptr<Command>> storage;
    luisa::vector<Command const *> order;
    void write(Reorder &reorder, size_t offset) {
        auto cmd = std::make_unique<BufferUploadCommand>(buffer, offset, range_size, data.data());
        order.emplace_back(cmd.get());
        reorder.visit(cmd.get());
        storage.emplace_back(std::move(cmd));
    }
    void read(Reorder &reorder, size_t offset) {
        auto cmd = std::make_unique<BufferDownloadCommand>(buffer, offset, range_size, data.data());
        order.emplace_back(cmd.get());
        reorder.visit(cmd.get());
        storage.emplace_back(std::move(cmd));
    }
};

[[nodiscard]] size_t disjoint_write_layers(size_t count, size_t stride = AccessList::range_size) {
    Reorder reorder{FakeReorderFuncTable{std::make_shared<FakeReorderState>()}};
    AccessList accesses;
    for (size_t i = 0u; i < count; i++) {
        accesses.write(reorder, i * stride);
    }
    return reorder.command_lists().size();
}

[[nodiscard]] size_t rotating_write_layers(size_t range_count, size_t rounds) {
    Reorder reorder{FakeReorderFuncTable{std::make_shared<FakeReorderState>()}};
    AccessList accesses;
    for (size_t round = 0u; round < rounds; round++) {
        for (size_t r = 0u; r < range_count; r++) {
            accesses.write(reorder, r * AccessList::range_size);
        }
    }
    return reorder.command_lists().size();
}

void test_disjoint_ascending_ranges_stay_one_layer() {
    // Way past the old 16-range give-up limit: every write is disjoint, so the
    // whole stream must still share one layer.
    expect(eq(disjoint_write_layers(64u), 1u));
    expect(eq(disjoint_write_layers(1024u), 1u));
}

void test_same_range_writes_serialize() {
    Reorder reorder{FakeReorderFuncTable{std::make_shared<FakeReorderState>()}};
    AccessList accesses;
    for (auto i = 0u; i < 8u; i++) {
        accesses.write(reorder, 0u);
    }
    expect(eq(reorder.command_lists().size(), 8u));
}

void test_hazards_on_shared_range() {
    {
        // write then read (RAW)
        Reorder reorder{FakeReorderFuncTable{std::make_shared<FakeReorderState>()}};
        AccessList accesses;
        accesses.write(reorder, 0u);
        accesses.read(reorder, 0u);
        expect(eq(reorder.command_lists().size(), 2u));
    }
    {
        // read then write (WAR)
        Reorder reorder{FakeReorderFuncTable{std::make_shared<FakeReorderState>()}};
        AccessList accesses;
        accesses.read(reorder, 0u);
        accesses.write(reorder, 0u);
        expect(eq(reorder.command_lists().size(), 2u));
    }
    {
        // read after read shares the layer
        Reorder reorder{FakeReorderFuncTable{std::make_shared<FakeReorderState>()}};
        AccessList accesses;
        for (auto i = 0u; i < 8u; i++) {
            accesses.read(reorder, 0u);
        }
        expect(eq(reorder.command_lists().size(), 1u));
    }
}

void test_overlapping_ranges_merge_transitively() {
    Reorder reorder{FakeReorderFuncTable{std::make_shared<FakeReorderState>()}};
    AccessList accesses;
    accesses.write(reorder, 0u); // [0, 16)  -> layer 0
    accesses.write(reorder, 8u); // [8, 24)  -> collides, layer 1, merges into [0, 24)
    accesses.write(reorder, 16u);// [16, 32) -> collides the merged view, layer 2
    accesses.write(reorder, 32u);// [32, 48) -> disjoint from [0, 32), back to layer 0
    expect(eq(reorder.command_lists().size(), 3u));
}

void test_rotating_writes_keep_per_range_chains() {
    // Four rounds over sixteen ranges: each range is written four times, so the
    // ideal is exactly four layers. A range tracker that collapses per-range
    // information serializes far more.
    expect(eq(rotating_write_layers(16u, 4u), 4u));
    expect(eq(rotating_write_layers(16u, 8u), 8u));
}

void test_cold_range_keeps_its_own_layer() {
    // One hot range with a long write-after-write chain pushes the
    // handle-global maximum to layer 4. Fifteen cold disjoint writes land in
    // layer 0. Reading a cold range must see only that range's write (layer 0)
    // and land in layer 1, not after the hot chain.
    Reorder reorder{FakeReorderFuncTable{std::make_shared<FakeReorderState>()}};
    AccessList accesses;
    constexpr size_t hot = 0u;
    for (auto i = 0u; i < 5u; i++) {
        accesses.write(reorder, hot);
    }
    for (size_t i = 1u; i <= 15u; i++) {
        accesses.write(reorder, i * AccessList::range_size);
    }
    accesses.read(reorder, AccessList::range_size);
    // Hot chain contributes layers 0..4; the cold read sits in layer 1, so the
    // batch needs five layers in total.
    expect(eq(reorder.command_lists().size(), 5u));
}

void test_disjoint_after_merge_keeps_precision() {
    // After ranges merge, a range that only touches the tail of the merged
    // view must still land right after it instead of after unrelated work.
    Reorder reorder{FakeReorderFuncTable{std::make_shared<FakeReorderState>()}};
    AccessList accesses;
    accesses.write(reorder, 0u); // [0, 16) -> layer 0
    accesses.write(reorder, 8u); // [8, 24) -> layer 1, merged [0, 24)
    accesses.read(reorder, 16u); // [16, 32) -> collides [0, 24) -> layer 2
    expect(eq(reorder.command_lists().size(), 3u));
}

void test_dispatch_commands_use_same_tracking() {
    // The ShaderDispatchCommand path must agree with the copy-command path.
    auto state = std::make_shared<FakeReorderState>();
    constexpr uint64_t shader = 99u;
    state->shader_usages.emplace(shader, Usage::WRITE);
    Reorder reorder{FakeReorderFuncTable{state}};
    constexpr size_t range_size = 16u;
    auto make_dispatch = [&](size_t offset) {
        Argument argument{
            .tag = Argument::Tag::BUFFER,
            .buffer = {AccessList::buffer, offset, range_size}};
        luisa::vector<std::byte> argument_buffer(sizeof(argument));
        std::memcpy(argument_buffer.data(), &argument, sizeof(argument));
        return ShaderDispatchCommand{
            shader, std::move(argument_buffer), 1u, uint3{1u, 1u, 1u}};
    };
    luisa::vector<std::unique_ptr<Command>> storage;
    for (size_t i = 0u; i < 64u; i++) {
        auto cmd = std::make_unique<ShaderDispatchCommand>(make_dispatch(i * range_size));
        reorder.visit(cmd.get());
        storage.emplace_back(std::move(cmd));
    }
    expect(eq(reorder.command_lists().size(), 1u));
    // Rotating over the same ranges must serialize per range chain.
    Reorder rotating{FakeReorderFuncTable{state}};
    for (size_t round = 0u; round < 4u; round++) {
        for (size_t r = 0u; r < 16u; r++) {
            auto cmd = std::make_unique<ShaderDispatchCommand>(make_dispatch(r * range_size));
            rotating.visit(cmd.get());
            storage.emplace_back(std::move(cmd));
        }
    }
    expect(eq(rotating.command_lists().size(), 4u));
}

void test_disabled_switch_keeps_submission_order() {
    Reorder reorder{FakeReorderFuncTable{std::make_shared<FakeReorderState>()}};
    reorder.set_enabled(false);
    AccessList accesses;
    for (size_t i = 0u; i < 64u; i++) {
        accesses.write(reorder, i * AccessList::range_size);
    }
    expect(eq(reorder.command_lists().size(), 64u));
    // ... and re-enabling merges again.
    reorder.clear();
    reorder.set_enabled(true);
    AccessList more;
    for (size_t i = 0u; i < 64u; i++) {
        more.write(reorder, i * AccessList::range_size);
    }
    expect(eq(reorder.command_lists().size(), 1u));
}

// ---------------------------------------------------------------------------
// Differential verification against a brute-force reference model. The oracle
// keeps the complete per-handle access history (no merging at all) and assigns
// each command the layer the hazard rules dictate; the visitor must agree on
// every single command, not just on the layer count.
struct Oracle {
    struct Entry {
        Range range;
        int64_t read_layer;
        int64_t write_layer;
    };
    std::unordered_map<uint64_t, luisa::vector<Entry>> history;

    [[nodiscard]] static int64_t max_write(const luisa::vector<Entry> *entries, Range const &range) {
        if (entries == nullptr) {
            return -1;
        }
        int64_t layer = -1;
        for (auto &&e : *entries) {
            if (e.range.collide(range)) {
                layer = std::max(layer, e.write_layer);
            }
        }
        return layer;
    }
    [[nodiscard]] static int64_t max_read_write(const luisa::vector<Entry> *entries, Range const &range) {
        if (entries == nullptr) {
            return -1;
        }
        int64_t layer = -1;
        for (auto &&e : *entries) {
            if (e.range.collide(range)) {
                layer = std::max(layer, std::max(e.read_layer, e.write_layer));
            }
        }
        return layer;
    }
    [[nodiscard]] luisa::vector<Entry> *find(uint64_t handle) {
        if (auto iter = history.find(handle); iter != history.end()) {
            return &iter->second;
        }
        return nullptr;
    }
    int64_t write(uint64_t handle, Range const &range) {
        auto layer = max_read_write(find(handle), range) + 1;
        history[handle].emplace_back(Entry{range, layer, layer});
        return layer;
    }
    int64_t read(uint64_t handle, Range const &range) {
        auto layer = max_write(find(handle), range) + 1;
        history[handle].emplace_back(Entry{range, layer, -1});
        return layer;
    }
    int64_t copy(uint64_t src_handle, Range const &src_range,
                 uint64_t dst_handle, Range const &dst_range) {
        auto layer = std::max(max_write(find(src_handle), src_range),
                              max_read_write(find(dst_handle), dst_range)) +
            1;
        history[src_handle].emplace_back(Entry{src_range, layer, -1});
        history[dst_handle].emplace_back(Entry{dst_range, layer, layer});
        return layer;
    }
};

// Walk the visitor's layer lists and return the layer of every command.
[[nodiscard]] std::unordered_map<Command const *, int64_t> visitor_layers(const Reorder &reorder) {
    std::unordered_map<Command const *, int64_t> result;
    auto lists = reorder.command_lists();
    for (auto layer = int64_t{0}; layer < static_cast<int64_t>(lists.size()); layer++) {
        for (auto link = lists[layer]; link != nullptr; link = link->p_next) {
            result.emplace(link->cmd, layer);
        }
    }
    return result;
}

void run_oracle_comparison(uint32_t seed, size_t access_count, size_t slot_count,
                           size_t buffer_count, bool with_copies) {
    LUISA_ASSERT(buffer_count >= 1u && buffer_count <= 4u, "buffer_count out of range");
    std::mt19937 rng{seed};
    std::uniform_int_distribution<size_t> slot_dist{0u, slot_count - 1u};
    std::uniform_int_distribution<size_t> buffer_dist{0u, buffer_count - 1u};
    std::uniform_int_distribution<size_t> dst_buffer_dist{0u, buffer_count - 1u};
    std::uniform_int_distribution<int> kind_dist{0u, with_copies ? 2 : 1};
    // Occasionally emit spans wider than one slot so ranges overlap
    // non-trivially.
    std::uniform_int_distribution<size_t> span_dist{1u, 4u};

    Reorder reorder{FakeReorderFuncTable{std::make_shared<FakeReorderState>()}};
    Oracle oracle;
    std::array<std::byte, 256u> data{};
    luisa::vector<std::unique_ptr<Command>> storage;
    luisa::vector<std::pair<Command const *, int64_t>> expected;
    expected.reserve(access_count);
    luisa::vector<std::tuple<size_t, size_t, size_t, size_t, size_t, size_t>> access_log;// kind, buffer, offset, size, dst, dst_offset

    for (size_t i = 0u; i < access_count; i++) {
        auto buffer = buffer_dist(rng);
        auto offset = slot_dist(rng) * AccessList::range_size;
        auto size = span_dist(rng) * AccessList::range_size;
        if (size > data.size()) {
            size = data.size();
        }
        auto kind = kind_dist(rng);
        if (kind == 0) {
            access_log.emplace_back(kind, buffer, offset, size, buffer, offset);
            auto cmd = std::make_unique<BufferUploadCommand>(buffer, offset, size, data.data());
            expected.emplace_back(cmd.get(), oracle.write(buffer, Range::from_offset_size(offset, size)));
            reorder.visit(cmd.get());
            storage.emplace_back(std::move(cmd));
        } else if (kind == 1) {
            access_log.emplace_back(kind, buffer, offset, size, buffer, offset);
            auto cmd = std::make_unique<BufferDownloadCommand>(buffer, offset, size, data.data());
            expected.emplace_back(cmd.get(), oracle.read(buffer, Range::from_offset_size(offset, size)));
            reorder.visit(cmd.get());
            storage.emplace_back(std::move(cmd));
        } else {
            auto dst = dst_buffer_dist(rng);
            auto dst_offset = slot_dist(rng) * AccessList::range_size;
            access_log.emplace_back(kind, buffer, offset, size, dst, dst_offset);
            auto cmd = std::make_unique<BufferCopyCommand>(buffer, dst, offset, dst_offset, size);
            expected.emplace_back(cmd.get(), oracle.copy(buffer, Range::from_offset_size(offset, size),
                                                         dst, Range::from_offset_size(dst_offset, size)));
            reorder.visit(cmd.get());
            storage.emplace_back(std::move(cmd));
        }
    }

    if (std::getenv("DUMP_ORACLE_SEQ")) {
        luisa::string name = "oracle_seq_";
        name.append(std::to_string(seed)).append(".txt");
        std::ofstream out{name.c_str(), std::ios::trunc};
        auto actual_dbg = visitor_layers(reorder);
        for (size_t idx = 0u; idx < expected.size(); idx++) {
            auto &&[k, b, o, sz, db, dob] = access_log[idx];
            out << idx << ' ' << k << ' ' << b << ' ' << o << ' ' << sz << ' ' << db << ' ' << dob << ' '
                << expected[idx].second << ' ' << actual_dbg[expected[idx].first] << std::endl;
        }
    }
    auto actual = visitor_layers(reorder);
    int64_t max_layer = -1;
    for (auto &&[cmd, layer] : expected) {
        max_layer = std::max(max_layer, layer);
    }

    // Soundness is the real contract of the reorder pass: whenever two
    // *different* commands genuinely hazard on the same buffer (colliding
    // ranges and at least one of them writes), the later one must land in a
    // strictly later layer. Conservative extra layers (from interval merging)
    // are legal and expected in dense-overlap patterns; a violated hazard is
    // not. The per-command layers are checked against the complete access
    // history, so this is exhaustive rather than sampled.
    struct Span {
        size_t access_index;
        size_t buffer;
        uint64_t min;
        uint64_t max;
        bool is_write;
    };
    luisa::vector<Span> spans;
    spans.reserve(expected.size() * 2u);
    for (size_t idx = 0u; idx < access_log.size(); idx++) {
        auto &&[k, b, o, sz, db, dob] = access_log[idx];
        if (k == 2u) {
            spans.emplace_back(Span{idx, b, o, o + sz, false});
            spans.emplace_back(Span{idx, db, dob, dob + sz, true});
        } else {
            spans.emplace_back(Span{idx, b, o, o + sz, k == 0u});
        }
    }
    size_t hazard_pairs = 0u;
    auto sound = true;
    for (size_t a = 0u; a < spans.size(); a++) {
        auto &&sa = spans[a];
        auto layer_a = actual[expected[sa.access_index].first];
        for (size_t b = a + 1u; b < spans.size(); b++) {
            auto &&sb = spans[b];
            if (sa.access_index == sb.access_index ||
                sa.buffer != sb.buffer ||
                !(sa.is_write || sb.is_write) ||
                !(sa.min < sb.max && sb.min < sa.max)) {
                continue;
            }
            hazard_pairs++;
            auto layer_b = actual[expected[sb.access_index].first];
            if (layer_b <= layer_a) {
                LUISA_WARNING("seed {}: hazard violated between access #{} and #{} "
                              "(buffer {}, layers {} -> {})",
                              seed, sa.access_index, sb.access_index,
                              sa.buffer, layer_a, layer_b);
                sound = false;
            }
        }
    }
    expect(sound) << "every hazard pair must be ordered by strictly increasing layers";
    LUISA_INFO("seed {}: {} accesses, {} hazard pairs verified, visitor layers {} "
               "(minimal-oracle {})",
               seed, expected.size(), hazard_pairs,
               reorder.command_lists().size(),
               static_cast<size_t>(max_layer + 1));
}

void test_oracle_random_sequences() {
    // Single buffer: dense overlap pressure on one RangeHandle.
    run_oracle_comparison(42u, 2000u, 64u, 1u, false);
    // Single buffer including buffer-to-buffer copies (self-copies included).
    run_oracle_comparison(1337u, 2000u, 48u, 1u, true);
    // Four buffers: cross-resource layer computation.
    run_oracle_comparison(7u, 3000u, 64u, 4u, true);
    // Few slots: heavy same-range write-after-write chains.
    run_oracle_comparison(2024u, 1500u, 4u, 2u, true);
    // Many slots: mostly disjoint traffic with occasional spans.
    run_oracle_comparison(99u, 3000u, 512u, 3u, true);
}

void test_double_clear_and_reuse() {
    Reorder reorder{FakeReorderFuncTable{std::make_shared<FakeReorderState>()}};
    AccessList accesses;
    for (size_t i = 0u; i < 32u; i++) {
        accesses.write(reorder, i * AccessList::range_size);
    }
    expect(eq(reorder.command_lists().size(), 1u));
    reorder.clear();
    reorder.clear();// must be idempotent
    AccessList more;
    for (size_t i = 0u; i < 8u; i++) {
        more.write(reorder, 0u);
    }
    expect(eq(reorder.command_lists().size(), 8u));
}

void test_visitor_destruction_with_live_state() {
    // Construct and destroy visitors that still hold live batch state: the
    // destructor must run clear() so the heap interval vectors are released
    // instead of leaked (and must not crash on the destroyed handle storage).
    for (auto i = 0u; i < 64u; i++) {
        Reorder reorder{FakeReorderFuncTable{std::make_shared<FakeReorderState>()}};
        AccessList accesses;
        for (size_t j = 0u; j < 64u; j++) {
            accesses.write(reorder, j * AccessList::range_size);
            accesses.read(reorder, j * AccessList::range_size);
        }
    }
    expect(true);
}

void test_empty_range_semantics() {
    // Zero-size ranges keep the historical collide() semantics: an empty range
    // strictly inside a recorded one counts as colliding, one outside does
    // not. This must match the previous exact-range implementation.
    Reorder reorder{FakeReorderFuncTable{std::make_shared<FakeReorderState>()}};
    std::array<std::byte, 16u> data{};
    luisa::vector<std::unique_ptr<Command>> storage;
    auto write = [&](size_t offset, size_t size) {
        auto cmd = std::make_unique<BufferUploadCommand>(AccessList::buffer, offset, size, data.data());
        reorder.visit(cmd.get());
        storage.emplace_back(std::move(cmd));
    };
    write(0u, 16u); // [0, 16) -> layer 0
    write(16u, 0u); // [16, 16) empty, outside -> layer 0
    write(8u, 0u);  // [8, 8) empty, strictly inside [0, 16) -> layer 1
    expect(eq(reorder.command_lists().size(), 2u));
}

void test_whole_range_write_merges_sub_ranges() {
    // A bindless WRITE dispatch records a whole-range write on every snapshot
    // resource. Subsequent sub-range accesses must serialize after it, and the
    // whole-range view must absorb existing sub-range views.
    auto state = std::make_shared<FakeReorderState>();
    constexpr uint64_t shader = 55u;
    constexpr uint64_t heap = 66u;
    state->shader_usages.emplace(shader, Usage::WRITE);
    state->bindless_resources.emplace(
        heap, std::vector{FakeReorderState::Resource{
                  .handle = AccessList::buffer, .is_buffer = true}});
    Reorder reorder{FakeReorderFuncTable{state}};
    AccessList accesses;
    // First two disjoint sub-range writes share layer 0.
    accesses.write(reorder, 0u);
    accesses.write(reorder, 32u);
    // Whole-range write through the bindless dispatch -> layer 1.
    Argument argument{
        .tag = Argument::Tag::BINDLESS_ARRAY,
        .bindless_array = {heap}};
    luisa::vector<std::byte> argument_buffer(sizeof(argument));
    std::memcpy(argument_buffer.data(), &argument, sizeof(argument));
    ShaderDispatchCommand dispatch{
        shader, std::move(argument_buffer), 1u, uint3{1u, 1u, 1u}};
    reorder.visit(&dispatch);
    // Disjoint sub-range writes now both collide the whole-range view.
    accesses.write(reorder, 16u);
    accesses.write(reorder, 48u);
    // With split tracking the whole-range view is clipped at [16, 32), so the
    // second post-dispatch write still only sees the layer-1 whole-range view.
    expect(eq(reorder.command_lists().size(), 3u));
}

}// namespace

static auto test_command_reorder_ranges_registration = [] {
    "disjoint ascending ranges stay one layer"_test = [] { test_disjoint_ascending_ranges_stay_one_layer(); };
    "same range writes serialize"_test = [] { test_same_range_writes_serialize(); };
    "hazards on shared range"_test = [] { test_hazards_on_shared_range(); };
    "overlapping ranges merge transitively"_test = [] { test_overlapping_ranges_merge_transitively(); };
    "rotating writes keep per-range chains"_test = [] { test_rotating_writes_keep_per_range_chains(); };
    "cold range keeps its own layer"_test = [] { test_cold_range_keeps_its_own_layer(); };
    "disjoint after merge keeps precision"_test = [] { test_disjoint_after_merge_keeps_precision(); };
    "dispatch commands use same tracking"_test = [] { test_dispatch_commands_use_same_tracking(); };
    "disabled switch keeps submission order"_test = [] { test_disabled_switch_keeps_submission_order(); };
    "oracle random sequences"_test = [] { test_oracle_random_sequences(); };
    "double clear and reuse"_test = [] { test_double_clear_and_reuse(); };
    "visitor destruction with live state"_test = [] { test_visitor_destruction_with_live_state(); };
    "empty range semantics"_test = [] { test_empty_range_semantics(); };
    "whole range write merges sub ranges"_test = [] { test_whole_range_write_merges_sub_ranges(); };
    return 0;
}();

int main() {}
