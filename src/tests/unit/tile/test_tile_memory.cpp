// Test explicit addressable Memory capture and structured MemorySSA.
// Covers stable identity, snapshots, loop state, initialization, invalid
// state rewrites, lexical ownership, and explicit-only load/store syntax.
#include "ut/ut.hpp"

#include <luisa/tile.h>
#include <optional>
#include <type_traits>
#include <utility>

using namespace luisa::compute::tile;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

static_assert(!std::is_copy_constructible_v<Memory<float>>);
static_assert(std::is_move_constructible_v<Memory<float>>);
static_assert(!std::is_copy_assignable_v<Memory<float>>);
static_assert(!std::is_move_assignable_v<Memory<float>>);
static_assert(!std::is_assignable_v<Memory<float> &, Tile<float>>);
static_assert(!std::is_convertible_v<Memory<float>, Tile<float>>);
static_assert(std::same_as<decltype(std::declval<Memory<float>>().load()), Tile<float>>);

template<typename V>
concept memory_storable = requires(Memory<float> &memory, const V &value) { memory.store(value); };
static_assert(memory_storable<Tile<float>>);
static_assert(!memory_storable<Tile<int32_t>>);
static_assert(!memory_storable<MemoryRef<float, 1>>);
static_assert(!memory_storable<Scalar<float>>);

[[nodiscard]] luisa::vector<Operation *> operations(Region &region, OperationKind kind) {
    luisa::vector<Operation *> result;
    for (auto block : region.blocks()) {
        for (auto operation : block->operations()) {
            if (operation->kind() == kind) { result.emplace_back(operation); }
            for (auto &&child : operation->regions()) {
                auto nested = operations(*child, kind);
                result.insert(result.end(), nested.begin(), nested.end());
            }
        }
    }
    return result;
}

[[nodiscard]] bool has_diagnostic(const Kernel &kernel, luisa::string_view text) {
    for (auto &&diagnostic : kernel.diagnostics()) {
        if (diagnostic.find(text) != luisa::string::npos) { return true; }
    }
    auto verified = verify(kernel.module());
    for (auto &&diagnostic : verified.diagnostics()) {
        if (diagnostic.message.find(text) != luisa::string::npos) { return true; }
    }
    return false;
}

void test_identity_and_snapshot() {
    auto kernel = tile_kernel("memory_identity", [] {
                      for (auto &nest : parallel(shape(3))) {
                          auto space = shape(7);
                          auto a = memory<float>(space, mem::private_);
                          auto b = memory<float>(space);
                          a.store(full<float>(space, 1.0f));
                          auto snapshot = a.load();
                          a.store(full<float>(space, 2.0f));
                          auto moved = std::move(b);
                          expect(!b.valid());
                          moved.store(snapshot + a.load());
                          expect(moved.valid());
                          expect(moved.space() == space);
                      }
                  }).capture();
    expect(kernel.valid());
    auto &body = kernel.function().body();
    auto allocations = operations(body, OperationKind::MEMORY_ALLOC);
    auto loads = operations(body, OperationKind::MEMORY_LOAD);
    auto stores = operations(body, OperationKind::MEMORY_STORE);
    expect(eq(allocations.size(), 2u));
    expect(eq(loads.size(), 2u));
    expect(eq(stores.size(), 3u));
    if (allocations.size() != 2u || loads.size() != 2u || stores.size() != 3u) { return; }
    expect(allocations[0]->parent_block() == allocations[1]->parent_block());
    auto &&resource = allocations[0]->resource_class_constraint();
    expect(resource.has_value());
    if (resource) { expect(*resource == "private"); }
    expect(!allocations[1]->resource_class_constraint());
    expect(loads[0]->operand(1) == stores[0]->result(0));
    expect(stores[1]->operand(1) == stores[0]->result(0));
    expect(loads[1]->operand(1) == stores[1]->result(0));
    expect(stores[2]->operand(0) == allocations[1]->result(0));
}

void test_temporal_carries() {
    for (auto iterations : {0, 1, 5}) {
        auto kernel = tile_kernel("memory_carries", [iterations] {
                          for (auto &nest : parallel(shape(3))) {
                              auto space = shape(7);
                              auto a = memory<float>(space);
                              auto b = memory<float>(space);
                              a.store(zeros<float>(space));
                              b.store(zeros<float>(space));
                              auto snapshot = a.load();
                              auto accumulator = zeros<float>(space);
                              for (auto &step : nest.pipeline(shape(iterations))) {
                                  step.stage("read");
                                  auto old_a = a.load();
                                  auto old_b = b.load();
                                  step.stage("write");
                                  a.store(old_b + 1.0f);
                                  for (auto &serial : step.serial(shape(2))) {
                                      b.store(b.load() + old_a + 2.0f);
                                  }
                                  accumulator += old_a + old_b;
                              }
                              b.store(a.load() + b.load() + snapshot + accumulator);
                          }
                      }).capture();
        expect(kernel.valid()) << iterations;
        auto loops = operations(kernel.function().body(), OperationKind::PIPELINE);
        expect(eq(loops.size(), 1u));
        if (loops.size() != 1u) { continue; }
        auto loop = loops.front();
        auto memory_states = 0u;
        auto tile_states = 0u;
        for (auto i = 0u; i < loop->result_count(); i++) {
            memory_states += loop->result(i)->type().kind() == TypeKind::MEMORY_STATE;
            tile_states += loop->result(i)->type().is_tile();
        }
        expect(eq(memory_states, 2u));
        expect(eq(tile_states, 1u));
        // A bad transform cannot swap the two same-typed MemoryState yields.
        auto yield = loop->region(0)->block(0)->operations().back();
        auto old_state = yield->operand(0);
        yield->set_operand(0, yield->operand(1));
        expect(!kernel.valid());
        yield->set_operand(0, old_state);
        expect(kernel.valid());
    }
}

void test_initialization_and_stale_states() {
    for (auto iterations : {0, 1, 3}) {
        auto kernel = tile_kernel("memory_definite_init", [iterations] {
                          for (auto &nest : parallel(shape(1))) {
                              auto a = memory<float>(shape(7));
                              for (auto &step : nest.serial(shape(iterations))) { a.store(zeros<float>(shape(7))); }
                              static_cast<void>(a.load());
                          }
                      }).capture();
        expect(eq(kernel.valid(), iterations != 0));
        if (iterations == 0) { expect(has_diagnostic(kernel, "definite preceding store")); }
    }
    auto uninitialized = tile_kernel("memory_uninitialized_loop", [] {
                             for (auto &nest : parallel(shape(1))) {
                                 auto a = memory<float>(shape(7));
                                 for (auto &step : nest.serial(shape(3))) { a.store(a.load() + 1.0f); }
                             }
                         }).capture();
    expect(!uninitialized.valid());
    expect(has_diagnostic(uninitialized, "definite preceding store"));

    auto stale = tile_kernel("memory_stale", [] {
                     auto a = memory<float>(shape(7));
                     a.store(zeros<float>(shape(7)));
                     a.store(full<float>(shape(7), 1.0f));
                     static_cast<void>(a.load());
                 }).capture();
    expect(stale.valid());
    auto stores = operations(stale.function().body(), OperationKind::MEMORY_STORE);
    auto loads = operations(stale.function().body(), OperationKind::MEMORY_LOAD);
    if (stores.size() != 2u || loads.size() != 1u) {
        expect(false);
        return;
    }
    auto reaching = loads[0]->operand(1);
    loads[0]->set_operand(1, stores[0]->result(0));
    expect(!stale.valid());
    expect(has_diagnostic(stale, "reaching MemoryState"));
    loads[0]->set_operand(1, reaching);
    expect(stale.valid());
}

void test_lexical_ownership() {
    auto valid = tile_kernel("memory_ancestor_read", [] {
                     for (auto &group : parallel(shape(2), exec::Scope::GROUP)) {
                         auto a = memory<float>(shape(7), mem::shared);
                         a.store(zeros<float>(shape(7)));
                         for (auto &worker : group.parallel(shape(3), exec::Scope::WORKER)) {
                             auto b = memory<float>(shape(7), mem::private_);
                             b.store(a.load());
                         }
                     }
                 }).capture();
    expect(valid.valid());

    auto raced = tile_kernel("memory_ancestor_write", [] {
                     for (auto &group : parallel(shape(2))) {
                         auto a = memory<float>(shape(7));
                         for (auto &worker : group.parallel(shape(3))) { a.store(zeros<float>(shape(7))); }
                     }
                 }).capture();
    expect(!raced.valid());

    auto mismatched = tile_kernel("memory_shape_mismatch", [] {
                          auto a = memory<float>(shape(7));
                          a.store(zeros<float>(shape(8)));
                      }).capture();
    expect(!mismatched.valid());
    expect(has_diagnostic(mismatched, "same element space and type"));

    auto impure = tile_kernel("memory_in_map", [] {
                      static_cast<void>(map<float>(shape(7), [](const Nest &) {
                          auto a = memory<float>(shape(1));
                          a.store(zeros<float>(shape(1)));
                          return a.load().at(coord(0));
                      }));
                  }).capture();
    expect(!impure.valid());
    expect(has_diagnostic(impure, "tile.map is pure"));
}

void test_capture_lifetime() {
    std::optional<Memory<float>> escaped;
    auto original = tile_kernel("memory_original_capture", [&] {
                        escaped.emplace(memory<float>(shape(7)));
                        escaped->store(zeros<float>(shape(7)));
                    }).capture();
    expect(original.valid());
    expect(!escaped->valid());
    // Both a still-live old Module and an expired allocation must be rejected
    // before consulting the old IR. Repeated captures can reuse host addresses.
    for (auto i = 0; i < 8; i++) {
        auto foreign = tile_kernel("memory_foreign_capture", [&] {
                           static_cast<void>(escaped->load());
                           escaped->store(zeros<float>(shape(7)));
                       }).capture();
        expect(!foreign.valid());
        expect(has_diagnostic(foreign, "live resource from the active capture"));
    }
    original = tile_kernel("memory_replaced_capture", [] {}).capture();
    auto expired = tile_kernel("memory_expired_capture", [&] { static_cast<void>(escaped->load()); }).capture();
    expect(!expired.valid());
    expect(has_diagnostic(expired, "live resource from the active capture"));
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    "tile_memory_identity_and_snapshot"_test = test_identity_and_snapshot;
    "tile_memory_temporal_carries"_test = test_temporal_carries;
    "tile_memory_initialization_and_stale_state"_test = test_initialization_and_stale_states;
    "tile_memory_lexical_ownership"_test = test_lexical_ownership;
    "tile_memory_capture_lifetime"_test = test_capture_lifetime;
}
