// Stream-scoped scratch for the single Vulkan argument-planning pass (plan P1.4).
//
// Why this exists
// ---------------
// A Vulkan dispatch that needs an argument block used to plan that block twice
// (plan §4, F4: `argument_plans == 2 x dispatches`):
//
//   1. in the sizing pass of `CommandBuffer::execute`, only to learn how large
//      the batch's argument-buffer upload has to be, and
//   2. again in the per-layer preprocessing pass, which emits the block and its
//      trailers into that upload.
//
// The sizing pass cannot simply be moved after reordering: the upload buffer is
// allocated once per batch, before any block is written, and its cumulative size
// is only known once every command has been planned. So the second plan is what
// has to go, which is what this file provides.
//
// `ArgumentPlanTable` keeps the one `PlannedArgumentBlock` of every command of
// the current batch and is keyed by the *identity of the command object*
// (`Command const *`), never by a positional index: reordering permutes the
// emission order, so "the i-th planned dispatch" is a different command in the
// preprocessing pass than it was in the sizing pass. Pointers are unique by
// construction - the batch owns its commands through `unique_ptr`, and the
// commands the software-RTX fallback splices in are owned by the expansion list
// that lives in the same `execute()` frame - so recording one command twice is a
// bug, and so is preprocessing a command that was never planned. Both fail an
// assertion instead of silently re-planning or pairing a command with somebody
// else's layout.
//
// Lifetime and reentrancy
// -----------------------
// `ArgumentScratch` lives on the `Stream` (like `_uniform_data`,
// `_dispatch_offsets` and the other per-batch scratch), so nothing here
// allocates per dispatch: the table and both trailer vectors keep their capacity
// across batches and are cleared at the start of each one. The only user is
// `CommandBuffer::execute`, which the stream calls under its own
// `_dispatch_mtx`, so the scratch is serialized by that mutex; `in_use` asserts
// that assumption instead of trusting it (a re-entrant `execute` on one stream
// would otherwise overwrite a plan the outer frame still needs).
//
// The keys are only ever compared, never dereferenced, and the table is cleared
// when the batch ends, so no pointer outlives the `Command` objects that own it.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include <luisa/core/logging.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/vstl/vector.h>

#include "../common/argument_block_layout.h"
#include "../common/storage_buffer_metadata.h"

namespace lc::vk::detail {

// The per-dispatch argument block of one command: its layout (uniform arguments
// plus trailers) and the trailer placement the emitter has to fill.
struct PlannedArgumentBlock {
    ArgumentBlockLayout layout;
    size_t buffer_metadata_count{};
    ArgumentBlockTrailerPlacement trailer{};
};

// One entry of a layer's argument-block table: which command owns the block and
// where that block sits in the batch's argument-buffer upload. Kept next to the
// offset so the encoding pass can assert that it is about to bind the block of
// the very command it is encoding.
struct DispatchArgumentBlock {
    Command const *command{nullptr};
    size_t offset{0u};
    size_t size{0u};
};

// The plan of every command of one batch, keyed by command identity.
class ArgumentPlanTable {
private:
    struct Slot {
        Command const *command{nullptr};
        size_t plan_index{0u};
        bool consumed{false};
    };

    // `_commands[i]` owns `_plans[i]`. The keys are kept in their own vector so
    // that growing the slot array can rehash them without allocating a temporary
    // copy of the old slots.
    vstd::vector<Command const *> _commands;
    // Parallel to _commands: direct slot positions allow O(N) cleanup without
    // probing through chains that earlier cleanup stores may have broken.
    vstd::vector<size_t> _command_slots;
    vstd::vector<PlannedArgumentBlock> _plans;
    // Open addressing with linear probing. `_slots.size()` is always a power of
    // two and always strictly larger than `_plans.size()`, so a probe always
    // terminates on an empty slot.
    vstd::vector<Slot> _slots;
    size_t _consumed{0u};
    // TODO: use luisa::hash API here
    [[nodiscard]] static size_t _hash(Command const *command) noexcept {
        // Commands are aligned heap objects, so the low pointer bits carry no
        // information; drop them and mix what is left.
        auto value = reinterpret_cast<uintptr_t>(command) / sizeof(void *);
        value ^= value >> 17u;
        value *= static_cast<uintptr_t>(0x9e3779b97f4a7c15ull);
        value ^= value >> 17u;
        return static_cast<size_t>(value);
    }

    // Smallest power of two >= count + ceil(count / 2) + 1, at least 64: enough
    // headroom that a pre-sized batch never fills the table and every probe
    // still meets an empty slot.
    [[nodiscard]] static size_t _capacity_for(size_t count) noexcept {
        auto target = count + (count - (count >> 1u)) + 1u;
        auto capacity = size_t{64u};
        while (capacity < target) {
            if (capacity > (std::numeric_limits<size_t>::max() >> 1u)) {
                return capacity;
            }
            capacity <<= 1u;
        }
        return capacity;
    }

    void _empty_slots(size_t capacity) noexcept {
        _slots.clear();
        vstd::push_back_all(_slots, capacity, Slot{});
    }

    void _rehash(size_t capacity) noexcept {
        _empty_slots(capacity);
        _command_slots.clear();
        for (auto i = size_t{0u}; i < _plans.size(); ++i) {
            auto mask = _slots.size() - 1u;
            auto slot_index = _hash(_commands[i]) & mask;
            while (_slots[slot_index].command != nullptr) {
                slot_index = (slot_index + 1u) & mask;
            }
            _slots[slot_index].command = _commands[i];
            _slots[slot_index].plan_index = i;
            _command_slots.emplace_back(slot_index);
        }
    }

public:
    // Prepares the table for a batch that may contain `command_count` commands.
    // Clears every association and grows the slot array once for the largest
    // batch seen so far; a batch of the same or a smaller size allocates nothing
    // here.
    void begin_batch(size_t command_count) noexcept {
        clear();
        auto const capacity = _capacity_for(command_count);
        if (capacity > _slots.size()) { _empty_slots(capacity); }
    }

    // Drops every association but keeps the capacity, so the next batch of the
    // same size allocates nothing.
    void clear() noexcept {
        if (_commands.empty()) {
            _plans.clear();
            _consumed = 0u;
            return;
        }
        // Clear exactly the slots occupied by the previous batch. Sweeping the
        // entire capacity here would make a one-dispatch batch after a large
        // batch pay the large batch's cost again (and begin_batch calls clear
        // once more after the execute scope guard has already cleaned up).
        for (auto slot_index : _command_slots) {
            LUISA_ASSERT(
                slot_index < _slots.size() &&
                    _slots[slot_index].command != nullptr,
                "Vulkan argument-plan table lost an occupied slot while clearing.");
            _slots[slot_index] = Slot{};
        }
        _commands.clear();
        _command_slots.clear();
        _plans.clear();
        _consumed = 0u;
    }

    // Records the plan of `command`. A command may be planned once per batch.
    void record(Command const *command,
                const PlannedArgumentBlock &plan) noexcept {
        LUISA_ASSERT(
            command != nullptr,
            "Vulkan argument planning requires a non-null command identity.");
        // Keep the load factor at or below 1/2, so probe chains stay short and an
        // empty slot is always available for the next insert.
        if (_slots.empty() || (_plans.size() + 1u) * 2u > _slots.size()) {
            _rehash(_slots.empty() ? size_t{64u} : _slots.size() * 2u);
        }
        auto mask = _slots.size() - 1u;
        auto slot_index = _hash(command) & mask;
        for (;;) {
            auto &slot = _slots[slot_index];
            if (slot.command == nullptr) {
                slot.command = command;
                slot.plan_index = _plans.size();
                _commands.emplace_back(command);
                _command_slots.emplace_back(slot_index);
                _plans.emplace_back(plan);
                return;
            }
            LUISA_ASSERT(
                slot.command != command,
                "Vulkan argument planning planned one command twice in one "
                "batch; the sizing pass must plan every dispatch exactly once.");
            slot_index = (slot_index + 1u) & mask;
        }
    }

    // The plan of `command`, or nullptr when the sizing pass never planned it.
    // Every successful lookup is counted so a batch can assert that it
    // preprocessed exactly as many commands as it planned.
    [[nodiscard]] const PlannedArgumentBlock *consume(
        Command const *command) noexcept {
        if (command == nullptr || _slots.empty()) { return nullptr; }
        auto mask = _slots.size() - 1u;
        for (auto slot_index = _hash(command) & mask;; slot_index = (slot_index + 1u) & mask) {
            auto &slot = _slots[slot_index];
            if (slot.command == nullptr) { return nullptr; }
            if (slot.command == command) {
                LUISA_ASSERT(
                    !slot.consumed,
                    "Vulkan argument preprocessing consumed one command plan twice.");
                slot.consumed = true;
                ++_consumed;
                return &_plans[slot.plan_index];
            }
        }
    }

    [[nodiscard]] size_t size() const noexcept { return _plans.size(); }
    [[nodiscard]] size_t consumed() const noexcept { return _consumed; }
};

// Everything the single argument-planning pass has to remember between the
// phases of one command batch. Lives on the `Stream`.
struct ArgumentScratch {
    ArgumentPlanTable plans;
    // Trailer payload of the dispatch being preprocessed. Cleared and refilled
    // per dispatch (see `CommandBuffer::execute`), so no value of a previous
    // dispatch can survive into the next block.
    vstd::vector<StorageBufferMetadata> buffer_metadata;
    vstd::vector<uint32_t> validation_values;
    // Guards the "one execute per stream at a time" assumption the scratch makes.
    bool in_use{false};
};

}// namespace lc::vk::detail