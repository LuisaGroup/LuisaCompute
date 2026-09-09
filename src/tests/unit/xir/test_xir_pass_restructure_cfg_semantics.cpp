// Compare observable execution, not merely the validity of the output graph.
#include "ut/ut.hpp"
#include <luisa/ast/type_registry.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/debug_printer.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/restructure_cfg.h>
#include <luisa/xir/verifier.h>
#include <luisa/xir/translators/xir_interchange.h>
#include <array>
#include <charconv>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
namespace {
struct ExecutionResult {
    uint32_t value;
    std::vector<uint32_t> stores;
    bool operator==(const ExecutionResult &) const = default;
};

// Deliberately tiny scalar interpreter. The verifier separately enforces SSA
// dominance; re-entering a block invalidates its old scalar definitions here.
// Missing operands, uninitialized memory, unsupported types/operations, and
// nontermination all fail. Compare the complete observable store sequence so
// duplicated or dropped side effects cannot hide behind an equal final value.
std::optional<ExecutionResult> execute(FunctionDefinition *f, Value *input, Value *output, uint32_t argument) {
    std::unordered_map<const Value *, uint32_t> values{{input, argument}};
    struct Address {
        const Value *root;
        uint32_t index;
    };
    std::unordered_map<const Value *, Address> addresses{{output, {output, 0u}}};
    std::unordered_map<const Value *, std::vector<std::optional<uint32_t>>> memory{{output, {std::nullopt}}};
    auto locate = [&](const Value *variable) -> std::optional<uint32_t> * {
        auto address = addresses.find(variable);
        if (address == addresses.end()) { return nullptr; }
        auto allocation = memory.find(address->second.root);
        if (allocation == memory.end() || address->second.index >= allocation->second.size()) { return nullptr; }
        return &allocation->second[address->second.index];
    };
    std::vector<uint32_t> stores;
    bool valid = true;
    auto get = [&](const Value *v) -> uint32_t {
        if (v == nullptr || v->type() == nullptr ||
            (v->type() != Type::of<uint32_t>() && v->type() != Type::of<bool>())) {
            valid = false;
            return 0u;
        }
        if (v->isa<Constant>()) {
            auto *c = static_cast<const Constant *>(v);
            return c->type()->is_bool() ? uint32_t(c->as<bool>()) : c->as<uint32_t>();
        }
        auto it = values.find(v);
        if (it == values.end()) {
            valid = false;
            return 0u;
        }
        return it->second;
    };
    auto *block = f->body_block();
    for (auto step = 0u; step < 100000u && valid; ++step) {
        BasicBlock *next = nullptr;
        for (auto *inst : block->instructions()) {
            values.erase(inst);
            addresses.erase(inst);
        }
        for (auto *inst : block->instructions()) {
            if (inst->isa<AllocaInst>()) {
                auto *type = inst->type();
                auto scalar = type == Type::of<uint32_t>() || type == Type::of<bool>();
                auto array = type->is_array() && type->element() == Type::of<uint32_t>();
                if (!scalar && !array) { return std::nullopt; }
                addresses[inst] = {inst, 0u};
                memory[inst] = std::vector<std::optional<uint32_t>>(scalar ? 1u : type->dimension());
            } else if (inst->isa<GEPInst>()) {
                auto *gep = static_cast<GEPInst *>(inst);
                auto base = addresses.find(gep->base());
                if (base == addresses.end()) { return std::nullopt; }
                auto address = base->second;
                if (gep->index_count() != 0u) {
                    auto *type = gep->base()->type();
                    if (gep->index_count() != 1u || !type->is_array() || type->element() != Type::of<uint32_t>()) { return std::nullopt; }
                    auto index = get(gep->index(0u));
                    if (!valid || index >= type->dimension()) { return std::nullopt; }
                    address.index += index;
                }
                addresses[inst] = address;
                if (locate(inst) == nullptr) { return std::nullopt; }
            } else if (inst->isa<LoadInst>()) {
                auto *slot = locate(static_cast<LoadInst *>(inst)->variable());
                if (slot == nullptr || !slot->has_value()) { return std::nullopt; }
                values[inst] = **slot;
            } else if (inst->isa<StoreInst>()) {
                auto *s = static_cast<StoreInst *>(inst);
                auto *slot = locate(s->variable());
                if (slot == nullptr) { return std::nullopt; }
                auto value = get(s->value());
                if (!valid) { return std::nullopt; }
                *slot = value;
                if (addresses.at(s->variable()).root == output) { stores.emplace_back(value); }
            } else if (inst->isa<ArithmeticInst>()) {
                auto *a = static_cast<ArithmeticInst *>(inst);
                if (a->operand_count() != 1u && a->operand_count() != 2u) { return std::nullopt; }
                auto x = get(a->operand(0));
                if (a->operand_count() == 1u) {
                    switch (a->op()) {
                        case ArithmeticOp::UNARY_MINUS: values[a] = 0u - x; break;
                        case ArithmeticOp::UNARY_BIT_NOT: values[a] = a->type()->is_bool() ? uint32_t(!x) : ~x; break;
                        default: return std::nullopt;
                    }
                } else {
                    auto y = get(a->operand(1));
                    switch (a->op()) {
                        case ArithmeticOp::BINARY_ADD: values[a] = x + y; break;
                        case ArithmeticOp::BINARY_SUB: values[a] = x - y; break;
                        case ArithmeticOp::BINARY_MUL: values[a] = x * y; break;
                        case ArithmeticOp::BINARY_BIT_AND: values[a] = x & y; break;
                        case ArithmeticOp::BINARY_BIT_OR: values[a] = x | y; break;
                        case ArithmeticOp::BINARY_BIT_XOR: values[a] = x ^ y; break;
                        case ArithmeticOp::BINARY_LESS: values[a] = x < y; break;
                        case ArithmeticOp::BINARY_LESS_EQUAL: values[a] = x <= y; break;
                        case ArithmeticOp::BINARY_GREATER: values[a] = x > y; break;
                        case ArithmeticOp::BINARY_GREATER_EQUAL: values[a] = x >= y; break;
                        case ArithmeticOp::BINARY_EQUAL: values[a] = x == y; break;
                        case ArithmeticOp::BINARY_NOT_EQUAL: values[a] = x != y; break;
                        default: return std::nullopt;
                    }
                }
            } else if (inst->isa<ReturnInst>()) {
                if (static_cast<ReturnInst *>(inst)->return_value() != nullptr) { return std::nullopt; }
                auto *slot = locate(output);
                return valid && slot != nullptr && slot->has_value() ? std::optional{ExecutionResult{**slot, std::move(stores)}} : std::nullopt;
            } else if (inst->isa<LoopInst>()) {
                next = static_cast<LoopInst *>(inst)->prepare_block();
            } else if (inst->isa<SimpleLoopInst>()) {
                next = static_cast<SimpleLoopInst *>(inst)->body_block();
            } else if (inst->isa<IfInst>()) {
                auto *c = static_cast<IfInst *>(inst);
                next = get(c->condition()) ? c->true_block() : c->false_block();
            } else if (inst->isa<ConditionalBranchInst>()) {
                auto *c = static_cast<ConditionalBranchInst *>(inst);
                next = get(c->condition()) ? c->true_block() : c->false_block();
            } else if (inst->isa<SwitchInst>() || inst->isa<IndexedBranchInst>()) {
                auto *s = static_cast<IndexedBranchTerminatorInstruction *>(inst);
                auto selector = get(s->value());
                next = s->default_block();
                for (auto i = 0u; i < s->case_count(); ++i) {
                    if (selector == s->case_value(i)) {
                        next = s->case_block(i);
                        break;
                    }
                }
            } else if (inst->isa<BranchInst>() || inst->isa<BreakInst>() || inst->isa<ContinueInst>()) {
                next = static_cast<BranchTerminatorInstruction *>(inst)->target_block();
            } else {
                return std::nullopt;
            }
        }
        if (next == nullptr) { return std::nullopt; }
        block = next;
    }
    return std::nullopt;
}

std::optional<uint32_t> environment_uint(const char *name, uint32_t upper_bound) {
    if (auto *text = std::getenv(name)) {
        uint32_t value = 0u;
        std::string_view str{text};
        auto [end, error] = std::from_chars(str.data(), str.data() + str.size(), value);
        auto valid = error == std::errc{} && end == str.data() + str.size() && value <= upper_bound;
        expect(valid) << "invalid " << name;
        if (valid) { return value; }
    }
    return std::nullopt;
}

// Opt-in repro artifacts are named per case/mode and use a caller-provided
// directory, so a routine test run neither writes /tmp nor overwrites evidence.
void dump_graph(Module &module, std::string_view name, RestructureCFGMutationMode mode, bool before) {
    if (auto *directory = std::getenv("LUISA_XIR_CFG_TEST_DUMP_DIR")) {
        std::filesystem::create_directories(directory);
        auto suffix = mode == RestructureCFGMutationMode::TRANSACTIONAL ? "-transactional" : "-in-place";
        auto filename = std::string{name} + suffix + (before ? "-input.xir" : "-output.xir");
        auto interchange = xir_to_interchange_text(&module);
        if (!interchange.succeeded()) {
            // Canonical interchange rejects malformed IR. Preserve the actual
            // failing graph in a distinct debug artifact for verifier errors.
            filename += ".invalid-debug";
            XIRDebugPrinter printer;
            printer.emit_module(interchange.text, &module);
        }
        std::ofstream file{std::filesystem::path{directory} / filename};
        file << interchange.text;
        expect(file.good());
    }
}

template<typename Oracle>
void check_execution(Module &module, FunctionDefinition *f, Value *input, Value *output,
                     RestructureCFGMutationMode mode, std::string_view name, uint32_t input_count, Oracle &&oracle) {
    auto before_verification = xir_verify_module(&module);
    expect(before_verification.succeeded()) << name;
    if (!before_verification.succeeded()) { return; }
    std::vector<std::optional<ExecutionResult>> expected(input_count);
    for (auto i = 0u; i < input_count; ++i) {
        expected[i] = execute(f, input, output, i);
        expect(expected[i].has_value()) << name << " source input=" << i;
        oracle(i, expected[i]);
    }
    dump_graph(module, name, mode, true);
    auto info = restructure_cfg_pass_run_on_function(f, {.mutation_mode = mode, .verify_remaining_divergent_index = true});
    expect(info.succeeded()) << name << " mode=" << uint32_t(mode);
    dump_graph(module, name, mode, false);
    if (!info.succeeded()) { return; }
    auto after_verification = xir_verify_module(&module, {.require_no_phi = true, .require_unique_merge_blocks = true, .require_canonical_break_continue_targets = true});
    expect(after_verification.succeeded()) << name;
    if (!after_verification.succeeded()) { return; }
    for (auto i = 0u; i < input_count; ++i) {
        auto actual = execute(f, input, output, i);
        expect(expected[i].has_value() && actual == expected[i]) << name << " input=" << i << " mode=" << uint32_t(mode);
    }
}

void check_graph(uint32_t seed, bool cyclic, RestructureCFGMutationMode mode, bool cross_block_addresses = false) {
    Module module;
    auto *f = module.create_kernel();
    auto *entry = f->create_body_block();
    auto *input = f->create_value_argument(Type::of<uint32_t>());
    auto *output = f->create_reference_argument(Type::of<uint32_t>());
    auto n = environment_uint("LUISA_XIR_CFG_TEST_SIZE", 64u).value_or(12u);
    expect(n != 0u);
    if (n == 0u) { return; }
    std::vector<BasicBlock *> blocks(n + 1u);
    // Allocation order is independent of executable order.
    for (auto i = 0u; i <= n; ++i) { blocks[(i + seed) % (n + 1u)] = f->create_basic_block(); }
    auto *latch = f->create_basic_block();
    auto *exit = f->create_basic_block();
    auto *header = f->create_basic_block();
    auto constant = [&](uint32_t x) { return module.create_constant(Type::of<uint32_t>(), &x); };
    XIRBuilder b;
    b.set_insertion_point(entry);
    auto *counter = b.alloca_local(Type::of<uint32_t>());
    auto *storage = cross_block_addresses ? b.alloca_local(Type::of<std::array<uint32_t, 2u>>()) : nullptr;
    b.store(counter, constant(0));
    b.store(output, constant(17));
    b.br(header);
    b.set_insertion_point(header);
    auto *iteration = b.load(Type::of<uint32_t>(), counter);
    b.br(blocks[0]);
    auto random = seed + 1u;
    auto rng = [&] { random = random * 1664525u + 1013904223u; return random; };
    for (auto i = 0u; i < n; ++i) {
        b.set_insertion_point(blocks[i]);
        auto *old = b.load(Type::of<uint32_t>(), output);
        auto *hash = b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_MUL, {old, constant(31)});
        auto *sum = b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_ADD, {hash, constant(i + 1u)});
        if (cross_block_addresses) {
            auto *offset = b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_ADD, {input, iteration});
            auto *index = b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_BIT_AND, {offset, constant(1u)});
            auto *address = b.gep(Type::of<uint32_t>(), storage, {index});
            b.store(address, sum);
            // Each consumer starts in its own block. A re-entry SCC may begin
            // after the original address/value definition and later revisit
            // its copied definition with a different loop-iteration index.
            auto *consumer = f->create_basic_block();
            b.br(consumer);
            b.set_insertion_point(consumer);
            auto *loaded = b.load(Type::of<uint32_t>(), address);
            b.store(output, b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_ADD, {loaded, sum}));
        } else {
            b.store(output, sum);
        }
        auto *choice = b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_ADD, {input, iteration});
        auto mask = 1u << (i % 5u);
        auto *bit = b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_BIT_AND, {choice, constant(mask)});
        auto next = [&](uint32_t r) { return blocks[i + 1u + r % (n - i)]; };
        auto *left = next(rng()), *right = next(rng());
        if (i % 3u == 0u) {
            auto *s = b.indexed_branch(bit);
            s->set_default_block(right);
            s->add_case(0u, left);
            s->add_case(mask, next(rng()));
        } else {
            auto *condition = b.call(Type::of<bool>(), ArithmeticOp::BINARY_EQUAL, {bit, constant(0)});
            b.cond_br(condition, left, right);
        }
    }
    b.set_insertion_point(blocks[n]);
    b.br(latch);
    b.set_insertion_point(latch);
    auto *inc = b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_ADD, {iteration, constant(1)});
    b.store(counter, inc);
    if (cyclic) {
        auto *again = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {inc, constant(3)});
        b.cond_br(again, header, exit);
    } else {
        b.br(exit);
    }
    b.set_insertion_point(exit);
    b.return_void();
    auto name = std::string{cyclic ? "cyclic-" : "acyclic-"} + std::to_string(seed) + "-size-" + std::to_string(n);
    if (cross_block_addresses) { name += "-cross-block-addresses"; }
    check_execution(module, f, input, output, mode, name, 32u, [](auto, auto const &) {});
}

ExecutionResult nested_loop_oracle(uint32_t input, uint32_t seed) {
    ExecutionResult result{17u, {17u}};
    auto emit = [&](uint32_t value) {
        result.value = result.value * 37u + value;
        result.stores.emplace_back(result.value);
    };
    auto outer_limit = (input & 3u) + 1u;
    auto inner_limit = ((input ^ seed) & 3u) + 1u;
    for (auto outer = 0u; outer < outer_limit; ++outer) {
        auto payload = input + 31u * outer + seed;
        auto local = payload ^ 0x5au;
        auto inner = 0u;
        for (; inner < inner_limit; ++inner) {
            auto value = payload + 13u * inner;
            emit(value);
            if ((value & 31u) == 5u) {
                emit(0xf1u);
                return result;
            }
            if ((value & 3u) == 0u) { continue; }
            if ((value & 3u) == 1u) {
                emit(0x70u);
                break;
            }
            if ((value & 3u) == 2u) {
                emit(0x90u);
                emit(0xeeu);
                return result;
            }
            local += value * 7u;
            emit(local);
        }
        emit(local);
        if ((payload & 7u) == 2u) {
            emit(0xf1u);
            return result;
        }
    }
    emit(0xeeu);
    return result;
}

void check_nested_loop_graph(uint32_t seed, RestructureCFGMutationMode mode) {
    Module module;
    auto *f = module.create_kernel();
    auto *input = f->create_value_argument(Type::of<uint32_t>());
    auto *output = f->create_reference_argument(Type::of<uint32_t>());
    enum Block : uint32_t {
        ENTRY,
        OUTER_HEADER,
        OUTER_BODY,
        INNER_HEADER,
        INNER_BODY,
        DISPATCH,
        WORK,
        INNER_CONTINUE,
        INNER_BREAK,
        OUTER_BREAK,
        AFTER_INNER,
        OUTER_LATCH,
        EARLY_RETURN,
        NORMAL_RETURN,
        COUNT
    };
    std::array<BasicBlock *, COUNT> blocks{};
    blocks[ENTRY] = f->create_body_block();
    for (auto i = 1u; i < COUNT; ++i) {
        blocks[1u + (i - 1u + seed) % (COUNT - 1u)] = f->create_basic_block();
    }
    XIRBuilder b;
    auto constant = [&](uint32_t x) { return module.create_constant(Type::of<uint32_t>(), &x); };
    auto binary = [&](ArithmeticOp op, Value *a, Value *c) { return b.call(Type::of<uint32_t>(), op, {a, c}); };
    auto add = [&](Value *a, Value *c) { return binary(ArithmeticOp::BINARY_ADD, a, c); };
    auto mul = [&](Value *a, Value *c) { return binary(ArithmeticOp::BINARY_MUL, a, c); };
    auto mask = [&](Value *a, uint32_t c) { return binary(ArithmeticOp::BINARY_BIT_AND, a, constant(c)); };
    auto equal = [&](Value *a, uint32_t c) { return b.call(Type::of<bool>(), ArithmeticOp::BINARY_EQUAL, {a, constant(c)}); };
    auto emit = [&](Value *value) {
        auto *old = b.load(Type::of<uint32_t>(), output);
        b.store(output, add(mul(old, constant(37u)), value));
    };
    b.set_insertion_point(blocks[ENTRY]);
    auto *outer_slot = b.alloca_local(Type::of<uint32_t>());
    auto *inner_slot = b.alloca_local(Type::of<uint32_t>());
    b.store(outer_slot, constant(0u));
    b.store(output, constant(17u));
    auto *outer_limit = add(mask(input, 3u), constant(1u));
    auto *inner_limit = add(mask(binary(ArithmeticOp::BINARY_BIT_XOR, input, constant(seed)), 3u), constant(1u));
    b.br(blocks[OUTER_HEADER]);
    b.set_insertion_point(blocks[OUTER_HEADER]);
    auto *outer = b.load(Type::of<uint32_t>(), outer_slot);
    b.cond_br(b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {outer, outer_limit}), blocks[OUTER_BODY], blocks[NORMAL_RETURN]);
    b.set_insertion_point(blocks[OUTER_BODY]);
    auto *payload = add(add(input, mul(outer, constant(31u))), constant(seed));
    // This declaration executes once per outer iteration. Its initialized
    // storage and the scalar payload both cross the complete inner region.
    auto *local = b.alloca_local(Type::of<uint32_t>());
    b.store(local, binary(ArithmeticOp::BINARY_BIT_XOR, payload, constant(0x5au)));
    b.store(inner_slot, constant(0u));
    b.br(blocks[INNER_HEADER]);
    b.set_insertion_point(blocks[INNER_HEADER]);
    auto *inner = b.load(Type::of<uint32_t>(), inner_slot);
    b.cond_br(b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {inner, inner_limit}), blocks[INNER_BODY], blocks[AFTER_INNER]);
    b.set_insertion_point(blocks[INNER_BODY]);
    auto *value = add(payload, mul(inner, constant(13u)));
    emit(value);
    b.cond_br(equal(mask(value, 31u), 5u), blocks[EARLY_RETURN], blocks[DISPATCH]);
    b.set_insertion_point(blocks[DISPATCH]);
    auto *dispatch = b.indexed_branch(mask(value, 3u));
    dispatch->add_case(0u, blocks[INNER_CONTINUE]);
    dispatch->add_case(1u, blocks[INNER_BREAK]);
    dispatch->add_case(2u, blocks[OUTER_BREAK]);
    dispatch->set_default_block(blocks[WORK]);
    b.set_insertion_point(blocks[WORK]);
    auto *updated = add(b.load(Type::of<uint32_t>(), local), mul(value, constant(7u)));
    b.store(local, updated);
    emit(updated);
    b.br(blocks[INNER_CONTINUE]);
    b.set_insertion_point(blocks[INNER_CONTINUE]);
    b.store(inner_slot, add(inner, constant(1u)));
    b.br(blocks[INNER_HEADER]);
    b.set_insertion_point(blocks[INNER_BREAK]);
    emit(constant(0x70u));
    b.br(blocks[AFTER_INNER]);
    b.set_insertion_point(blocks[OUTER_BREAK]);
    emit(constant(0x90u));
    b.br(blocks[NORMAL_RETURN]);
    b.set_insertion_point(blocks[AFTER_INNER]);
    emit(b.load(Type::of<uint32_t>(), local));
    b.cond_br(equal(mask(payload, 7u), 2u), blocks[EARLY_RETURN], blocks[OUTER_LATCH]);
    b.set_insertion_point(blocks[OUTER_LATCH]);
    b.store(outer_slot, add(outer, constant(1u)));
    b.br(blocks[OUTER_HEADER]);
    b.set_insertion_point(blocks[EARLY_RETURN]);
    emit(constant(0xf1u));
    b.return_void();
    b.set_insertion_point(blocks[NORMAL_RETURN]);
    emit(constant(0xeeu));
    b.return_void();
    auto name = std::string{"nested-loops-"} + std::to_string(seed);
    check_execution(module, f, input, output, mode, name, 64u, [&](uint32_t argument, auto const &expected) {
        expect(expected == std::optional{nested_loop_oracle(argument, seed)}) << name << " oracle input=" << argument;
    });
}

void check_cloned_frontier_graph(RestructureCFGMutationMode mode, bool dynamic_gep) {
    Module module;
    auto *f = module.create_kernel();
    auto *input = f->create_value_argument(Type::of<uint32_t>());
    auto *output = f->create_reference_argument(Type::of<uint32_t>());
    auto *entry = f->create_body_block();
    auto *shared = f->create_basic_block();
    auto *predecessor = f->create_basic_block();
    auto *merge = f->create_basic_block();
    auto constant = [&](uint32_t x) { return module.create_constant(Type::of<uint32_t>(), &x); };
    XIRBuilder b;
    b.set_insertion_point(entry);
    b.store(output, constant(17u));
    auto *bit = b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_BIT_AND, {input, constant(1u)});
    auto *condition = b.call(Type::of<bool>(), ArithmeticOp::BINARY_EQUAL, {bit, constant(0u)});
    auto *selection = b.if_(condition);
    selection->set_true_target(shared);
    selection->set_false_target(predecessor);
    selection->set_merge_block(merge);
    b.set_insertion_point(predecessor);
    b.store(output, constant(100u));
    b.br(shared);
    b.set_insertion_point(shared);
    auto *local = b.alloca_local(dynamic_gep ? Type::of<std::array<uint32_t, 2u>>() : Type::of<uint32_t>());
    Value *variable = local;
    if (dynamic_gep) {
        // Only the selected element is initialized. A missing transported
        // index therefore fails as an uninitialized read, even for zero.
        auto *flipped = b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_BIT_XOR, {input, constant(1u)});
        auto *index = b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_BIT_AND, {flipped, constant(1u)});
        variable = b.gep(Type::of<uint32_t>(), local, {index});
    }
    auto *initial = b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_ADD, {input, constant(7u)});
    b.store(variable, initial);
    auto *scalar = b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_MUL, {input, constant(3u)});
    b.br(merge);
    b.set_insertion_point(merge);
    auto *loaded = b.load(Type::of<uint32_t>(), variable);
    b.store(output, b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_ADD, {loaded, scalar}));
    b.return_void();
    // Entry splitting must transport both SSA and the alloca's identity to an
    // un-cloned frontier. A fresh private alloca in the clone loses its store.
    check_execution(module, f, input, output, mode, dynamic_gep ? "cloned-gep-frontier" : "cloned-frontier", 32u, [](uint32_t argument, auto const &expected) {
        ExecutionResult oracle{4u * argument + 7u, {17u}};
        if ((argument & 1u) != 0u) { oracle.stores.emplace_back(100u); }
        oracle.stores.emplace_back(oracle.value);
        expect(expected == std::optional{oracle}) << "cloned frontier oracle input=" << argument;
    });
}

void check_cloned_merge_graph(RestructureCFGMutationMode mode) {
    Module module;
    auto *f = module.create_kernel();
    auto *input = f->create_value_argument(Type::of<uint32_t>());
    auto *output = f->create_reference_argument(Type::of<uint32_t>());
    auto *entry = f->create_body_block();
    auto *shared = f->create_basic_block();
    auto *predecessor = f->create_basic_block();
    auto *left = f->create_basic_block();
    auto *right = f->create_basic_block();
    auto *inner_merge = f->create_basic_block();
    auto *outer_merge = f->create_basic_block();
    auto constant = [&](uint32_t x) { return module.create_constant(Type::of<uint32_t>(), &x); };
    XIRBuilder b;
    auto condition = [&](uint32_t mask) {
        auto *bit = b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_BIT_AND, {input, constant(mask)});
        return b.call(Type::of<bool>(), ArithmeticOp::BINARY_EQUAL, {bit, constant(0u)});
    };
    b.set_insertion_point(entry);
    b.store(output, constant(17u));
    auto *outer = b.if_(condition(1u));
    outer->set_true_target(shared);
    outer->set_false_target(predecessor);
    outer->set_merge_block(outer_merge);
    b.set_insertion_point(shared);
    auto *inner = b.if_(condition(4u));
    inner->set_true_target(left);
    inner->set_false_target(right);
    inner->set_merge_block(inner_merge);
    b.set_insertion_point(left);
    b.store(output, constant(200u));
    b.br(inner_merge);
    b.set_insertion_point(right);
    b.store(output, constant(300u));
    b.br(inner_merge);
    b.set_insertion_point(predecessor);
    b.store(output, constant(100u));
    b.cond_br(condition(2u), shared, inner_merge);
    b.set_insertion_point(inner_merge);
    auto *old = b.load(Type::of<uint32_t>(), output);
    b.store(output, b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_ADD, {old, constant(7u)}));
    b.br(outer_merge);
    b.set_insertion_point(outer_merge);
    b.return_void();
    // The alternative predecessor can bypass the nested selection entirely;
    // its merge is outside the clone region and cannot acquire another owner.
    check_execution(module, f, input, output, mode, "cloned-merge-role", 32u, [](uint32_t argument, auto const &expected) {
        ExecutionResult oracle{17u, {17u}};
        if ((argument & 1u) != 0u) {
            oracle.value = 100u;
            oracle.stores.emplace_back(oracle.value);
        }
        if ((argument & 1u) == 0u || (argument & 2u) == 0u) {
            oracle.value = (argument & 4u) == 0u ? 200u : 300u;
            oracle.stores.emplace_back(oracle.value);
        }
        oracle.value += 7u;
        oracle.stores.emplace_back(oracle.value);
        expect(expected == std::optional{oracle}) << "cloned merge oracle input=" << argument;
    });
}

void check_crossing_loop_epoch_graph(RestructureCFGMutationMode mode, uint32_t rotation) {
    Module module;
    auto *f = module.create_kernel();
    std::array<BasicBlock *, 21u> blocks;
    blocks[0] = f->create_body_block();
    for (auto i = 1u; i < blocks.size(); ++i) {
        blocks[1u + (i - 1u + rotation) % (blocks.size() - 1u)] = f->create_basic_block();
    }
    auto *input = f->create_value_argument(Type::of<uint32_t>());
    auto *output = f->create_reference_argument(Type::of<uint32_t>());
    auto constant = [&](uint32_t value) { return module.create_constant(Type::of<uint32_t>(), &value); };
    XIRBuilder b;
    auto emit = [&](uint32_t tag) {
        auto *old = b.load(Type::of<uint32_t>(), output);
        auto *hash = b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_MUL, {old, constant(37u)});
        auto *value = b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_ADD, {hash, constant(tag)});
        b.store(output, value);
    };
    b.set_insertion_point(blocks[0]);
    auto *counter = b.alloca_local(Type::of<uint32_t>());
    auto *outer_index_slot = b.alloca_local(Type::of<uint32_t>());
    auto *initial_index = b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_BIT_AND, {input, constant(1u)});
    auto *limit_bits = b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_BIT_AND, {input, constant(3u)});
    auto *limit = b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_ADD, {limit_bits, constant(1u)});
    auto *enter_bit = b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_BIT_AND, {input, constant(2u)});
    auto *enter_condition = b.call(Type::of<bool>(), ArithmeticOp::BINARY_NOT_EQUAL, {enter_bit, constant(0u)});
    b.store(counter, constant(0u));
    b.store(outer_index_slot, initial_index);
    b.store(output, constant(17u));
    auto *outer_loop = b.simple_loop();
    outer_loop->set_body_block(blocks[7]);
    outer_loop->set_merge_block(blocks[14]);
    b.set_insertion_point(blocks[7]);
    auto *outer_index = b.load(Type::of<uint32_t>(), outer_index_slot);
    auto *outer = b.switch_(outer_index);
    outer->set_default_block(blocks[9]);
    outer->add_case(0u, blocks[10]);
    outer->add_case(1u, blocks[8]);
    // Preserve the original regression's crossing declaration: this merge
    // belongs inside the nested loop body, beyond its prepare/continue epoch.
    outer->set_merge_block(blocks[17]);
    b.set_insertion_point(blocks[8]);
    emit(8u);
    b.store(outer_index_slot, constant(0u));
    b.br(blocks[7]);
    b.set_insertion_point(blocks[10]);
    auto *enter = b.if_(enter_condition);
    enter->set_true_target(blocks[12]);
    enter->set_false_target(blocks[18]);
    enter->set_merge_block(blocks[18]);
    b.set_insertion_point(blocks[12]);
    auto *inner_loop = b.loop();
    inner_loop->set_prepare_block(blocks[2]);
    inner_loop->set_body_block(blocks[3]);
    inner_loop->set_update_block(blocks[11]);
    inner_loop->set_merge_block(blocks[13]);
    b.set_insertion_point(blocks[2]);
    auto *iteration = b.load(Type::of<uint32_t>(), counter);
    auto *again = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {iteration, limit});
    b.cond_br(again, blocks[3], blocks[13]);
    b.set_insertion_point(blocks[3]);
    auto *choice = b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_ADD, {iteration, input});
    auto *choice_bit = b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_BIT_AND, {choice, constant(1u)});
    auto *take_condition = b.call(Type::of<bool>(), ArithmeticOp::BINARY_EQUAL, {choice_bit, constant(0u)});
    auto *take_switch = b.if_(take_condition);
    take_switch->set_true_target(blocks[19]);
    take_switch->set_false_target(blocks[11]);
    take_switch->set_merge_block(blocks[19]);
    b.set_insertion_point(blocks[4]);
    auto *inner_index = b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_BIT_AND, {input, constant(4u)});
    auto *inner = b.indexed_branch(inner_index);
    inner->set_default_block(blocks[20]);
    inner->add_case(0u, blocks[15]);
    b.set_insertion_point(blocks[15]);
    emit(15u);
    b.br(blocks[20]);
    b.set_insertion_point(blocks[5]);
    emit(5u);
    b.br(blocks[11]);
    b.set_insertion_point(blocks[11]);
    auto *previous = b.load(Type::of<uint32_t>(), counter);
    auto *next = b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_ADD, {previous, constant(1u)});
    b.store(counter, next);
    b.br(blocks[2]);
    b.set_insertion_point(blocks[1]);
    emit(1u);
    b.br(blocks[18]);
    b.set_insertion_point(blocks[18]);
    emit(18u);
    b.br(blocks[6]);
    for (auto [source, target] : {std::pair{13u, 1u}, {17u, 4u}, {19u, 17u}, {20u, 5u}}) {
        b.set_insertion_point(blocks[source]);
        b.br(blocks[target]);
    }
    b.set_insertion_point(blocks[6]);
    b.return_void();
    for (auto i : {9u, 14u, 16u}) {
        b.set_insertion_point(blocks[i]);
        b.unreachable_();
    }
    auto name = std::string{"crossing-loop-epochs-"} + std::to_string(rotation);
    check_execution(module, f, input, output, mode, name, 32u, [](uint32_t argument, auto const &expected) {
        ExecutionResult oracle{17u, {17u}};
        auto emit = [&](uint32_t tag) {
            oracle.value = oracle.value * 37u + tag;
            oracle.stores.emplace_back(oracle.value);
        };
        if ((argument & 1u) != 0u) { emit(8u); }
        if ((argument & 2u) != 0u) {
            for (auto i = 0u; i < (argument & 3u) + 1u; ++i) {
                if (((i + argument) & 1u) == 0u) {
                    if ((argument & 4u) == 0u) { emit(15u); }
                    emit(5u);
                }
            }
            emit(1u);
        }
        emit(18u);
        expect(expected == std::optional{oracle}) << "crossing loop epoch oracle input=" << argument;
    });
}

void check_terminal_payload_graph(RestructureCFGMutationMode mode, bool shared_terminal) {
    Module module;
    auto *f = module.create_kernel();
    auto *input = f->create_value_argument(Type::of<uint32_t>());
    auto *output = f->create_reference_argument(Type::of<uint32_t>());
    auto *entry = f->create_body_block();
    auto *left = f->create_basic_block();
    auto *right = f->create_basic_block();
    auto *left_terminal = f->create_basic_block();
    auto *right_terminal = shared_terminal ? left_terminal : f->create_basic_block();
    auto *merge = f->create_basic_block();
    auto constant = [&](uint32_t x) { return module.create_constant(Type::of<uint32_t>(), &x); };
    XIRBuilder b;
    b.set_insertion_point(entry);
    b.store(output, constant(17u));
    auto *bit = b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_BIT_AND, {input, constant(1u)});
    auto *condition = b.call(Type::of<bool>(), ArithmeticOp::BINARY_EQUAL, {bit, constant(0u)});
    auto *selection = b.if_(condition);
    selection->set_true_target(left);
    selection->set_false_target(right);
    selection->set_merge_block(merge);
    auto arm = [&](BasicBlock *block, BasicBlock *terminal, uint32_t offset) {
        b.set_insertion_point(block);
        auto *value = b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_ADD, {input, constant(offset)});
        b.store(output, value);
        b.br(terminal);
        if (!shared_terminal) {
            b.set_insertion_point(terminal);
            b.store(output, b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_MUL, {value, constant(3u)}));
            b.return_void();
        }
    };
    arm(left, left_terminal, 100u);
    arm(right, right_terminal, 200u);
    if (shared_terminal) {
        b.set_insertion_point(left_terminal);
        auto *value = b.load(Type::of<uint32_t>(), output);
        b.store(output, b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_MUL, {value, constant(3u)}));
        b.return_void();
    }
    b.set_insertion_point(merge);
    b.unreachable_();
    // A Return block may carry ordinary payload and arm-local SSA uses. It
    // belongs inside that arm precisely when the arm entry dominates it;
    // sharing the same terminal across arms still requires a common merge.
    check_execution(module, f, input, output, mode, shared_terminal ? "shared-terminal-payload" : "owned-terminal-payload", 32u, [](uint32_t argument, auto const &expected) {
        auto value = argument + ((argument & 1u) == 0u ? 100u : 200u);
        expect(expected == std::optional{ExecutionResult{value * 3u, {17u, value, value * 3u}}});
    });
    if (!shared_terminal) {
        auto allocas = 0u;
        for (auto *block : f->basic_blocks()) {
            for (auto *inst : block->instructions()) { allocas += inst->isa<AllocaInst>(); }
        }
        expect(allocas == 0u) << "arm-owned terminal payload does not need selector or SSA spill state";
    }
}

void check_nested_exit_cut_graph(RestructureCFGMutationMode mode) {
    Module module;
    auto *f = module.create_kernel();
    auto *input = f->create_value_argument(Type::of<uint32_t>());
    auto *output = f->create_reference_argument(Type::of<uint32_t>());
    auto *entry = f->create_body_block();
    auto *parent = f->create_basic_block();
    auto *child = f->create_basic_block();
    auto *escape = f->create_basic_block();
    auto *alternative = f->create_basic_block();
    auto *child_merge = f->create_basic_block();
    auto *parent_merge = f->create_basic_block();
    auto *outer_merge = f->create_basic_block();
    auto constant = [&](uint32_t x) { return module.create_constant(Type::of<uint32_t>(), &x); };
    XIRBuilder b;
    auto bit = [&](uint32_t mask) { return b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_BIT_AND, {input, constant(mask)}); };
    auto condition = [&](uint32_t mask) { return b.call(Type::of<bool>(), ArithmeticOp::BINARY_EQUAL, {bit(mask), constant(0u)}); };
    auto emit = [&](uint32_t value) {
        auto *old = b.load(Type::of<uint32_t>(), output);
        auto *hash = b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_MUL, {old, constant(37u)});
        b.store(output, b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_ADD, {hash, constant(value)}));
    };
    b.set_insertion_point(entry);
    b.store(output, constant(17u));
    auto *outer = b.if_(condition(8u));
    outer->set_true_target(parent);
    outer->set_false_target(alternative);
    outer->set_merge_block(outer_merge);
    b.set_insertion_point(parent);
    emit(11u);
    auto *selection = b.switch_(bit(3u));
    selection->add_case(0u, child);
    selection->add_case(1u, escape);
    selection->set_default_block(parent_merge);
    selection->set_merge_block(parent_merge);
    b.set_insertion_point(child);
    emit(25u);
    auto *nested = b.if_(condition(4u));
    nested->set_true_target(parent_merge);
    nested->set_false_target(child_merge);
    nested->set_merge_block(child_merge);
    b.set_insertion_point(child_merge);
    emit(33u);
    b.br(parent_merge);
    b.set_insertion_point(escape);
    emit(22u);
    b.br(outer_merge);
    b.set_insertion_point(alternative);
    emit(66u);
    b.br(outer_merge);
    b.set_insertion_point(parent_merge);
    emit(44u);
    b.br(outer_merge);
    b.set_insertion_point(outer_merge);
    emit(55u);
    b.return_void();
    // The child's direct exit reaches its enclosing switch merge while the
    // switch also escapes to the outer merge. Treating a child as just its own
    // merge loses one side of the parent's executable exit cut.
    check_execution(module, f, input, output, mode, "nested-exit-cut", 32u, [](uint32_t argument, auto const &expected) {
        ExecutionResult oracle{17u, {17u}};
        auto emit = [&](uint32_t value) {
            oracle.value = oracle.value * 37u + value;
            oracle.stores.emplace_back(oracle.value);
        };
        if ((argument & 8u) == 0u) {
            emit(11u);
            switch (argument & 3u) {
                case 0u:
                    emit(25u);
                    if ((argument & 4u) != 0u) { emit(33u); }
                    emit(44u);
                    break;
                case 1u: emit(22u); break;
                default: emit(44u); break;
            }
        } else {
            emit(66u);
        }
        emit(55u);
        expect(expected == std::optional{oracle}) << "nested exit cut oracle input=" << argument;
    });
}

void check_nested_loop_exit_to_outer_header(RestructureCFGMutationMode mode, uint32_t rotation) {
    Module module;
    auto *f = module.create_kernel();
    auto *entry = f->create_body_block();
    auto *input = f->create_value_argument(Type::of<uint32_t>());
    auto *output = f->create_reference_argument(Type::of<uint32_t>());
    std::array<BasicBlock *, 4u> blocks;
    for (auto i = 0u; i < blocks.size(); ++i) {
        blocks[(i + rotation) % blocks.size()] = f->create_basic_block();
    }
    auto *outer = blocks[0];
    auto *inner = blocks[1];
    auto *latch = blocks[2];
    auto *exit = blocks[3];
    auto constant = [&](uint32_t value) { return module.create_constant(Type::of<uint32_t>(), &value); };
    XIRBuilder b;
    auto emit = [&](Value *tag) {
        auto *old = b.load(Type::of<uint32_t>(), output);
        auto *hash = b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_MUL, {old, constant(37u)});
        auto *value = b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_ADD, {hash, tag});
        b.store(output, value);
    };
    b.set_insertion_point(entry);
    auto *counter = b.alloca_local(Type::of<uint32_t>());
    auto *limit_bits = b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_BIT_AND, {input, constant(7u)});
    auto *limit = b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_ADD, {limit_bits, constant(4u)});
    b.store(counter, constant(0u));
    b.store(output, constant(17u));
    b.br(outer);
    b.set_insertion_point(outer);
    emit(constant(0xd1u));
    b.br(inner);
    b.set_insertion_point(inner);
    auto *previous = b.load(Type::of<uint32_t>(), counter);
    auto *iteration = b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_ADD, {previous, constant(1u)});
    b.store(counter, iteration);
    auto *inner_tag = b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_ADD, {iteration, constant(0xe0u)});
    emit(inner_tag);
    auto *phase = b.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_BIT_AND, {iteration, constant(3u)});
    auto *leave_inner = b.call(Type::of<bool>(), ArithmeticOp::BINARY_EQUAL, {phase, constant(1u)});
    b.cond_br(leave_inner, outer, latch);
    b.set_insertion_point(latch);
    emit(constant(0xa1u));
    auto *again = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {iteration, limit});
    b.cond_br(again, inner, exit);
    b.set_insertion_point(exit);
    emit(constant(0xf1u));
    b.return_void();

    // D -> E, E -> D or M, M -> E or exit. E dominates its inner
    // latch M, but does not dominate the outer header D. The E -> D exit
    // therefore leaves this inner loop even though D can later reach E again.
    // A reachability search must check its starting block against that same
    // dominance boundary before following successors. Both backedges execute:
    // iteration 1 takes E -> D and iteration 2 takes M -> E for every input.
    auto name = std::string{"nested-loop-exit-to-outer-header-"} + std::to_string(rotation);
    check_execution(module, f, input, output, mode, name, 8u, [](uint32_t argument, auto const &expected) {
        ExecutionResult oracle{17u, {17u}};
        auto emit = [&](uint32_t tag) {
            oracle.value = oracle.value * 37u + tag;
            oracle.stores.emplace_back(oracle.value);
        };
        auto limit = (argument & 7u) + 4u;
        emit(0xd1u);
        for (auto iteration = 1u;; ++iteration) {
            emit(0xe0u + iteration);
            if ((iteration & 3u) == 1u) {
                emit(0xd1u);
                continue;
            }
            emit(0xa1u);
            if (iteration >= limit) { break; }
        }
        emit(0xf1u);
        expect(expected == std::optional{oracle}) << "nested loop outer exit oracle input=" << argument;
    });
}

}// namespace
int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    "nested_loop_exit_to_outer_header_is_not_an_inner_reentry"_test = [] {
        for (auto mode : {RestructureCFGMutationMode::TRANSACTIONAL, RestructureCFGMutationMode::IN_PLACE_DISCARDABLE}) {
            for (auto rotation : {0u, 1u, 3u}) { check_nested_loop_exit_to_outer_header(mode, rotation); }
        }
    };
    "reentry_scc_preserves_cross_block_values_and_dynamic_addresses"_test = [] {
        for (auto mode : {RestructureCFGMutationMode::TRANSACTIONAL, RestructureCFGMutationMode::IN_PLACE_DISCARDABLE}) {
            for (auto seed : {1u, 7u, 31u}) { check_graph(seed, true, mode, true); }
        }
    };
    "crossing_loop_epochs_preserve_bounded_execution"_test = [] {
        for (auto mode : {RestructureCFGMutationMode::TRANSACTIONAL, RestructureCFGMutationMode::IN_PLACE_DISCARDABLE}) {
            for (auto rotation : {0u, 3u, 7u}) { check_crossing_loop_epoch_graph(mode, rotation); }
        }
    };
    "terminal_payload_preserves_arm_ownership_and_store_order"_test = [] {
        for (auto mode : {RestructureCFGMutationMode::TRANSACTIONAL, RestructureCFGMutationMode::IN_PLACE_DISCARDABLE}) {
            check_terminal_payload_graph(mode, false);
            check_terminal_payload_graph(mode, true);
        }
    };
    "nested_selection_exit_cut_includes_child_side_exits"_test = [] {
        for (auto mode : {RestructureCFGMutationMode::TRANSACTIONAL, RestructureCFGMutationMode::IN_PLACE_DISCARDABLE}) {
            check_nested_exit_cut_graph(mode);
        }
    };
    "cloned_selection_keeps_external_merge_ownership_unique"_test = [] {
        for (auto mode : {RestructureCFGMutationMode::TRANSACTIONAL, RestructureCFGMutationMode::IN_PLACE_DISCARDABLE}) {
            check_cloned_merge_graph(mode);
        }
    };
    "cloned_region_transports_scalar_and_local_storage_to_frontier"_test = [] {
        for (auto mode : {RestructureCFGMutationMode::TRANSACTIONAL, RestructureCFGMutationMode::IN_PLACE_DISCARDABLE}) {
            check_cloned_frontier_graph(mode, false);
            check_cloned_frontier_graph(mode, true);
        }
    };
    "nested_loops_preserve_multi_exit_early_return_and_local_storage"_test = [] {
        for (auto mode : {RestructureCFGMutationMode::TRANSACTIONAL, RestructureCFGMutationMode::IN_PLACE_DISCARDABLE}) {
            for (auto seed : {0u, 1u, 7u, 19u}) { check_nested_loop_graph(seed, mode); }
        }
    };
    "generated_reducible_cfg_preserves_scalar_execution"_test = [] {
        for (auto mode : {RestructureCFGMutationMode::TRANSACTIONAL, RestructureCFGMutationMode::IN_PLACE_DISCARDABLE}) {
            for (auto cyclic : {false, true}) {
                if (auto selected = environment_uint("LUISA_XIR_CFG_TEST_SEED", UINT32_MAX)) {
                    check_graph(*selected, cyclic, mode);
                } else {
                    for (auto seed = 0u; seed < 64u; ++seed) { check_graph(seed, cyclic, mode); }
                }
            }
        }
    };
}
