#include <algorithm>
#include <array>
#include <cctype>
#include <limits>

#include <luisa/ast/type.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/instructions/coro/id.h>
#include <luisa/xir/instructions/coro/register.h>
#include <luisa/xir/instructions/coro/suspend.h>
#include <luisa/xir/instructions/coro/token.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/print.h>
#include <luisa/xir/instructions/ray_query.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/instructions/unreachable.h>
#include <luisa/xir/metadata/comment.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/Canonicalize_Control_Flow.h>
#include <luisa/xir/passes/coro/materialize_coro.h>
#include <luisa/xir/special_register.h>
#include <luisa/xir/undefined.h>

namespace luisa::compute::xir {

namespace detail {

[[nodiscard]] static bool is_void_type(const Type *type) noexcept {
    return type == nullptr;
}

[[nodiscard]] static luisa::optional<luisa::string> parse_suspend_name(luisa::string_view comment) noexcept {
    constexpr luisa::string_view prefix = "CoroSplitMark(";
    if (!comment.starts_with(prefix) || comment.back() != ')') { return {}; }
    auto body = comment.substr(prefix.size(), comment.size() - prefix.size() - 1u);
    if (body.empty()) { return {}; }
    return luisa::string{body};
}

[[nodiscard]] static luisa::unordered_map<luisa::string, uint32_t>
collect_named_tokens(const CallableFunction *function) noexcept {
    luisa::unordered_map<luisa::string, uint32_t> result;
    luisa::optional<luisa::string> pending;
    if (auto definition = const_cast<CallableFunction *>(function)->definition()) {
        definition->traverse_basic_blocks([&](const BasicBlock *block) noexcept {
            block->traverse_instructions([&](const Instruction *inst) noexcept {
                for (auto md : inst->metadata_list()) {
                    if (auto comment = md->isa<CommentMD>() ? static_cast<const CommentMD *>(md) : nullptr) {
                        if (auto parsed = parse_suspend_name(comment->comment())) { pending = std::move(parsed); }
                    }
                }
                if (auto suspend = inst->isa<SuspendInst>() ? static_cast<const SuspendInst *>(inst) : nullptr) {
                    if (pending.has_value()) {
                        result.emplace(*pending, suspend->coro_token);
                        pending.reset();
                    }
                }
            });
        });
    }
    return result;
}

struct AccessPathKey {
    Value *root{nullptr};
    luisa::vector<int32_t> chain;
};

[[nodiscard]] static bool access_chain_is_prefix(luisa::span<const int32_t> prefix,
                                                 luisa::span<const int32_t> chain) noexcept {
    if (prefix.size() > chain.size()) { return false; }
    for (auto i = 0u; i < prefix.size(); i++) {
        if (prefix[i] != chain[i]) { return false; }
    }
    return true;
}

class AccessSet {

private:
    luisa::vector<AccessPathKey> _paths;

private:
    [[nodiscard]] static luisa::vector<int32_t> normalize_chain(luisa::span<const int32_t> chain) noexcept {
        luisa::vector<int32_t> result;
        result.reserve(chain.size());
        for (auto index : chain) {
            // Match the reference AccessTree behavior: a dynamic index
            // aliases the whole remaining subtree, so we collapse the
            // access chain at the dynamic component instead of keeping a
            // sentinel element in the key.
            if (index < 0) { break; }
            result.emplace_back(index);
        }
        return result;
    }

    [[nodiscard]] static const Type *child_type(const Type *type, uint32_t index) noexcept {
        if (type == nullptr) { return nullptr; }
        switch (type->tag()) {
            case Type::Tag::VECTOR:
            case Type::Tag::ARRAY:
                LUISA_ASSERT(index < type->dimension(),
                             "Access chain index {} out of bounds for '{}'.",
                             index, type->description());
                return type->element();
            case Type::Tag::MATRIX:
                LUISA_ASSERT(index < type->dimension(),
                             "Access chain index {} out of bounds for '{}'.",
                             index, type->description());
                return Type::vector(type->element(), type->dimension());
            case Type::Tag::STRUCTURE: {
                auto members = type->members();
                LUISA_ASSERT(index < members.size(),
                             "Access chain index {} out of bounds for '{}'.",
                             index, type->description());
                return members[index];
            }
            default: return nullptr;
        }
    }

    [[nodiscard]] static uint32_t child_count(const Type *type) noexcept {
        if (type == nullptr) { return 0u; }
        switch (type->tag()) {
            case Type::Tag::VECTOR:
            case Type::Tag::ARRAY:
            case Type::Tag::MATRIX: return type->dimension();
            case Type::Tag::STRUCTURE: return static_cast<uint32_t>(type->members().size());
            default: return 0u;
        }
    }

    [[nodiscard]] static const Type *type_at_path(const Type *root_type,
                                                  luisa::span<const int32_t> chain) noexcept {
        auto type = root_type;
        for (auto index : chain) {
            if (index < 0) { break; }
            type = child_type(type, static_cast<uint32_t>(index));
            if (type == nullptr) { break; }
        }
        return type;
    }

public:
    AccessSet() noexcept = default;
    explicit AccessSet(luisa::span<const CoroAccessPath> paths) noexcept {
        for (auto &&path : paths) { insert(path.root, path.chain); }
    }

    [[nodiscard]] bool contains(Value *root, luisa::span<const int32_t> chain) const noexcept {
        auto normalized = normalize_chain(chain);
        return std::any_of(_paths.cbegin(), _paths.cend(), [root, &normalized](auto &&path) noexcept {
            return path.root == root && access_chain_is_prefix(path.chain, normalized);
        });
    }

    void insert(Value *root, luisa::span<const int32_t> chain) noexcept {
        if (root == nullptr) { return; }
        auto normalized = normalize_chain(chain);
        if (contains(root, normalized)) { return; }
        _paths.erase(std::remove_if(_paths.begin(), _paths.end(), [root, &normalized](auto &&path) noexcept {
            return path.root == root && access_chain_is_prefix(normalized, path.chain);
        }), _paths.end());
        _paths.emplace_back(AccessPathKey{root, std::move(normalized)});
    }

    void union_with(const AccessSet &other) noexcept {
        for (auto &&path : other._paths) { insert(path.root, path.chain); }
    }

    bool coalesce_whole_access_chains() noexcept {
        auto any_change = false;
        for (;;) {
            auto changed = false;
            auto snapshot = _paths;
            luisa::vector<AccessPathKey> candidates;
            candidates.reserve(snapshot.size() * 2u + 1u);
            for (auto &&path : snapshot) {
                if (path.root == nullptr || path.root->type() == nullptr) { continue; }
                candidates.emplace_back(AccessPathKey{path.root, {}});
                for (auto i = 0u; i < path.chain.size(); i++) {
                    candidates.emplace_back(AccessPathKey{
                        path.root,
                        luisa::vector<int32_t>{path.chain.begin(), path.chain.begin() + i + 1u}});
                }
            }
            std::sort(candidates.begin(), candidates.end(), [](auto &&lhs, auto &&rhs) noexcept {
                if (lhs.root != rhs.root) { return lhs.root < rhs.root; }
                if (lhs.chain.size() != rhs.chain.size()) { return lhs.chain.size() > rhs.chain.size(); }
                return std::lexicographical_compare(lhs.chain.begin(), lhs.chain.end(),
                                                    rhs.chain.begin(), rhs.chain.end());
            });
            candidates.erase(std::unique(candidates.begin(), candidates.end(),
                                         [](auto &&lhs, auto &&rhs) noexcept {
                                             return lhs.root == rhs.root && lhs.chain == rhs.chain;
                                         }),
                             candidates.end());
            for (auto &&candidate : candidates) {
                auto type = type_at_path(candidate.root->type(), candidate.chain);
                auto count = child_count(type);
                if (count == 0u || contains(candidate.root, candidate.chain)) { continue; }
                auto covered = true;
                auto child = candidate.chain;
                child.reserve(candidate.chain.size() + 1u);
                for (auto i = 0u; i < count; i++) {
                    child.resize(candidate.chain.size());
                    child.emplace_back(static_cast<int32_t>(i));
                    if (!contains(candidate.root, child)) {
                        covered = false;
                        break;
                    }
                }
                if (covered) {
                    insert(candidate.root, candidate.chain);
                    changed = true;
                }
            }
            if (!changed) { break; }
            any_change = true;
        }
        return any_change;
    }

    void dynamic_access_chains_as_whole() noexcept {
        // Dynamic indices are normalized by truncating the access chain at the
        // first dynamic component, so they already behave as "whole subtree"
        // accesses like the reference AccessTree postprocess.
    }

    [[nodiscard]] static AccessSet union_of(const AccessSet &lhs, const AccessSet &rhs) noexcept {
        auto result = lhs;
        result.union_with(rhs);
        return result;
    }

    [[nodiscard]] static AccessSet intersect(const AccessSet &lhs, const AccessSet &rhs) noexcept {
        AccessSet result;
        for (auto &&path : lhs._paths) {
            if (rhs.contains(path.root, path.chain)) { result.insert(path.root, path.chain); }
        }
        for (auto &&path : rhs._paths) {
            if (lhs.contains(path.root, path.chain)) { result.insert(path.root, path.chain); }
        }
        return result;
    }

    [[nodiscard]] static AccessSet subtract(const AccessSet &lhs, const AccessSet &rhs) noexcept {
        AccessSet result;
        for (auto &&path : lhs._paths) {
            if (!rhs.contains(path.root, path.chain)) { result.insert(path.root, path.chain); }
        }
        return result;
    }

    [[nodiscard]] luisa::vector<CoroAccessPath> to_public() const noexcept {
        auto sorted = _paths;
        std::sort(sorted.begin(), sorted.end(), [](auto &&lhs, auto &&rhs) noexcept {
            if (lhs.root != rhs.root) { return lhs.root < rhs.root; }
            if (lhs.chain.size() != rhs.chain.size()) { return lhs.chain.size() < rhs.chain.size(); }
            return std::lexicographical_compare(lhs.chain.begin(), lhs.chain.end(),
                                                rhs.chain.begin(), rhs.chain.end());
        });
        luisa::vector<CoroAccessPath> result;
        result.reserve(sorted.size());
        for (auto &&path : sorted) { result.emplace_back(CoroAccessPath{path.root, path.chain}); }
        return result;
    }

    [[nodiscard]] bool operator==(const AccessSet &rhs) const noexcept {
        auto lhs_paths = to_public();
        auto rhs_paths = rhs.to_public();
        if (lhs_paths.size() != rhs_paths.size()) { return false; }
        for (auto i = 0u; i < lhs_paths.size(); i++) {
            if (lhs_paths[i].root != rhs_paths[i].root || lhs_paths[i].chain != rhs_paths[i].chain) {
                return false;
            }
        }
        return true;
    }
};

class ReplayableValueAnalysis {

private:
    luisa::unordered_map<const Value *, bool> _cache;

private:
    [[nodiscard]] bool _is_replayable_local_access(const Value *value) noexcept {
        if (value == nullptr) { return false; }
        if (value->isa<AllocaInst>()) { return true; }
        if (auto gep = value->isa<GEPInst>() ? static_cast<const GEPInst *>(value) : nullptr) {
            if (!_is_replayable_local_access(gep->base())) { return false; }
            for (auto i = 0u; i < gep->index_count(); i++) {
                if (!detect(gep->index(i))) { return false; }
            }
            return true;
        }
        return false;
    }

    [[nodiscard]] bool _compute(const Value *value) noexcept {
        if (value == nullptr) { return true; }
        if (value->isa<Constant>() || value->isa<Undefined>() || value->isa<SpecialRegister>()) { return true; }
        if (auto arg = value->isa<Argument>() ? static_cast<const Argument *>(value) : nullptr) {
            return arg->is_value() || arg->is_resource();
        }
        if (!value->isa<Instruction>()) { return false; }
        auto inst = static_cast<const Instruction *>(value);
        switch (inst->derived_instruction_tag()) {
            case DerivedInstructionTag::CORO_ID:
            case DerivedInstructionTag::CORO_TOKEN: return true;
            case DerivedInstructionTag::LOAD:
                return _is_replayable_local_access(static_cast<const LoadInst *>(inst)->variable());
            case DerivedInstructionTag::ARITHMETIC:
            case DerivedInstructionTag::CAST:
            case DerivedInstructionTag::RESOURCE_QUERY:
                return std::all_of(inst->operand_uses().cbegin(), inst->operand_uses().cend(),
                                   [&](auto use) noexcept { return detect(use->value()); });
            default: return false;
        }
    }

public:
    [[nodiscard]] bool detect(const Value *value) noexcept {
        if (auto iter = _cache.find(value); iter != _cache.end()) { return iter->second; }
        auto replayable = _compute(value);
        _cache.emplace(value, replayable);
        return replayable;
    }
};

static void validate_coroutine_function(CallableFunction *function) noexcept {
    LUISA_ASSERT(function != nullptr, "Coroutine materialization requires a valid callable.");
    LUISA_ASSERT(function->definition() != nullptr, "External functions cannot be materialized as coroutines.");
    LUISA_ASSERT(is_void_type(function->type()), "Only void coroutines are supported in the XIR materializer.");
}

struct LoopContext {
    const BasicBlock *continue_target{nullptr};
    const BasicBlock *break_target{nullptr};
};

struct CoroGraphIndexer {
    CoroInstructionRef parent{invalid_coro_instruction_ref};
    uint32_t parent_branch{0u};
    uint32_t index_in_parent_branch{0u};
};

struct PreliminaryGraph {
    luisa::vector<CoroInstruction> instructions;
    luisa::unordered_map<const Instruction *, CoroInstructionRef> instruction_to_ref;
    CoroInstructionRef entry_scope{invalid_coro_instruction_ref};
    luisa::unordered_set<CoroInstructionRef> terminators;

    [[nodiscard]] const CoroInstruction &instr(CoroInstructionRef ref) const noexcept { return instructions.at(ref); }
    [[nodiscard]] CoroInstruction &instr(CoroInstructionRef ref) noexcept { return instructions.at(ref); }
    [[nodiscard]] CoroInstructionRef add(CoroInstruction inst) noexcept {
        auto ref = static_cast<CoroInstructionRef>(instructions.size());
        instructions.emplace_back(std::move(inst));
        return ref;
    }

    [[nodiscard]] const luisa::vector<CoroInstructionRef> &get_parent_branch(const CoroGraphIndexer &indexer) const noexcept {
        auto &&parent = instr(indexer.parent);
        switch (parent.tag) {
            case CoroInstructionTag::ENTRY_SCOPE: return parent.body;
            case CoroInstructionTag::SIMPLE_LOOP: return parent.body;
            case CoroInstructionTag::IF: return indexer.parent_branch == 0u ? parent.true_branch : parent.false_branch;
            case CoroInstructionTag::SWITCH:
                if (indexer.parent_branch < parent.cases.size()) { return parent.cases[indexer.parent_branch].body; }
                return parent.default_body;
            default: LUISA_ERROR_WITH_LOCATION("Unexpected coroutine parent instruction.");
        }
    }

    [[nodiscard]] CoroInstructionRef get_instr_ref(const CoroGraphIndexer &indexer) const noexcept {
        return get_parent_branch(indexer).at(indexer.index_in_parent_branch);
    }

    [[nodiscard]] const CoroInstruction &get_instr(const CoroGraphIndexer &indexer) const noexcept {
        return instr(get_instr_ref(indexer));
    }

    [[nodiscard]] bool is_terminator(CoroInstructionRef ref) const noexcept { return terminators.contains(ref); }
};

struct PreliminaryTranslator {
    CallableFunction *function;
    PreliminaryGraph graph;

    [[nodiscard]] static Value *resolve_condition_value(const Instruction *terminator, const Value *value) noexcept {
        auto block = terminator != nullptr ? terminator->parent_block() : nullptr;
        if (block == nullptr) { return const_cast<Value *>(value); }
        auto current = value;
        for (auto depth = 0u; depth < 16u; depth++) {
            auto load = current != nullptr && current->isa<LoadInst>() ? static_cast<const LoadInst *>(current) : nullptr;
            if (load == nullptr) { break; }
            const StoreInst *store = nullptr;
            for (auto inst : block->instructions()) {
                if (inst == terminator) { break; }
                auto maybe_store = inst->isa<StoreInst>() ? static_cast<const StoreInst *>(inst) : nullptr;
                if (maybe_store != nullptr && maybe_store->variable() == load->variable()) {
                    store = maybe_store;
                }
            }
            if (store == nullptr || store->value() == current) { break; }
            current = store->value();
        }
        return const_cast<Value *>(current);
    }

    [[nodiscard]] CoroInstructionRef register_instruction(const Instruction *inst, CoroInstruction coro_inst) noexcept {
        if (auto iter = graph.instruction_to_ref.find(inst); iter != graph.instruction_to_ref.end()) { return iter->second; }
        auto ref = graph.add(std::move(coro_inst));
        graph.instruction_to_ref.emplace(inst, ref);
        return ref;
    }

    [[nodiscard]] CoroInstructionRef translate_linear_instruction(const Instruction *inst) noexcept;
    [[nodiscard]] CoroInstructionRef translate_if(const IfInst *inst, const LoopContext *loop_ctx) noexcept;
    [[nodiscard]] CoroInstructionRef translate_switch(const SwitchInst *inst, const LoopContext *loop_ctx) noexcept;
    [[nodiscard]] CoroInstructionRef translate_simple_loop(const SimpleLoopInst *inst) noexcept;
    [[nodiscard]] CoroInstructionRef translate_conditional_branch(const ConditionalBranchInst *inst,
                                                                  const BasicBlock *follow,
                                                                  const LoopContext *loop_ctx) noexcept;
    [[nodiscard]] luisa::vector<CoroInstructionRef> translate_path(const BasicBlock *block,
                                                                   const BasicBlock *follow,
                                                                   const LoopContext *loop_ctx,
                                                                   luisa::unordered_set<const BasicBlock *> visited = {}) noexcept;
    [[nodiscard]] PreliminaryGraph build() noexcept;
};

static void check_duplicate_suspend_tokens(CallableFunction *function) noexcept {
    luisa::vector<uint32_t> tokens;
    function->definition()->traverse_instructions([&](Instruction *inst) noexcept {
        if (auto suspend = inst->isa<SuspendInst>() ? static_cast<SuspendInst *>(inst) : nullptr) {
            tokens.emplace_back(suspend->coro_token);
        }
    });
    std::sort(tokens.begin(), tokens.end());
    LUISA_ASSERT(std::adjacent_find(tokens.begin(), tokens.end()) == tokens.end(),
                 "Duplicate suspend tokens in coroutine.");
}

[[nodiscard]] static bool find_terminators(const PreliminaryGraph &graph,
                                           CoroInstructionRef ref,
                                           luisa::unordered_map<CoroInstructionRef, bool> &known) noexcept {
    if (auto iter = known.find(ref); iter != known.end()) { return iter->second; }
    auto &&instr = graph.instr(ref);
    auto any_terminator = [&graph, &known](const luisa::vector<CoroInstructionRef> &body) noexcept {
        auto terminated = false;
        for (auto child : body) { terminated = find_terminators(graph, child, known) || terminated; }
        return terminated;
    };
    auto all_terminated = [&graph, &known](const luisa::vector<CoroInstructionRef> &body) noexcept {
        if (body.empty()) { return false; }
        for (auto child : body) {
            if (!find_terminators(graph, child, known)) { return false; }
        }
        return true;
    };
    auto result = false;
    switch (instr.tag) {
        case CoroInstructionTag::ENTRY:
        case CoroInstructionTag::SUSPEND:
        case CoroInstructionTag::TERMINATE:
        case CoroInstructionTag::LOOP_CONTINUE:
        case CoroInstructionTag::LOOP_BREAK:
            result = true;
            break;
        case CoroInstructionTag::ENTRY_SCOPE:
            LUISA_ASSERT(any_terminator(instr.body), "Coroutine entry must eventually terminate.");
            result = true;
            break;
        case CoroInstructionTag::SIMPLE:
            result = instr.simple != nullptr && instr.simple->isa<UnreachableInst>();
            break;
        case CoroInstructionTag::SIMPLE_LOOP:
            result = any_terminator(instr.body);
            break;
        case CoroInstructionTag::IF:
            result = all_terminated(instr.true_branch) && all_terminated(instr.false_branch);
            break;
        case CoroInstructionTag::SWITCH: {
            auto cases_terminated = true;
            for (auto &&c : instr.cases) { cases_terminated = all_terminated(c.body) && cases_terminated; }
            result = cases_terminated && all_terminated(instr.default_body);
            break;
        }
        default:
            result = false;
            break;
    }
    known.emplace(ref, result);
    return result;
}

[[nodiscard]] CoroInstructionRef
PreliminaryTranslator::translate_linear_instruction(const Instruction *inst) noexcept {
    if (auto iter = graph.instruction_to_ref.find(inst); iter != graph.instruction_to_ref.end()) { return iter->second; }
    if (auto suspend = inst->isa<SuspendInst>() ? static_cast<const SuspendInst *>(inst) : nullptr) {
        return register_instruction(inst, CoroInstruction{
                                        .tag = CoroInstructionTag::SUSPEND,
                                        .token = suspend->coro_token});
    }
    if (inst->isa<ReturnInst>()) {
        return register_instruction(inst, CoroInstruction{.tag = CoroInstructionTag::TERMINATE});
    }
    return register_instruction(inst, CoroInstruction{
                                         .tag = CoroInstructionTag::SIMPLE,
                                         .simple = const_cast<Instruction *>(inst)});
}

[[nodiscard]] CoroInstructionRef
PreliminaryTranslator::translate_if(const IfInst *inst, const LoopContext *loop_ctx) noexcept {
    if (auto iter = graph.instruction_to_ref.find(inst); iter != graph.instruction_to_ref.end()) { return iter->second; }
    auto ref = graph.add(CoroInstruction{.tag = CoroInstructionTag::IF});
    graph.instruction_to_ref.emplace(inst, ref);
    auto condition = resolve_condition_value(inst, inst->condition());
    auto true_branch = translate_path(inst->true_block(), inst->merge_block(), loop_ctx);
    auto false_branch = translate_path(inst->false_block(), inst->merge_block(), loop_ctx);
    auto &coro = graph.instr(ref);
    coro.condition = condition;
    coro.true_branch = std::move(true_branch);
    coro.false_branch = std::move(false_branch);
    return ref;
}

[[nodiscard]] CoroInstructionRef
PreliminaryTranslator::translate_switch(const SwitchInst *inst, const LoopContext *loop_ctx) noexcept {
    if (auto iter = graph.instruction_to_ref.find(inst); iter != graph.instruction_to_ref.end()) { return iter->second; }
    auto ref = graph.add(CoroInstruction{.tag = CoroInstructionTag::SWITCH});
    graph.instruction_to_ref.emplace(inst, ref);
    auto condition = resolve_condition_value(inst, inst->value());
    luisa::vector<CoroSwitchCase> cases;
    cases.reserve(inst->case_count());
    for (auto i = 0u; i < inst->case_count(); i++) {
        cases.emplace_back(CoroSwitchCase{
            .value = inst->case_value(i),
            .body = translate_path(inst->case_block(i), inst->merge_block(), loop_ctx)});
    }
    auto default_body = translate_path(inst->default_block(), inst->merge_block(), loop_ctx);
    auto &coro = graph.instr(ref);
    coro.condition = condition;
    coro.cases = std::move(cases);
    coro.default_body = std::move(default_body);
    return ref;
}

[[nodiscard]] CoroInstructionRef
PreliminaryTranslator::translate_simple_loop(const SimpleLoopInst *inst) noexcept {
    if (auto iter = graph.instruction_to_ref.find(inst); iter != graph.instruction_to_ref.end()) { return iter->second; }
    auto ref = graph.add(CoroInstruction{.tag = CoroInstructionTag::SIMPLE_LOOP});
    graph.instruction_to_ref.emplace(inst, ref);
    auto loop_ctx = LoopContext{
        .continue_target = inst->body_block(),
        .break_target = inst->merge_block()};
    auto body = translate_path(inst->body_block(), nullptr, &loop_ctx);
    graph.instr(ref).body = std::move(body);
    return ref;
}

[[nodiscard]] CoroInstructionRef
PreliminaryTranslator::translate_conditional_branch(const ConditionalBranchInst *inst,
                                                    const BasicBlock *follow,
                                                    const LoopContext *loop_ctx) noexcept {
    if (auto iter = graph.instruction_to_ref.find(inst); iter != graph.instruction_to_ref.end()) { return iter->second; }
    auto ref = graph.add(CoroInstruction{.tag = CoroInstructionTag::IF});
    graph.instruction_to_ref.emplace(inst, ref);
    auto condition = resolve_condition_value(inst, inst->condition());
    luisa::vector<CoroInstructionRef> true_branch;
    luisa::vector<CoroInstructionRef> false_branch;
    auto make_loop_leaf = [&](const BasicBlock *target, luisa::vector<CoroInstructionRef> &body) noexcept -> bool {
        if (loop_ctx != nullptr && target == loop_ctx->continue_target) {
            body.emplace_back(graph.add(CoroInstruction{.tag = CoroInstructionTag::LOOP_CONTINUE}));
            return true;
        }
        if (loop_ctx != nullptr && target == loop_ctx->break_target) {
            body.emplace_back(graph.add(CoroInstruction{.tag = CoroInstructionTag::LOOP_BREAK}));
            return true;
        }
        return false;
    };
    auto branch_follow = follow;
    if (branch_follow == nullptr && loop_ctx != nullptr) {
        auto true_is_continue = inst->true_block() == loop_ctx->continue_target;
        auto false_is_continue = inst->false_block() == loop_ctx->continue_target;
        auto true_is_break = inst->true_block() == loop_ctx->break_target;
        auto false_is_break = inst->false_block() == loop_ctx->break_target;
        // Canonicalized SimpleLoop exit guards can end in a low-level cond_br where one
        // successor breaks the loop and the other runs the update path before continuing.
        // In that case, use the opposite loop leaf as the synthetic stopping block.
        if (true_is_break != false_is_break) {
            branch_follow = loop_ctx->continue_target;
        } else if (true_is_continue != false_is_continue) {
            branch_follow = loop_ctx->break_target;
        }
    }
    if (!make_loop_leaf(inst->true_block(), true_branch)) {
        LUISA_ASSERT(branch_follow != nullptr,
                     "Low-level conditional branch without a reconvergence target is unsupported in coroutine graph extraction.");
        true_branch = translate_path(inst->true_block(), branch_follow, loop_ctx);
    }
    if (!make_loop_leaf(inst->false_block(), false_branch)) {
        LUISA_ASSERT(branch_follow != nullptr,
                     "Low-level conditional branch without a reconvergence target is unsupported in coroutine graph extraction.");
        false_branch = translate_path(inst->false_block(), branch_follow, loop_ctx);
    }
    auto &coro = graph.instr(ref);
    coro.condition = condition;
    coro.true_branch = std::move(true_branch);
    coro.false_branch = std::move(false_branch);
    return ref;
}

[[nodiscard]] luisa::vector<CoroInstructionRef>
PreliminaryTranslator::translate_path(const BasicBlock *block,
                                      const BasicBlock *follow,
                                      const LoopContext *loop_ctx,
                                      luisa::unordered_set<const BasicBlock *> visited) noexcept {
    luisa::vector<CoroInstructionRef> result;
    while (block != nullptr && block != follow) {
        LUISA_ASSERT(visited.emplace(block).second,
                     "Unsupported non-canonical cycle while extracting coroutine graph.");
        LUISA_ASSERT(block->is_terminated(), "Coroutine graph extraction requires terminated basic blocks.");
        auto terminator = block->terminator();
        for (auto inst : block->instructions()) {
            if (inst == terminator) { break; }
            result.emplace_back(translate_linear_instruction(inst));
        }
        switch (terminator->derived_instruction_tag()) {
            case DerivedInstructionTag::IF:
                result.emplace_back(translate_if(static_cast<const IfInst *>(terminator), loop_ctx));
                block = static_cast<const IfInst *>(terminator)->merge_block();
                break;
            case DerivedInstructionTag::SWITCH:
                result.emplace_back(translate_switch(static_cast<const SwitchInst *>(terminator), loop_ctx));
                block = static_cast<const SwitchInst *>(terminator)->merge_block();
                break;
            case DerivedInstructionTag::SIMPLE_LOOP:
                result.emplace_back(translate_simple_loop(static_cast<const SimpleLoopInst *>(terminator)));
                block = static_cast<const SimpleLoopInst *>(terminator)->merge_block();
                break;
            case DerivedInstructionTag::BRANCH: {
                auto branch = static_cast<const BranchInst *>(terminator);
                if (loop_ctx != nullptr && branch->target_block() == loop_ctx->continue_target) {
                    result.emplace_back(register_instruction(terminator, CoroInstruction{.tag = CoroInstructionTag::LOOP_CONTINUE}));
                    return result;
                }
                if (loop_ctx != nullptr && branch->target_block() == loop_ctx->break_target) {
                    result.emplace_back(register_instruction(terminator, CoroInstruction{.tag = CoroInstructionTag::LOOP_BREAK}));
                    return result;
                }
                if (branch->target_block() == follow) { return result; }
                block = branch->target_block();
                break;
            }
            case DerivedInstructionTag::CONDITIONAL_BRANCH: {
                auto branch = static_cast<const ConditionalBranchInst *>(terminator);
                result.emplace_back(translate_conditional_branch(branch, follow, loop_ctx));
                if (loop_ctx != nullptr &&
                    ((branch->true_block() == loop_ctx->continue_target && branch->false_block() == loop_ctx->break_target) ||
                     (branch->false_block() == loop_ctx->continue_target && branch->true_block() == loop_ctx->break_target))) {
                    return result;
                }
                block = follow;
                break;
            }
            case DerivedInstructionTag::RETURN:
            case DerivedInstructionTag::UNREACHABLE:
                result.emplace_back(translate_linear_instruction(terminator));
                return result;
            case DerivedInstructionTag::LOOP:
            case DerivedInstructionTag::BREAK:
            case DerivedInstructionTag::CONTINUE:
                LUISA_ERROR_WITH_LOCATION("Coroutine graph extraction only accepts canonicalized SimpleLoop control flow.");
            default:
                LUISA_ERROR_WITH_LOCATION("Unsupported terminator '{}' in coroutine graph extraction.",
                                          xir::to_string(terminator->derived_instruction_tag()));
        }
    }
    return result;
}

[[nodiscard]] PreliminaryGraph PreliminaryTranslator::build() noexcept {
    validate_coroutine_function(function);
    check_duplicate_suspend_tokens(function);
    auto entry = graph.add(CoroInstruction{.tag = CoroInstructionTag::ENTRY});
    auto body = translate_path(function->body_block(), nullptr, nullptr);
    body.insert(body.begin(), entry);
    graph.entry_scope = graph.add(CoroInstruction{
        .tag = CoroInstructionTag::ENTRY_SCOPE,
        .body = std::move(body)});
    luisa::unordered_map<CoroInstructionRef, bool> known_terminators;
    find_terminators(graph, graph.entry_scope, known_terminators);
    for (auto &&[ref, is_terminator] : known_terminators) {
        if (is_terminator) { graph.terminators.emplace(ref); }
    }
    return std::move(graph);
}

struct GraphBuilder {

    enum class TerminationKind : uint32_t {
        NONE,
        LOOP_LEAF,
        HARD
    };

    static void replay_condition_stack(CoroGraph &graph,
                                       const PreliminaryGraph &preliminary,
                                       const luisa::vector<CoroGraphIndexer> &ancestors,
                                       CoroScope &scope) noexcept {
        luisa::vector<CoroConditionStackItem> items;
        for (auto &&ancestor : ancestors) {
            auto &&instr = preliminary.instr(ancestor.parent);
            switch (instr.tag) {
                case CoroInstructionTag::IF:
                    items.emplace_back(CoroConditionStackItem{
                        .value = instr.condition,
                        .selected_value = ancestor.parent_branch == 0u ? 1 : 0});
                    break;
                case CoroInstructionTag::SWITCH:
                    if (ancestor.parent_branch < instr.cases.size()) {
                        items.emplace_back(CoroConditionStackItem{
                            .value = instr.condition,
                            .selected_value = instr.cases[ancestor.parent_branch].value});
                    }
                    break;
                default: break;
            }
        }
        if (!items.empty()) {
            scope.instructions.emplace_back(static_cast<CoroInstructionRef>(graph.instructions.size()));
            graph.instructions.emplace_back(CoroInstruction{
                .tag = CoroInstructionTag::CONDITION_STACK_REPLAY,
                .condition_stack = std::move(items)});
        }
    }

    [[nodiscard]] static CoroInstructionRef clone_instruction(CoroGraph &graph,
                                                              const PreliminaryGraph &preliminary,
                                                              CoroInstructionRef ref) noexcept {
        auto &&instr = preliminary.instr(ref);
        auto clone_block = [&](const luisa::vector<CoroInstructionRef> &body) noexcept {
            luisa::vector<CoroInstructionRef> result;
            for (auto child : body) {
                result.emplace_back(clone_instruction(graph, preliminary, child));
                if (preliminary.is_terminator(child)) { break; }
            }
            return result;
        };
        switch (instr.tag) {
            case CoroInstructionTag::SIMPLE:
            case CoroInstructionTag::SUSPEND:
            case CoroInstructionTag::TERMINATE:
            case CoroInstructionTag::LOOP_CONTINUE:
            case CoroInstructionTag::LOOP_BREAK:
                return ref;
            case CoroInstructionTag::IF: {
                auto cloned_true_branch = clone_block(instr.true_branch);
                auto cloned_false_branch = clone_block(instr.false_branch);
                auto ref_new = static_cast<CoroInstructionRef>(graph.instructions.size());
                graph.instructions.emplace_back(CoroInstruction{
                    .tag = CoroInstructionTag::IF,
                    .condition = instr.condition,
                    .true_branch = std::move(cloned_true_branch),
                    .false_branch = std::move(cloned_false_branch)});
                return ref_new;
            }
            case CoroInstructionTag::SWITCH: {
                luisa::vector<CoroSwitchCase> cases;
                cases.reserve(instr.cases.size());
                for (auto &&c : instr.cases) { cases.emplace_back(CoroSwitchCase{.value = c.value, .body = clone_block(c.body)}); }
                auto default_body = clone_block(instr.default_body);
                auto ref_new = static_cast<CoroInstructionRef>(graph.instructions.size());
                graph.instructions.emplace_back(CoroInstruction{
                    .tag = CoroInstructionTag::SWITCH,
                    .condition = instr.condition,
                    .cases = std::move(cases),
                    .default_body = std::move(default_body)});
                return ref_new;
            }
            case CoroInstructionTag::SIMPLE_LOOP: {
                auto body = clone_block(instr.body);
                auto ref_new = static_cast<CoroInstructionRef>(graph.instructions.size());
                graph.instructions.emplace_back(CoroInstruction{
                    .tag = CoroInstructionTag::SIMPLE_LOOP,
                    .body = std::move(body)});
                return ref_new;
            }
            default:
                LUISA_ERROR_WITH_LOCATION("Unexpected instruction while cloning coroutine scope.");
        }
    }

    static TerminationKind remove_unreachable_from_block(CoroGraph &graph,
                                                         luisa::vector<CoroInstructionRef> &block) noexcept {
        for (auto i = 0u; i < block.size(); i++) {
            auto terminated = remove_unreachable_from_instruction(graph, block[i]);
            if (terminated != TerminationKind::NONE) {
                block.erase(block.begin() + i + 1u, block.end());
                if (terminated == TerminationKind::HARD &&
                    !block.empty() &&
                    graph.instructions[block.back()].tag == CoroInstructionTag::SIMPLE_LOOP) {
                    auto body = graph.instructions[block.back()].body;
                    block.pop_back();
                    block.insert(block.end(), body.begin(), body.end());
                }
                return terminated;
            }
        }
        return TerminationKind::NONE;
    }

    static TerminationKind remove_unreachable_from_instruction(CoroGraph &graph, CoroInstructionRef ref) noexcept {
        auto &instr = graph.instructions[ref];
        switch (instr.tag) {
            case CoroInstructionTag::SIMPLE:
                return instr.simple != nullptr && instr.simple->isa<UnreachableInst>() ?
                           TerminationKind::HARD :
                           TerminationKind::NONE;
            case CoroInstructionTag::SIMPLE_LOOP: {
                auto body_terminated = remove_unreachable_from_block(graph, instr.body);
                return body_terminated == TerminationKind::HARD ?
                           TerminationKind::HARD :
                           TerminationKind::NONE;
            }
            case CoroInstructionTag::SKIP_IF_FIRST_FLAG:
                static_cast<void>(remove_unreachable_from_block(graph, instr.body));
                return TerminationKind::NONE;
            case CoroInstructionTag::IF: {
                auto true_terminated = remove_unreachable_from_block(graph, instr.true_branch);
                auto false_terminated = remove_unreachable_from_block(graph, instr.false_branch);
                if (true_terminated == TerminationKind::NONE ||
                    false_terminated == TerminationKind::NONE) {
                    return TerminationKind::NONE;
                }
                return true_terminated == TerminationKind::HARD &&
                               false_terminated == TerminationKind::HARD ?
                           TerminationKind::HARD :
                           TerminationKind::LOOP_LEAF;
            }
            case CoroInstructionTag::SWITCH: {
                auto all_terminated = true;
                auto all_hard_terminated = true;
                for (auto &c : instr.cases) {
                    auto terminated = remove_unreachable_from_block(graph, c.body);
                    all_terminated = all_terminated && terminated != TerminationKind::NONE;
                    all_hard_terminated = all_hard_terminated && terminated == TerminationKind::HARD;
                }
                auto default_terminated = remove_unreachable_from_block(graph, instr.default_body);
                all_terminated = all_terminated && default_terminated != TerminationKind::NONE;
                all_hard_terminated = all_hard_terminated && default_terminated == TerminationKind::HARD;
                if (!all_terminated) { return TerminationKind::NONE; }
                return all_hard_terminated ? TerminationKind::HARD : TerminationKind::LOOP_LEAF;
            }
            case CoroInstructionTag::LOOP_CONTINUE:
            case CoroInstructionTag::LOOP_BREAK:
                return TerminationKind::LOOP_LEAF;
            case CoroInstructionTag::SUSPEND:
            case CoroInstructionTag::TERMINATE:
                return TerminationKind::HARD;
            default: return TerminationKind::NONE;
        }
    }

    static void collect_designated_values(const CoroGraph &graph,
                                          const luisa::vector<CoroInstructionRef> &instructions,
                                          luisa::unordered_map<luisa::string, Value *> &values) noexcept {
        for (auto ref : instructions) {
            auto &&instr = graph.instructions[ref];
            switch (instr.tag) {
                case CoroInstructionTag::SIMPLE:
                    if (auto reg = instr.simple != nullptr && instr.simple->isa<CoroRegisterInst>() ?
                                       static_cast<CoroRegisterInst *>(instr.simple) :
                                       nullptr) {
                        values.emplace(luisa::string{reg->name()}, reg->value());
                    }
                    break;
                case CoroInstructionTag::IF:
                    collect_designated_values(graph, instr.true_branch, values);
                    collect_designated_values(graph, instr.false_branch, values);
                    break;
                case CoroInstructionTag::SWITCH:
                    for (auto &&c : instr.cases) { collect_designated_values(graph, c.body, values); }
                    collect_designated_values(graph, instr.default_body, values);
                    break;
                case CoroInstructionTag::SIMPLE_LOOP:
                case CoroInstructionTag::SKIP_IF_FIRST_FLAG:
                    collect_designated_values(graph, instr.body, values);
                    break;
                default: break;
            }
        }
    }

    [[nodiscard]] static bool terminated_in_current_branch(const PreliminaryGraph &preliminary,
                                                           const CoroGraphIndexer &current) noexcept {
        if (preliminary.instr(current.parent).tag == CoroInstructionTag::SIMPLE_LOOP) {
            // XIR lowers the loop backedge/exit into explicit loop-local leaves
            // inside the SimpleLoop body. Unlike the reference Loop{body, cond}
            // form, these leaves do not mean outer ancestors are unreachable;
            // the loop ancestor still needs to be reconstructed in the
            // continuation to preserve the self-loop and exit transition.
            return false;
        }
        auto &parent_branch = preliminary.get_parent_branch(current);
        for (auto i = current.index_in_parent_branch + 1u; i < parent_branch.size(); i++) {
            if (preliminary.is_terminator(parent_branch[i])) { return true; }
        }
        return false;
    }

    [[nodiscard]] static luisa::vector<CoroGraphIndexer>
    find_reachable_ancestors(const PreliminaryGraph &preliminary,
                             const CoroGraphIndexer &current,
                             const luisa::vector<CoroGraphIndexer> &ancestors) noexcept {
        luisa::vector<CoroGraphIndexer> reachable;
        auto cursor = current;
        for (auto iter = ancestors.rbegin(); iter != ancestors.rend(); ++iter) {
            if (terminated_in_current_branch(preliminary, cursor)) { break; }
            reachable.emplace_back(*iter);
            cursor = *iter;
        }
        std::reverse(reachable.begin(), reachable.end());
        return reachable;
    }

    static void construct_subscope_for_ancestors(CoroGraph &graph,
                                                 const PreliminaryGraph &preliminary,
                                                 CoroGraphIndexer suspend,
                                                 luisa::span<const CoroGraphIndexer> stack,
                                                 CoroInstructionRef first_flag,
                                                 bool inside_loop,
                                                 luisa::vector<CoroInstructionRef> &block) noexcept;

    [[nodiscard]] static CoroScope construct_subscope(CoroGraph &graph,
                                                      const PreliminaryGraph &preliminary,
                                                      const CoroGraphIndexer &current,
                                                      const luisa::vector<CoroGraphIndexer> &ancestors) noexcept;

    static void extract_continuation_at_suspend(CoroGraph &graph,
                                                const PreliminaryGraph &preliminary,
                                                const CoroGraphIndexer &current,
                                                const luisa::vector<CoroGraphIndexer> &ancestors) noexcept;

    static void recurse_continuation_extraction(CoroGraph &graph,
                                                const PreliminaryGraph &preliminary,
                                                CoroInstructionRef parent,
                                                uint32_t parent_branch,
                                                const luisa::vector<CoroInstructionRef> &body,
                                                luisa::vector<CoroGraphIndexer> &ancestors) noexcept;
};

void GraphBuilder::construct_subscope_for_ancestors(CoroGraph &graph,
                                                    const PreliminaryGraph &preliminary,
                                                    CoroGraphIndexer suspend,
                                                    luisa::span<const CoroGraphIndexer> stack,
                                                    CoroInstructionRef first_flag,
                                                    bool inside_loop,
                                                    luisa::vector<CoroInstructionRef> &block) noexcept {
    if (stack.empty()) {
        if (inside_loop) {
            auto suspend_ref = preliminary.get_instr_ref(suspend);
            if (!block.empty() &&
                graph.instructions[block.back()].tag == CoroInstructionTag::SKIP_IF_FIRST_FLAG &&
                graph.instructions[block.back()].related_instruction == first_flag) {
                graph.instructions[block.back()].body.emplace_back(suspend_ref);
            } else {
                block.emplace_back(static_cast<CoroInstructionRef>(graph.instructions.size()));
                graph.instructions.emplace_back(CoroInstruction{
                    .tag = CoroInstructionTag::SKIP_IF_FIRST_FLAG,
                    .related_instruction = first_flag,
                    .body = {suspend_ref}});
            }
        }
        block.emplace_back(static_cast<CoroInstructionRef>(graph.instructions.size()));
        graph.instructions.emplace_back(CoroInstruction{
            .tag = CoroInstructionTag::CLEAR_FIRST_FLAG,
            .related_instruction = first_flag});
        return;
    }

    auto clone_branch_range = [&](const luisa::vector<CoroInstructionRef> &src,
                                  luisa::vector<CoroInstructionRef> &dst,
                                  uint32_t begin,
                                  uint32_t end) noexcept {
        for (auto i = begin; i < end; i++) {
            auto ref = src[i];
            dst.emplace_back(clone_instruction(graph, preliminary, ref));
            if (preliminary.is_terminator(ref)) { break; }
        }
    };
    auto current = stack.front();
    auto make_suspend_branch = [&](const luisa::vector<CoroInstructionRef> &src) noexcept {
        luisa::vector<CoroInstructionRef> preceding;
        clone_branch_range(src, preceding, 0u, current.index_in_parent_branch);
        auto masked_ref = static_cast<CoroInstructionRef>(graph.instructions.size());
        graph.instructions.emplace_back(CoroInstruction{
            .tag = CoroInstructionTag::SKIP_IF_FIRST_FLAG,
            .related_instruction = first_flag,
            .body = std::move(preceding)});
        luisa::vector<CoroInstructionRef> cloned{masked_ref};
        auto loop_inside = inside_loop || preliminary.instr(current.parent).tag == CoroInstructionTag::SIMPLE_LOOP;
        construct_subscope_for_ancestors(graph, preliminary, suspend, stack.subspan(1u), first_flag, loop_inside, cloned);
        clone_branch_range(src, cloned, current.index_in_parent_branch + 1u, static_cast<uint32_t>(src.size()));
        return cloned;
    };
    auto clone_non_suspend_branch = [&](const luisa::vector<CoroInstructionRef> &src) noexcept {
        luisa::vector<CoroInstructionRef> cloned;
        clone_branch_range(src, cloned, 0u, static_cast<uint32_t>(src.size()));
        return cloned;
    };
    auto append_remaining_in_parent = [&] {
        auto &parent_branch = preliminary.get_parent_branch(current);
        auto loop_inside = inside_loop || preliminary.instr(current.parent).tag == CoroInstructionTag::SIMPLE_LOOP;
        construct_subscope_for_ancestors(graph, preliminary, suspend, stack.subspan(1u), first_flag, loop_inside, block);
        clone_branch_range(parent_branch, block, current.index_in_parent_branch + 1u, static_cast<uint32_t>(parent_branch.size()));
    };

    auto &&parent = preliminary.instr(current.parent);
    switch (parent.tag) {
        case CoroInstructionTag::IF:
            if (inside_loop) {
                auto true_branch = current.parent_branch == 0u ? make_suspend_branch(parent.true_branch) :
                                                                  clone_non_suspend_branch(parent.true_branch);
                auto false_branch = current.parent_branch == 1u ? make_suspend_branch(parent.false_branch) :
                                                                   clone_non_suspend_branch(parent.false_branch);
                block.emplace_back(static_cast<CoroInstructionRef>(graph.instructions.size()));
                graph.instructions.emplace_back(CoroInstruction{
                    .tag = CoroInstructionTag::IF,
                    .condition = parent.condition,
                    .true_branch = std::move(true_branch),
                    .false_branch = std::move(false_branch)});
            } else {
                append_remaining_in_parent();
            }
            break;
        case CoroInstructionTag::SWITCH:
            if (inside_loop) {
                luisa::vector<CoroSwitchCase> cases;
                cases.reserve(parent.cases.size());
                for (auto i = 0u; i < parent.cases.size(); i++) {
                    auto body = i == current.parent_branch ? make_suspend_branch(parent.cases[i].body) :
                                                             clone_non_suspend_branch(parent.cases[i].body);
                    cases.emplace_back(CoroSwitchCase{.value = parent.cases[i].value, .body = std::move(body)});
                }
                auto default_body = current.parent_branch == parent.cases.size() ? make_suspend_branch(parent.default_body) :
                                                                                    clone_non_suspend_branch(parent.default_body);
                block.emplace_back(static_cast<CoroInstructionRef>(graph.instructions.size()));
                graph.instructions.emplace_back(CoroInstruction{
                    .tag = CoroInstructionTag::SWITCH,
                    .condition = parent.condition,
                    .cases = std::move(cases),
                    .default_body = std::move(default_body)});
            } else {
                append_remaining_in_parent();
            }
            break;
        case CoroInstructionTag::SIMPLE_LOOP: {
            auto body = make_suspend_branch(parent.body);
            block.emplace_back(static_cast<CoroInstructionRef>(graph.instructions.size()));
            graph.instructions.emplace_back(CoroInstruction{
                .tag = CoroInstructionTag::SIMPLE_LOOP,
                .body = std::move(body)});
            break;
        }
        default:
            LUISA_ERROR_WITH_LOCATION("Unexpected coroutine ancestor instruction.");
    }
}

[[nodiscard]] CoroScope GraphBuilder::construct_subscope(CoroGraph &graph,
                                                         const PreliminaryGraph &preliminary,
                                                         const CoroGraphIndexer &current,
                                                         const luisa::vector<CoroGraphIndexer> &ancestors) noexcept {
    CoroScope scope;
    auto stack = ancestors;
    stack.emplace_back(current);
    replay_condition_stack(graph, preliminary, stack, scope);
    auto first_flag = static_cast<CoroInstructionRef>(graph.instructions.size());
    graph.instructions.emplace_back(CoroInstruction{.tag = CoroInstructionTag::MAKE_FIRST_FLAG});
    scope.instructions.emplace_back(first_flag);
    construct_subscope_for_ancestors(graph, preliminary, current, luisa::span{stack}.subspan(1u), first_flag, false, scope.instructions);
    auto outermost = stack.front();
    auto &parent_branch = preliminary.get_parent_branch(outermost);
    for (auto i = outermost.index_in_parent_branch + 1u; i < parent_branch.size(); i++) {
        auto ref = parent_branch[i];
        scope.instructions.emplace_back(clone_instruction(graph, preliminary, ref));
        if (preliminary.is_terminator(ref)) { break; }
    }
    remove_unreachable_from_block(graph, scope.instructions);
    collect_designated_values(graph, scope.instructions, scope.designated_values);
    return scope;
}

void GraphBuilder::extract_continuation_at_suspend(CoroGraph &graph,
                                                   const PreliminaryGraph &preliminary,
                                                   const CoroGraphIndexer &current,
                                                   const luisa::vector<CoroGraphIndexer> &ancestors) noexcept {
    auto ref = preliminary.get_instr_ref(current);
    luisa::optional<uint32_t> token;
    if (preliminary.instr(ref).tag == CoroInstructionTag::SUSPEND) { token = preliminary.instr(ref).token; }
    if (token.has_value() && graph.tokens.find(*token) != graph.tokens.end()) { return; }
    auto reachable_ancestors = find_reachable_ancestors(preliminary, current, ancestors);
    auto scope = construct_subscope(graph, preliminary, current, reachable_ancestors);
    auto scope_ref = static_cast<CoroScopeRef>(graph.scopes.size());
    graph.scopes.emplace_back(std::move(scope));
    if (token.has_value()) {
        graph.tokens.emplace(*token, scope_ref);
    } else {
        graph.entry = scope_ref;
    }
}

void GraphBuilder::recurse_continuation_extraction(CoroGraph &graph,
                                                   const PreliminaryGraph &preliminary,
                                                   CoroInstructionRef parent,
                                                   uint32_t parent_branch,
                                                   const luisa::vector<CoroInstructionRef> &body,
                                                   luisa::vector<CoroGraphIndexer> &ancestors) noexcept {
    for (auto index = 0u; index < body.size(); index++) {
        auto ref = body[index];
        auto current = CoroGraphIndexer{
            .parent = parent,
            .parent_branch = parent_branch,
            .index_in_parent_branch = index};
        auto &&instr = preliminary.instr(ref);
        switch (instr.tag) {
            case CoroInstructionTag::ENTRY:
            case CoroInstructionTag::SUSPEND:
                extract_continuation_at_suspend(graph, preliminary, current, ancestors);
                break;
            case CoroInstructionTag::IF:
                ancestors.emplace_back(current);
                recurse_continuation_extraction(graph, preliminary, ref, 0u, instr.true_branch, ancestors);
                recurse_continuation_extraction(graph, preliminary, ref, 1u, instr.false_branch, ancestors);
                ancestors.pop_back();
                break;
            case CoroInstructionTag::SWITCH:
                ancestors.emplace_back(current);
                for (auto i = 0u; i < instr.cases.size(); i++) {
                    recurse_continuation_extraction(graph, preliminary, ref, i, instr.cases[i].body, ancestors);
                }
                recurse_continuation_extraction(graph, preliminary, ref, static_cast<uint32_t>(instr.cases.size()), instr.default_body, ancestors);
                ancestors.pop_back();
                break;
            case CoroInstructionTag::SIMPLE_LOOP:
                ancestors.emplace_back(current);
                recurse_continuation_extraction(graph, preliminary, ref, 0u, instr.body, ancestors);
                ancestors.pop_back();
                break;
            default: break;
        }
    }
}

} // namespace detail

CoroGraph compute_coro_graph(CallableFunction *function) noexcept {
    auto preliminary = detail::PreliminaryTranslator{function}.build();
    CoroGraph graph;
    graph.instructions = preliminary.instructions;
    luisa::vector<detail::CoroGraphIndexer> ancestors;
    auto &entry_scope = preliminary.instr(preliminary.entry_scope);
    detail::GraphBuilder::recurse_continuation_extraction(graph, preliminary, preliminary.entry_scope, 0u, entry_scope.body, ancestors);
    for (auto &&scope : graph.scopes) {
        for (auto &&[name, value] : scope.designated_values) { graph.designated_values.emplace(name, value); }
    }
    return graph;
}

namespace detail {

struct AccessPathInfo {
    Value *root{nullptr};
    luisa::vector<int32_t> chain;
};

[[nodiscard]] static bool equal_access_paths(luisa::span<const CoroAccessPath> lhs,
                                             luisa::span<const CoroAccessPath> rhs) noexcept {
    if (lhs.size() != rhs.size()) { return false; }
    for (auto i = 0u; i < lhs.size(); i++) {
        if (lhs[i].root != rhs[i].root || lhs[i].chain != rhs[i].chain) { return false; }
    }
    return true;
}

[[nodiscard]] static luisa::optional<int32_t> try_evaluate_access_index(const Value *value) noexcept {
    if (auto constant = value->isa<Constant>() ? static_cast<const Constant *>(value) : nullptr) {
        switch (constant->type()->tag()) {
            case Type::Tag::INT8: return static_cast<int32_t>(constant->as<byte>());
            case Type::Tag::UINT8: return static_cast<int32_t>(constant->as<ubyte>());
            case Type::Tag::INT16: return static_cast<int32_t>(constant->as<short>());
            case Type::Tag::UINT16: return static_cast<int32_t>(constant->as<ushort>());
            case Type::Tag::INT32: return static_cast<int32_t>(constant->as<int>());
            case Type::Tag::UINT32: return static_cast<int32_t>(constant->as<uint>());
            case Type::Tag::INT64: return static_cast<int32_t>(constant->as<slong>());
            case Type::Tag::UINT64: return static_cast<int32_t>(constant->as<ulong>());
            default: break;
        }
    }
    return {};
}

[[nodiscard]] static AccessPathInfo access_path_from_value(Value *value) noexcept {
    if (auto gep = value->isa<GEPInst>() ? static_cast<GEPInst *>(value) : nullptr) {
        auto info = access_path_from_value(gep->base());
        for (auto i = 0u; i < gep->index_count(); i++) {
            if (auto index = try_evaluate_access_index(gep->index(i))) {
                info.chain.emplace_back(*index);
            } else {
                info.chain.emplace_back(-1);
                break;
            }
        }
        return info;
    }
    return AccessPathInfo{
        .root = value,
        .chain = {}};
}

[[nodiscard]] static bool is_frame_excluded_root(const Value *value) noexcept {
    if (value == nullptr) { return true; }
    return value->isa<Argument>() ||
           value->isa<Constant>() ||
           value->isa<Undefined>() ||
           value->isa<SpecialRegister>() ||
           value->isa<CoroIdInst>() ||
           value->isa<CoroTokenInst>() ||
           value->isa<Function>() ||
           value->isa<BasicBlock>();
}

[[nodiscard]] static luisa::vector<int32_t> to_signed_chain(luisa::span<const uint32_t> chain) noexcept {
    luisa::vector<int32_t> result;
    result.reserve(chain.size());
    for (auto i : chain) { result.emplace_back(static_cast<int32_t>(i)); }
    return result;
}

struct ScopeUseDefAnalyzer {
    const CoroGraph &graph;
    CoroScopeUseDef result;
    ReplayableValueAnalysis replayable;

    void add_external_use(Value *root, luisa::span<const int32_t> chain) noexcept {
        AccessSet uses{result.external_uses};
        uses.insert(root, chain);
        result.external_uses = uses.to_public();
    }

    void add_touch(Value *root, luisa::span<const int32_t> chain) noexcept {
        AccessSet touches{result.internal_touches};
        touches.insert(root, chain);
        result.internal_touches = touches.to_public();
    }

    void record_suspend_kills(CoroScopeRef target, const AccessSet &kills) noexcept {
        auto coalesced = kills;
        coalesced.coalesce_whole_access_chains();
        coalesced.dynamic_access_chains_as_whole();
        if (auto iter = result.internal_kills.find(target); iter != result.internal_kills.end()) {
            auto merged = AccessSet::intersect(AccessSet{iter->second}, coalesced);
            merged.coalesce_whole_access_chains();
            merged.dynamic_access_chains_as_whole();
            iter->second = merged.to_public();
        } else {
            result.internal_kills.emplace(target, coalesced.to_public());
        }
    }

    void mark_gep_index_uses(Value *value, AccessSet &kills) noexcept {
        if (auto gep = value->isa<GEPInst>() ? static_cast<GEPInst *>(value) : nullptr) {
            mark_gep_index_uses(gep->base(), kills);
            for (auto i = 0u; i < gep->index_count(); i++) {
                if (!try_evaluate_access_index(gep->index(i)).has_value()) {
                    mark_use(gep->index(i), kills);
                }
            }
        }
    }

    void mark_use(Value *value, AccessSet &kills) noexcept {
        if (value == nullptr) { return; }
        mark_gep_index_uses(value, kills);
        if (replayable.detect(value)) { return; }
        auto info = access_path_from_value(value);
        if (info.root == nullptr || is_frame_excluded_root(info.root)) { return; }
        if (!kills.contains(info.root, info.chain)) {
            kills.coalesce_whole_access_chains();
            kills.dynamic_access_chains_as_whole();
            if (!kills.contains(info.root, info.chain)) { add_external_use(info.root, info.chain); }
        }
    }

    void mark_kill(Value *value, AccessSet &kills) noexcept {
        if (value == nullptr) { return; }
        mark_gep_index_uses(value, kills);
        auto info = access_path_from_value(value);
        if (info.root == nullptr || is_frame_excluded_root(info.root)) { return; }
        kills.insert(info.root, info.chain);
        add_touch(info.root, info.chain);
    }

    void mark_touch(Value *value, AccessSet &kills) noexcept {
        if (value == nullptr) { return; }
        mark_gep_index_uses(value, kills);
        auto info = access_path_from_value(value);
        if (info.root == nullptr || is_frame_excluded_root(info.root)) { return; }
        add_touch(info.root, info.chain);
    }

    void analyze_call(CallInst *call, AccessSet &kills) noexcept {
        auto callee = call->callee();
        for (auto i = 0u; i < call->argument_count(); i++) {
            auto arg = call->argument(i);
            mark_use(arg, kills);
            if (callee != nullptr) {
                auto formal = static_cast<const Argument *>(nullptr);
                auto index = 0u;
                for (auto a : callee->arguments()) {
                    if (index++ == i) {
                        formal = a;
                        break;
                    }
                }
                if (formal != nullptr && (formal->is_reference() || formal->is_resource())) {
                    mark_touch(arg, kills);
                }
            }
        }
        if (call->type() != nullptr) { mark_kill(call, kills); }
    }

    void analyze_simple_instruction(Instruction *inst, AccessSet &kills) noexcept {
        switch (inst->derived_instruction_tag()) {
            case DerivedInstructionTag::ALLOCA:
            case DerivedInstructionTag::CORO_REGISTER:
            case DerivedInstructionTag::CORO_ID:
            case DerivedInstructionTag::CORO_TOKEN:
            case DerivedInstructionTag::CLOCK:
                if (inst->type() != nullptr) { mark_kill(inst, kills); }
                break;
            case DerivedInstructionTag::LOAD: {
                auto load = static_cast<LoadInst *>(inst);
                mark_use(load->variable(), kills);
                mark_kill(load, kills);
                break;
            }
            case DerivedInstructionTag::STORE: {
                auto store = static_cast<StoreInst *>(inst);
                mark_use(store->value(), kills);
                mark_kill(store->variable(), kills);
                break;
            }
            case DerivedInstructionTag::GEP:
                mark_gep_index_uses(inst, kills);
                break;
            case DerivedInstructionTag::ATOMIC:
                if (inst->operand_count() != 0u) {
                    mark_use(inst->operand(0u), kills);
                    mark_touch(inst->operand(0u), kills);
                    mark_kill(inst->operand(0u), kills);
                }
                for (auto i = 1u; i < inst->operand_count(); i++) { mark_use(inst->operand(i), kills); }
                if (inst->type() != nullptr) { mark_kill(inst, kills); }
                break;
            case DerivedInstructionTag::CALL:
                analyze_call(static_cast<CallInst *>(inst), kills);
                break;
            case DerivedInstructionTag::RESOURCE_WRITE:
                for (auto i = 0u; i < inst->operand_count(); i++) { mark_use(inst->operand(i), kills); }
                if (inst->operand_count() != 0u) { mark_touch(inst->operand(0u), kills); }
                break;
            default:
                for (auto i = 0u; i < inst->operand_count(); i++) { mark_use(inst->operand(i), kills); }
                if (inst->type() != nullptr) { mark_kill(inst, kills); }
                break;
        }
    }

    void analyze_block(const luisa::vector<CoroInstructionRef> &block,
                       AccessSet &kills,
                       bool execute_skip_if_first_body = false) noexcept {
        for (auto ref : block) {
            auto &&instr = graph.instructions.at(ref);
            switch (instr.tag) {
                case CoroInstructionTag::SIMPLE:
                    if (instr.simple != nullptr) { analyze_simple_instruction(instr.simple, kills); }
                    break;
                case CoroInstructionTag::CONDITION_STACK_REPLAY:
                    for (auto &&item : instr.condition_stack) { mark_kill(item.value, kills); }
                    break;
                case CoroInstructionTag::IF: {
                    mark_use(instr.condition, kills);
                    auto true_kills = kills;
                    auto false_kills = kills;
                    analyze_block(instr.true_branch, true_kills, execute_skip_if_first_body);
                    analyze_block(instr.false_branch, false_kills, execute_skip_if_first_body);
                    kills = AccessSet::intersect(true_kills, false_kills);
                    break;
                }
                case CoroInstructionTag::SWITCH: {
                    mark_use(instr.condition, kills);
                    luisa::optional<AccessSet> merged;
                    for (auto &&c : instr.cases) {
                        auto case_kills = kills;
                        analyze_block(c.body, case_kills, execute_skip_if_first_body);
                        merged = merged.has_value() ? AccessSet::intersect(*merged, case_kills) : case_kills;
                    }
                    auto default_kills = kills;
                    analyze_block(instr.default_body, default_kills, execute_skip_if_first_body);
                    kills = merged.has_value() ? AccessSet::intersect(*merged, default_kills) : default_kills;
                    break;
                }
                case CoroInstructionTag::SIMPLE_LOOP: {
                    auto loop_kills = kills;
                    analyze_block(instr.body, loop_kills, false);
                    kills = loop_kills;
                    // A loop-local suspend can be reached again only after one full
                    // iteration has updated the loop-carried state and the backedge
                    // re-enters the loop head.
                    auto backedge_kills = loop_kills;
                    analyze_block(instr.body, backedge_kills, true);
                    break;
                }
                case CoroInstructionTag::SKIP_IF_FIRST_FLAG: {
                    auto branch_kills = kills;
                    analyze_block(instr.body, branch_kills, true);
                    break;
                }
                case CoroInstructionTag::SUSPEND:
                    if (auto iter = graph.tokens.find(instr.token); iter != graph.tokens.end()) {
                        record_suspend_kills(iter->second, kills);
                    }
                    return;
                case CoroInstructionTag::LOOP_CONTINUE:
                case CoroInstructionTag::LOOP_BREAK:
                case CoroInstructionTag::TERMINATE:
                    return;
                default: break;
            }
        }
    }
};

static void probe_suspend_tokens(const CoroGraph &graph,
                                 const luisa::vector<CoroInstructionRef> &instructions,
                                 luisa::vector<uint32_t> &tokens) noexcept {
    for (auto ref : instructions) {
        auto &&instr = graph.instructions.at(ref);
        switch (instr.tag) {
            case CoroInstructionTag::IF:
                probe_suspend_tokens(graph, instr.true_branch, tokens);
                probe_suspend_tokens(graph, instr.false_branch, tokens);
                break;
            case CoroInstructionTag::SWITCH:
                for (auto &&c : instr.cases) { probe_suspend_tokens(graph, c.body, tokens); }
                probe_suspend_tokens(graph, instr.default_body, tokens);
                break;
            case CoroInstructionTag::SIMPLE_LOOP:
            case CoroInstructionTag::SKIP_IF_FIRST_FLAG:
                probe_suspend_tokens(graph, instr.body, tokens);
                break;
            case CoroInstructionTag::SUSPEND:
                tokens.emplace_back(instr.token);
                break;
            default: break;
        }
    }
}

[[nodiscard]] static luisa::unordered_set<Value *> collect_designated_value_roots(const CoroGraph &graph) noexcept {
    luisa::unordered_set<Value *> result;
    for (auto &&[_, value] : graph.designated_values) { result.emplace(value); }
    return result;
}

struct StableValueIndexer {
    const CoroGraph &graph;
    luisa::unordered_map<const Value *, uint32_t> indices;

    void record_value(const Value *value) noexcept {
        if (value == nullptr) { return; }
        if (indices.contains(value)) { return; }
        indices.emplace(value, static_cast<uint32_t>(indices.size()));
        auto inst = value->isa<Instruction>() ? static_cast<const Instruction *>(value) : nullptr;
        if (inst == nullptr) { return; }
        for (auto i = 0u; i < inst->operand_count(); i++) { record_value(inst->operand(i)); }
    }

    void record_block(const luisa::vector<CoroInstructionRef> &block) noexcept {
        for (auto ref : block) {
            auto &&instr = graph.instructions.at(ref);
            switch (instr.tag) {
                case CoroInstructionTag::SIMPLE:
                    record_value(instr.simple);
                    break;
                case CoroInstructionTag::CONDITION_STACK_REPLAY:
                    for (auto &&item : instr.condition_stack) { record_value(item.value); }
                    break;
                case CoroInstructionTag::MAKE_FIRST_FLAG:
                case CoroInstructionTag::CLEAR_FIRST_FLAG:
                case CoroInstructionTag::LOOP_CONTINUE:
                case CoroInstructionTag::LOOP_BREAK:
                case CoroInstructionTag::SUSPEND:
                case CoroInstructionTag::TERMINATE:
                    break;
                case CoroInstructionTag::IF:
                    record_value(instr.condition);
                    record_block(instr.true_branch);
                    record_block(instr.false_branch);
                    break;
                case CoroInstructionTag::SWITCH:
                    record_value(instr.condition);
                    for (auto &&c : instr.cases) { record_block(c.body); }
                    record_block(instr.default_body);
                    break;
                case CoroInstructionTag::SIMPLE_LOOP:
                case CoroInstructionTag::SKIP_IF_FIRST_FLAG:
                    record_block(instr.body);
                    break;
                default: break;
            }
        }
    }

    [[nodiscard]] luisa::unordered_map<const Value *, uint32_t> compute() noexcept {
        if (graph.entry != invalid_coro_scope_ref) { record_block(graph.scopes.at(graph.entry).instructions); }
        for (auto &&[_, scope_ref] : graph.tokens) { record_block(graph.scopes.at(scope_ref).instructions); }
        return indices;
    }
};

[[nodiscard]] static luisa::unordered_map<const Value *, uint32_t>
compute_stable_value_indices(const CoroGraph &graph) noexcept {
    return StableValueIndexer{graph}.compute();
}

[[nodiscard]] static uint32_t stable_value_index(const luisa::unordered_map<const Value *, uint32_t> &indices,
                                                 const Value *value) noexcept {
    if (auto iter = indices.find(value); iter != indices.end()) { return iter->second; }
    return std::numeric_limits<uint32_t>::max();
}

[[nodiscard]] static size_t stable_field_sort_size(const Type *type) noexcept {
    LUISA_ASSERT(type != nullptr && type->is_scalar(),
                 "Coroutine frame fields must be scalar leaves, got '{}'.",
                 type == nullptr ? "<null>" : type->description());
    return type->is_bool() ? 1u : type->size() * 8u;
}

[[nodiscard]] static bool debug_analysis_dump_enabled() noexcept {
    if (auto env = std::getenv("LUISA_CORO_DEBUG_ANALYSIS")) {
        return std::string_view{env} == "1";
    }
    return false;
}

[[nodiscard]] static luisa::string describe_value_brief(
    const luisa::unordered_map<const Value *, uint32_t> &stable_indices,
    const Value *value) noexcept {
    if (value == nullptr) { return "<null>"; }
    luisa::string s;
    s.append("#").append(std::to_string(stable_value_index(stable_indices, value)));
    s.append(":");
    if (auto arg = value->isa<Argument>() ? static_cast<const Argument *>(value) : nullptr) {
        if (arg->is_value()) {
            s.append("arg[value]");
        } else if (arg->is_reference()) {
            s.append("arg[ref]");
        } else if (arg->is_resource()) {
            s.append("arg[res]");
        } else {
            s.append("arg[?]");
        }
    } else if (auto inst = value->isa<Instruction>() ? static_cast<const Instruction *>(value) : nullptr) {
        s.append(luisa::string{xir::to_string(inst->derived_instruction_tag())});
    } else if (value->isa<Constant>()) {
        s.append("const");
    } else if (value->isa<SpecialRegister>()) {
        s.append("special");
    } else {
        s.append("value");
    }
    if (auto type = value->type()) {
        s.append(":").append(type->description());
    }
    return s;
}

static void dump_access_paths(luisa::string_view label,
                              luisa::span<const CoroAccessPath> paths,
                              const luisa::unordered_map<const Value *, uint32_t> &stable_indices) noexcept {
    LUISA_INFO("{} (count={})", label, paths.size());
    for (auto &&path : paths) {
        luisa::string s;
        s.append("  - ").append(describe_value_brief(stable_indices, path.root)).append(" [");
        for (auto i = 0u; i < path.chain.size(); i++) {
            if (i != 0u) { s.append(", "); }
            s.append(std::to_string(path.chain[i]));
        }
        s.append("]");
        LUISA_INFO("{}", s);
    }
}

static void dump_frame_fields(const CoroFrame &frame,
                              const luisa::unordered_map<const Value *, uint32_t> &stable_indices) noexcept {
    LUISA_INFO("frame fields (count={})", frame.fields.size());
    for (auto &&field : frame.fields) {
        luisa::string s;
        s.append("  - frame[").append(std::to_string(field.frame_index)).append("] ");
        s.append(describe_value_brief(stable_indices, field.root)).append(" [");
        for (auto i = 0u; i < field.chain.size(); i++) {
            if (i != 0u) { s.append(", "); }
            s.append(std::to_string(field.chain[i]));
        }
        s.append("] : ").append(field.type != nullptr ? field.type->description() : "<null>");
        LUISA_INFO("{}", s);
    }
}

static void dump_coro_analysis(const CoroGraph &graph,
                               const CoroGraphUseDef &use_def,
                               const CoroTransitionGraph &transition,
                               const CoroFrame &frame) noexcept {
    if (!debug_analysis_dump_enabled()) { return; }
    auto stable_indices = compute_stable_value_indices(graph);
    for (auto i = 0u; i < graph.scopes.size(); i++) {
        auto scope_ref = static_cast<CoroScopeRef>(i);
        auto &&scope_use_def = use_def.scopes.at(scope_ref);
        dump_access_paths(luisa::format("scope {} external_uses", i), scope_use_def.external_uses, stable_indices);
        dump_access_paths(luisa::format("scope {} internal_touches", i), scope_use_def.internal_touches, stable_indices);
        for (auto &&[target, kills] : scope_use_def.internal_kills) {
            dump_access_paths(luisa::format("scope {} internal_kills -> {}", i, target), kills, stable_indices);
        }
    }
    dump_access_paths("graph union_uses", use_def.union_uses, stable_indices);
    for (auto i = 0u; i < graph.scopes.size(); i++) {
        auto scope_ref = static_cast<CoroScopeRef>(i);
        auto &&node = transition.nodes.at(scope_ref);
        dump_access_paths(luisa::format("transition {} union_live", i), node.union_live_states, stable_indices);
        dump_access_paths(luisa::format("transition {} union_load", i), node.union_states_to_load, stable_indices);
        dump_access_paths(luisa::format("transition {} union_save", i), node.union_states_to_save, stable_indices);
        for (auto &&[token, edge] : node.outlets) {
            dump_access_paths(luisa::format("transition {} edge {} live", i, token), edge.live_states, stable_indices);
            dump_access_paths(luisa::format("transition {} edge {} load", i, token), edge.states_to_load, stable_indices);
            dump_access_paths(luisa::format("transition {} edge {} save", i, token), edge.states_to_save, stable_indices);
        }
    }
    dump_frame_fields(frame, stable_indices);
}

[[nodiscard]] static bool debug_graph_dump_enabled() noexcept {
    if (auto env = std::getenv("LUISA_CORO_DEBUG_GRAPH")) {
        return std::string_view{env} == "1";
    }
    return false;
}

static void dump_coro_instruction_tree(const CoroGraph &graph,
                                       const luisa::unordered_map<const Value *, uint32_t> &stable_indices,
                                       const luisa::vector<CoroInstructionRef> &block,
                                       uint32_t indent) noexcept {
    auto print_indent = [indent](uint32_t extra) noexcept {
        luisa::string s;
        s.reserve((indent + extra) * 2u);
        for (auto i = 0u; i < indent + extra; i++) { s.append("  "); }
        return s;
    };
    for (auto ref : block) {
        auto &&instr = graph.instructions.at(ref);
        switch (instr.tag) {
            case CoroInstructionTag::SIMPLE:
                LUISA_INFO("{}#{} SIMPLE {}",
                           print_indent(0u),
                           ref,
                           instr.simple != nullptr ? xir::to_string(instr.simple->derived_instruction_tag()) : "<null>");
                break;
            case CoroInstructionTag::CONDITION_STACK_REPLAY:
                LUISA_INFO("{}#{} CONDITION_STACK_REPLAY ({} items)",
                           print_indent(0u), ref, instr.condition_stack.size());
                for (auto &&item : instr.condition_stack) {
                    LUISA_INFO("{}{} = {}",
                               print_indent(1u),
                               describe_value_brief(stable_indices, item.value),
                               item.selected_value);
                }
                break;
            case CoroInstructionTag::MAKE_FIRST_FLAG:
                LUISA_INFO("{}#{} MAKE_FIRST_FLAG", print_indent(0u), ref);
                break;
            case CoroInstructionTag::SKIP_IF_FIRST_FLAG:
                LUISA_INFO("{}#{} SKIP_IF_FIRST_FLAG(flag=#{}) {{",
                           print_indent(0u), ref, instr.related_instruction);
                dump_coro_instruction_tree(graph, stable_indices, instr.body, indent + 1u);
                LUISA_INFO("{}}}", print_indent(0u));
                break;
            case CoroInstructionTag::CLEAR_FIRST_FLAG:
                LUISA_INFO("{}#{} CLEAR_FIRST_FLAG(flag=#{})",
                           print_indent(0u), ref, instr.related_instruction);
                break;
            case CoroInstructionTag::IF:
                LUISA_INFO("{}#{} IF [{}] {{",
                           print_indent(0u),
                           ref,
                           describe_value_brief(stable_indices, instr.condition));
                LUISA_INFO("{}true:", print_indent(1u));
                dump_coro_instruction_tree(graph, stable_indices, instr.true_branch, indent + 2u);
                LUISA_INFO("{}false:", print_indent(1u));
                dump_coro_instruction_tree(graph, stable_indices, instr.false_branch, indent + 2u);
                LUISA_INFO("{}}}", print_indent(0u));
                break;
            case CoroInstructionTag::SWITCH:
                LUISA_INFO("{}#{} SWITCH [{}] {{",
                           print_indent(0u),
                           ref,
                           describe_value_brief(stable_indices, instr.condition));
                for (auto &&c : instr.cases) {
                    LUISA_INFO("{}case {}:", print_indent(1u), c.value);
                    dump_coro_instruction_tree(graph, stable_indices, c.body, indent + 2u);
                }
                LUISA_INFO("{}default:", print_indent(1u));
                dump_coro_instruction_tree(graph, stable_indices, instr.default_body, indent + 2u);
                LUISA_INFO("{}}}", print_indent(0u));
                break;
            case CoroInstructionTag::SIMPLE_LOOP:
                LUISA_INFO("{}#{} SIMPLE_LOOP {{", print_indent(0u), ref);
                dump_coro_instruction_tree(graph, stable_indices, instr.body, indent + 1u);
                LUISA_INFO("{}}}", print_indent(0u));
                break;
            case CoroInstructionTag::LOOP_CONTINUE:
                LUISA_INFO("{}#{} LOOP_CONTINUE", print_indent(0u), ref);
                break;
            case CoroInstructionTag::LOOP_BREAK:
                LUISA_INFO("{}#{} LOOP_BREAK", print_indent(0u), ref);
                break;
            case CoroInstructionTag::SUSPEND:
                LUISA_INFO("{}#{} SUSPEND(token={})", print_indent(0u), ref, instr.token);
                break;
            case CoroInstructionTag::TERMINATE:
                LUISA_INFO("{}#{} TERMINATE", print_indent(0u), ref);
                break;
            case CoroInstructionTag::ENTRY:
                LUISA_INFO("{}#{} ENTRY", print_indent(0u), ref);
                break;
            case CoroInstructionTag::ENTRY_SCOPE:
                LUISA_INFO("{}#{} ENTRY_SCOPE {{", print_indent(0u), ref);
                dump_coro_instruction_tree(graph, stable_indices, instr.body, indent + 1u);
                LUISA_INFO("{}}}", print_indent(0u));
                break;
        }
    }
}

static void dump_coro_graph_if_requested(const CoroGraph &graph) noexcept {
    if (!debug_graph_dump_enabled()) { return; }
    auto stable_indices = compute_stable_value_indices(graph);
    LUISA_INFO("coro graph entry scope = {}", graph.entry);
    for (auto i = 0u; i < graph.scopes.size(); i++) {
        LUISA_INFO("scope {} {{", i);
        dump_coro_instruction_tree(graph, stable_indices, graph.scopes.at(i).instructions, 1u);
        LUISA_INFO("}}");
    }
}

}// namespace detail

namespace detail {

[[nodiscard]] static bool is_leaf_type(const Type *type) noexcept {
    return type != nullptr && type->is_scalar();
}

static void enumerate_leaf_paths(const Type *type,
                                 luisa::vector<uint32_t> &path,
                                 const luisa::function<void(const Type *, luisa::span<const uint32_t>)> &visit) noexcept {
    if (is_leaf_type(type)) {
        visit(type, path);
        return;
    }
    switch (type->tag()) {
        case Type::Tag::VECTOR:
        case Type::Tag::ARRAY: {
            auto element = type->element();
            auto dim = type->dimension();
            for (auto i = 0u; i < dim; i++) {
                path.emplace_back(i);
                enumerate_leaf_paths(element, path, visit);
                path.pop_back();
            }
            return;
        }
        case Type::Tag::MATRIX: {
            auto row = Type::vector(type->element(), type->dimension());
            for (auto i = 0u; i < type->dimension(); i++) {
                path.emplace_back(i);
                enumerate_leaf_paths(row, path, visit);
                path.pop_back();
            }
            return;
        }
        case Type::Tag::STRUCTURE: {
            auto members = type->members();
            LUISA_ASSERT(!members.empty(), "Unsupported empty structure '{}' in coroutine frame.", type->description());
            for (auto i = 0u; i < members.size(); i++) {
                path.emplace_back(i);
                enumerate_leaf_paths(members[i], path, visit);
                path.pop_back();
            }
            return;
        }
        default:
            LUISA_ERROR_WITH_LOCATION("Unsupported non-leaf type '{}' in coroutine frame.", type->description());
    }
}

static void enumerate_leaf_paths(const Type *type,
                                 const luisa::function<void(const Type *, luisa::span<const uint32_t>)> &visit) noexcept {
    luisa::vector<uint32_t> path;
    enumerate_leaf_paths(type, path, visit);
}

[[nodiscard]] static luisa::vector<uint32_t> collect_frame_indices(const CoroFrame &frame,
                                                                   luisa::span<const CoroAccessPath> paths,
                                                                   bool include_coro_id) noexcept {
    AccessSet set{paths};
    luisa::vector<uint32_t> indices;
    if (include_coro_id) { indices.emplace_back(0u); }
    for (auto &&field : frame.fields) {
        if (set.contains(field.root, to_signed_chain(field.chain))) { indices.emplace_back(field.frame_index); }
    }
    for (auto &&field : frame.designated_fields) {
        if (set.contains(field.value, {})) { indices.emplace_back(field.frame_index); }
    }
    indices.emplace_back(1u);
    std::sort(indices.begin(), indices.end());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    return indices;
}

}// namespace detail

CoroGraphUseDef compute_coro_use_def(const CoroGraph &graph) noexcept {
    CoroGraphUseDef result;
    detail::AccessSet union_uses;
    for (auto i = 0u; i < graph.scopes.size(); i++) {
        auto scope_ref = static_cast<CoroScopeRef>(i);
        auto analyzer = detail::ScopeUseDefAnalyzer{.graph = graph};
        detail::AccessSet kills;
        analyzer.analyze_block(graph.scopes.at(i).instructions, kills);
        auto scope_use_def = std::move(analyzer.result);
        {
            auto touches = detail::AccessSet{scope_use_def.internal_touches};
            touches.coalesce_whole_access_chains();
            touches.dynamic_access_chains_as_whole();
            scope_use_def.internal_touches = touches.to_public();
        }
        {
            auto external_uses = detail::AccessSet{scope_use_def.external_uses};
            external_uses.coalesce_whole_access_chains();
            external_uses.dynamic_access_chains_as_whole();
            scope_use_def.external_uses = external_uses.to_public();
        }
        union_uses.union_with(detail::AccessSet{scope_use_def.external_uses});
        result.scopes.emplace(scope_ref, std::move(scope_use_def));
    }
    union_uses.coalesce_whole_access_chains();
    union_uses.dynamic_access_chains_as_whole();
    result.union_uses = union_uses.to_public();
    return result;
}

CoroTransitionGraph compute_coro_transition_graph(const CoroGraph &graph, const CoroGraphUseDef &use_def) noexcept {
    CoroTransitionGraph result;
    result.union_states = use_def.union_uses;
    auto designated_roots = detail::collect_designated_value_roots(graph);
    for (auto i = 0u; i < graph.scopes.size(); i++) {
        auto scope_ref = static_cast<CoroScopeRef>(i);
        luisa::vector<uint32_t> suspend_tokens;
        detail::probe_suspend_tokens(graph, graph.scopes.at(i).instructions, suspend_tokens);
        std::sort(suspend_tokens.begin(), suspend_tokens.end());
        suspend_tokens.erase(std::unique(suspend_tokens.begin(), suspend_tokens.end()), suspend_tokens.end());
        CoroTransitionState state;
        state.scope = scope_ref;
        for (auto token : suspend_tokens) {
            auto target = graph.tokens.at(token);
            state.outlets.emplace(token, CoroTransitionEdge{
                                            .target = target,
                                            .live_states = use_def.scopes.at(target).external_uses});
        }
        result.nodes.emplace(scope_ref, std::move(state));
    }

    auto any_change = true;
    while (any_change) {
        any_change = false;
        auto snapshot = result.nodes;
        for (auto &[scope_ref, node] : result.nodes) {
            for (auto &[token, edge] : node.outlets) {
                detail::AccessSet live{use_def.scopes.at(edge.target).external_uses};
                auto target_use_def = use_def.scopes.at(edge.target);
                if (auto target_node_iter = snapshot.find(edge.target); target_node_iter != snapshot.end()) {
                    for (auto &&[_, child_edge] : target_node_iter->second.outlets) {
                        detail::AccessSet child_live{child_edge.live_states};
                        detail::AccessSet child_kill;
                        if (auto kill_iter = target_use_def.internal_kills.find(child_edge.target);
                            kill_iter != target_use_def.internal_kills.end()) {
                            child_kill = detail::AccessSet{kill_iter->second};
                        }
                        live.union_with(detail::AccessSet::subtract(child_live, child_kill));
                    }
                }
                for (auto root : designated_roots) { live.insert(root, {}); }
                auto public_live = live.to_public();
                if (!detail::equal_access_paths(public_live, edge.live_states)) {
                    edge.live_states = std::move(public_live);
                    any_change = true;
                }
            }
        }
    }

    for (auto &[scope_ref, node] : result.nodes) {
        auto scope_use_def = use_def.scopes.at(scope_ref);
        auto external_use = detail::AccessSet{scope_use_def.external_uses};
        auto internal_touch = detail::AccessSet{scope_use_def.internal_touches};
        detail::AccessSet union_live;
        auto union_load = external_use;
        detail::AccessSet union_save;
        for (auto &[token, edge] : node.outlets) {
            auto live = detail::AccessSet{edge.live_states};
            if (edge.target == scope_ref) {
                // XIR scopes lower source loops into self-resuming synthetic
                // subscopes. To preserve loop-carried state across the
                // suspend/backedge boundary, treat locally touched values as
                // live on self-edges before applying the reference load/save
                // equations below.
                live.union_with(internal_touch);
            }
            detail::AccessSet internal_kill;
            if (auto kill_iter = scope_use_def.internal_kills.find(edge.target);
                kill_iter != scope_use_def.internal_kills.end()) {
                internal_kill = detail::AccessSet{kill_iter->second};
            }
            auto load = detail::AccessSet::union_of(
                detail::AccessSet::intersect(
                    detail::AccessSet::subtract(live, internal_kill),
                    internal_touch),
                external_use);
            auto save = detail::AccessSet::intersect(live, internal_touch);
            edge.states_to_load = load.to_public();
            edge.states_to_save = save.to_public();
            union_live.union_with(live);
            union_load.union_with(load);
            union_save.union_with(save);
        }
        node.union_live_states = union_live.to_public();
        node.union_states_to_load = union_load.to_public();
        node.union_states_to_save = union_save.to_public();
    }
    return result;
}

CoroFrame compute_coro_frame(const CoroGraph &graph, const CoroTransitionGraph &transition_graph) noexcept {
    CoroFrame result;
    auto union_states = detail::AccessSet{transition_graph.union_states};
    auto designated_roots = detail::collect_designated_value_roots(graph);
    auto stable_indices = detail::compute_stable_value_indices(graph);
    auto public_union_states = union_states.to_public();
    luisa::vector<Value *> union_roots;
    union_roots.reserve(public_union_states.size());
    for (auto &&path : public_union_states) {
        if (path.root == nullptr || designated_roots.contains(path.root)) { continue; }
        if (std::find(union_roots.begin(), union_roots.end(), path.root) == union_roots.end()) {
            union_roots.emplace_back(path.root);
        }
    }
    std::stable_sort(union_roots.begin(), union_roots.end(), [&](auto lhs, auto rhs) noexcept {
        return detail::stable_value_index(stable_indices, lhs) < detail::stable_value_index(stable_indices, rhs);
    });

    auto append_fields = [&](bool for_aggregates) noexcept {
        luisa::vector<CoroFrameFieldInfo> fields;
        for (auto root : union_roots) {
            if (root == nullptr || root->type() == nullptr) { continue; }
            auto is_aggregate_root = !root->type()->is_scalar();
            if (is_aggregate_root != for_aggregates) { continue; }
            detail::enumerate_leaf_paths(root->type(), [&](const Type *leaf_type,
                                                           luisa::span<const uint32_t> chain) noexcept {
                if (union_states.contains(root, detail::to_signed_chain(chain))) {
                    fields.emplace_back(CoroFrameFieldInfo{
                        .type = leaf_type,
                        .root = root,
                        .chain = luisa::vector<uint32_t>{chain.begin(), chain.end()}});
                }
            });
        }
        std::stable_sort(fields.begin(), fields.end(), [&](auto &&lhs, auto &&rhs) noexcept {
            auto lhs_index = detail::stable_value_index(stable_indices, lhs.root);
            auto rhs_index = detail::stable_value_index(stable_indices, rhs.root);
            auto lhs_size = detail::stable_field_sort_size(lhs.type);
            auto rhs_size = detail::stable_field_sort_size(rhs.type);
            if (for_aggregates) {
                if (lhs_index != rhs_index) { return lhs_index < rhs_index; }
                if (lhs_size != rhs_size) { return lhs_size > rhs_size; }
            } else {
                if (lhs_size != rhs_size) { return lhs_size > rhs_size; }
                if (lhs_index != rhs_index) { return lhs_index < rhs_index; }
            }
            return false;
        });
        result.fields.insert(result.fields.end(), fields.begin(), fields.end());
    };
    append_fields(true);
    append_fields(false);
    for (auto i = 0u; i < result.fields.size(); i++) { result.fields[i].frame_index = 2u + i; }

    luisa::vector<CoroDesignatedFieldInfo> designated_fields;
    designated_fields.reserve(graph.designated_values.size());
    for (auto &&[name, value] : graph.designated_values) {
        designated_fields.emplace_back(CoroDesignatedFieldInfo{
            .name = name,
            .type = value->type(),
            .value = value});
    }
    std::stable_sort(designated_fields.begin(), designated_fields.end(), [&](auto &&lhs, auto &&rhs) noexcept {
        auto lhs_align = lhs.type != nullptr ? lhs.type->alignment() : 0u;
        auto rhs_align = rhs.type != nullptr ? rhs.type->alignment() : 0u;
        if (lhs_align != rhs_align) { return lhs_align < rhs_align; }
        auto lhs_index = detail::stable_value_index(stable_indices, lhs.value);
        auto rhs_index = detail::stable_value_index(stable_indices, rhs.value);
        if (lhs_index != rhs_index) { return lhs_index < rhs_index; }
        return lhs.name < rhs.name;
    });
    result.designated_fields = std::move(designated_fields);
    for (auto i = 0u; i < result.designated_fields.size(); i++) {
        result.designated_fields[i].frame_index = 2u + static_cast<uint32_t>(result.fields.size()) + i;
    }

    luisa::vector<const Type *> members;
    members.emplace_back(Type::of<uint3>());
    members.emplace_back(Type::of<uint>());
    for (auto &&field : result.fields) { members.emplace_back(field.type); }
    for (auto &&field : result.designated_fields) { members.emplace_back(field.type); }
    auto alignment = size_t{16u};
    for (auto &&field : result.fields) {
        if (field.type != nullptr) { alignment = std::max(alignment, field.type->size()); }
    }
    result.interface_type = Type::structure(alignment, members);
    return result;
}

namespace detail {

[[nodiscard]] static Constant *create_u32_constant(Module *module, uint32_t value) noexcept {
    return module->create_constant(Type::of<uint>(), &value);
}

[[nodiscard]] static Constant *create_bool_constant(Module *module, bool value) noexcept {
    return module->create_constant(Type::of<bool>(), &value);
}

[[nodiscard]] static Value *create_selected_value_constant(Module *module,
                                                           const Type *type,
                                                           int32_t value) noexcept {
    if (type->is_bool()) {
        auto v = value != 0;
        return module->create_constant(type, &v);
    }
    if (type->is_int8()) {
        auto v = static_cast<byte>(value);
        return module->create_constant(type, &v);
    }
    if (type->is_uint8()) {
        auto v = static_cast<ubyte>(value);
        return module->create_constant(type, &v);
    }
    if (type->is_int16()) {
        auto v = static_cast<short>(value);
        return module->create_constant(type, &v);
    }
    if (type->is_uint16()) {
        auto v = static_cast<ushort>(value);
        return module->create_constant(type, &v);
    }
    if (type->is_int32()) {
        auto v = static_cast<int>(value);
        return module->create_constant(type, &v);
    }
    if (type->is_uint32()) {
        auto v = static_cast<uint>(value);
        return module->create_constant(type, &v);
    }
    if (type->is_int64()) {
        auto v = static_cast<slong>(value);
        return module->create_constant(type, &v);
    }
    if (type->is_uint64()) {
        auto v = static_cast<ulong>(value);
        return module->create_constant(type, &v);
    }
    LUISA_ERROR_WITH_LOCATION("Unsupported condition replay type '{}'.", type->description());
}

[[nodiscard]] static luisa::string materialized_base_name(const CallableFunction *function) noexcept {
    if (auto name = function->name(); name.has_value() && !name->empty()) { return luisa::string{*name}; }
    return "anonymous";
}

[[nodiscard]] static Value *access_chain_lvalue(XIRBuilder &builder,
                                                Value *root,
                                                const Type *leaf_type,
                                                luisa::span<const uint32_t> chain) noexcept {
    if (chain.empty()) { return root; }
    auto module = builder.insertion_point()->parent_module();
    luisa::vector<Value *> indices;
    indices.reserve(chain.size());
    for (auto i : chain) { indices.emplace_back(create_u32_constant(module, i)); }
    return builder.gep(leaf_type, root, indices);
}

struct LoopTargets {
    BasicBlock *continue_target{nullptr};
    BasicBlock *break_target{nullptr};
};

class ScopeMaterializer;

class ScopeValueResolver final : public InstructionCloneValueResolver {

private:
    ScopeMaterializer &_materializer;
    XIRBuilder &_builder;

public:
    ScopeValueResolver(ScopeMaterializer &materializer, XIRBuilder &builder) noexcept
        : _materializer{materializer}, _builder{builder} {}
    [[nodiscard]] Value *resolve(const Value *value) noexcept override;
};

class ScopeMaterializer {

private:
    CallableFunction *_source;
    CallableFunction *_generated;
    const CoroGraph &_graph;
    const CoroTransitionGraph &_transition;
    const CoroFrame &_frame;
    CoroScopeRef _scope;
    luisa::optional<uint32_t> _token;
    XIRBuilder _entry_builder;
    Value *_frame_arg{nullptr};
    Value *_first_flag{nullptr};
    ReplayableValueAnalysis _replayable;
    bool _uses_coro_id{false};
    luisa::unordered_map<const Value *, Value *> _values;

private:
    [[nodiscard]] Module *module() const noexcept { return _generated->parent_module(); }
    void set_prologue_insertion_point(XIRBuilder &builder) noexcept {
        builder.set_insertion_point(_generated->body_block()->instructions().head_sentinel());
    }

    [[nodiscard]] static Argument *clone_argument_kind(Function *target, const Argument *source) noexcept {
        switch (source->derived_argument_tag()) {
            case DerivedArgumentTag::VALUE: return target->create_value_argument(source->type());
            case DerivedArgumentTag::REFERENCE: return target->create_reference_argument(source->type());
            case DerivedArgumentTag::RESOURCE: return target->create_resource_argument(source->type());
        }
        LUISA_ERROR_WITH_LOCATION("Unknown XIR argument kind.");
    }

    [[nodiscard]] Value *ensure_slot(const Value *value) noexcept {
        if (auto iter = _values.find(value); iter != _values.end()) { return iter->second; }
        LUISA_ASSERT(value != nullptr && value->type() != nullptr, "Invalid coroutine slot request.");
        if (value->isa<Argument>()) { return _values.at(value); }
        auto slot = static_cast<Value *>(_entry_builder.alloca_local(value->type()));
        if (!value->type()->is_resource() && !value->type()->is_custom()) {
            _entry_builder.store(slot, module()->create_constant_zero(value->type()));
        }
        _values.emplace(value, slot);
        return slot;
    }

    void define_value(const Value *old_value, Value *new_value, XIRBuilder &builder) noexcept {
        if (old_value == nullptr || old_value->type() == nullptr || new_value == nullptr) { return; }
        if (old_value->is_lvalue()) {
            _values[old_value] = new_value;
            return;
        }
        auto slot = ensure_slot(old_value);
        builder.store(slot, new_value);
    }

    [[nodiscard]] Value *read_coro_id(XIRBuilder &builder) const noexcept {
        auto ptr = builder.gep(Type::of<uint3>(), _frame_arg, {create_u32_constant(module(), 0u)});
        return builder.load(Type::of<uint3>(), ptr);
    }

    [[nodiscard]] Value *current_coro_token() const noexcept {
        return create_u32_constant(module(), _token.value_or(0u));
    }

    [[nodiscard]] Value *read_frame_field(XIRBuilder &builder, uint32_t index, const Type *type) const noexcept {
        auto ptr = builder.gep(type, _frame_arg, {create_u32_constant(module(), index)});
        return builder.load(type, ptr);
    }

    void write_frame_field(XIRBuilder &builder, uint32_t index, const Type *type, Value *value) const noexcept {
        auto ptr = builder.gep(type, _frame_arg, {create_u32_constant(module(), index)});
        builder.store(ptr, value);
    }

public:
    ScopeMaterializer(CallableFunction *source,
                      const CoroGraph &graph,
                      const CoroTransitionGraph &transition,
                      const CoroFrame &frame,
                      CoroScopeRef scope,
                      luisa::optional<uint32_t> token,
                      luisa::string_view name) noexcept
        : _source{source},
          _generated{source->parent_module()->create_callable(nullptr)},
          _graph{graph},
          _transition{transition},
          _frame{frame},
          _scope{scope},
          _token{token} {
        _generated->set_name(name);
        auto body = _generated->create_body_block();
        _entry_builder.set_insertion_point(body->instructions().head_sentinel());
        _frame_arg = _generated->create_reference_argument(frame.interface_type);
        _values.emplace(_frame_arg, _frame_arg);
        for (auto arg : source->arguments()) {
            auto mapped = clone_argument_kind(_generated, arg);
            _values.emplace(arg, mapped);
        }
    }

    [[nodiscard]] Value *ref_or_local(const Value *value, XIRBuilder &builder) noexcept;
    [[nodiscard]] Value *value_or_load(const Value *value, XIRBuilder &builder) noexcept;
    [[nodiscard]] bool uses_coro_id() const noexcept { return _uses_coro_id; }
    void resume_scope() noexcept;
    void suspend_scope(uint32_t target_token, XIRBuilder &builder) noexcept;
    void terminate_scope(XIRBuilder &builder) noexcept;
    void make_first_flag() noexcept;
    [[nodiscard]] bool emit_source_instruction(const Instruction *inst, XIRBuilder &builder) noexcept;
    [[nodiscard]] bool emit_sequence(const luisa::vector<CoroInstructionRef> &instructions,
                                     XIRBuilder &builder,
                                     const luisa::vector<LoopTargets> &loop_stack) noexcept;
    [[nodiscard]] CallableFunction *materialize() noexcept;
};

}// namespace detail

namespace detail {

Value *ScopeMaterializer::ref_or_local(const Value *value, XIRBuilder &builder) noexcept {
    if (value == nullptr) { return nullptr; }
    if (auto iter = _values.find(value); iter != _values.end()) { return iter->second; }
    if (value->isa<Argument>() || value->isa<AllocaInst>()) { return ensure_slot(value); }
    if (auto gep = value->isa<GEPInst>() ? static_cast<const GEPInst *>(value) : nullptr) {
        auto base = ref_or_local(gep->base(), builder);
        luisa::vector<Value *> indices;
        indices.reserve(gep->index_count());
        for (auto i = 0u; i < gep->index_count(); i++) { indices.emplace_back(value_or_load(gep->index(i), builder)); }
        return builder.gep(gep->type(), base, indices);
    }
    return ensure_slot(value);
}

Value *ScopeMaterializer::value_or_load(const Value *value, XIRBuilder &builder) noexcept {
    if (value == nullptr) { return nullptr; }
    if (value->isa<Constant>() || value->isa<Undefined>() || value->isa<Function>()) {
        return const_cast<Value *>(value);
    }
    if (auto sreg = value->isa<SpecialRegister>() ? static_cast<const SpecialRegister *>(value) : nullptr) {
        if (sreg->derived_special_register_tag() == DerivedSpecialRegisterTag::DISPATCH_ID) {
            _uses_coro_id = true;
            return read_coro_id(builder);
        }
        return const_cast<Value *>(value);
    }
    if (value->isa<CoroIdInst>()) {
        _uses_coro_id = true;
        return read_coro_id(builder);
    }
    if (value->isa<CoroTokenInst>()) { return current_coro_token(); }
    if (auto load = value->isa<LoadInst>() ? static_cast<const LoadInst *>(value) : nullptr) {
        return builder.load(load->type(), ref_or_local(load->variable(), builder));
    }
    if (value->is_lvalue()) { return ref_or_local(value, builder); }
    if (auto iter = _values.find(value); iter != _values.end()) {
        auto mapped = iter->second;
        return mapped->is_lvalue() ? static_cast<Value *>(builder.load(value->type(), mapped)) : mapped;
    }
    if (_replayable.detect(value)) {
        auto inst = value->isa<Instruction>() ? static_cast<const Instruction *>(value) : nullptr;
        LUISA_ASSERT(inst != nullptr, "Unexpected replayable non-instruction value.");
        ScopeValueResolver resolver{*this, builder};
        return inst->clone_with_metadata(builder, resolver);
    }
    auto slot = ensure_slot(value);
    return slot->is_lvalue() ? static_cast<Value *>(builder.load(value->type(), slot)) : slot;
}

void ScopeMaterializer::resume_scope() noexcept {
    auto node_iter = _transition.nodes.find(_scope);
    if (node_iter == _transition.nodes.end()) { return; }
    AccessSet load_set{node_iter->second.union_states_to_load};
    for (auto &&field : _frame.fields) {
        if (!load_set.contains(field.root, to_signed_chain(field.chain))) { continue; }
        auto slot = ensure_slot(field.root);
        auto value = read_frame_field(_entry_builder, field.frame_index, field.type);
        auto target = access_chain_lvalue(_entry_builder, slot, field.type, field.chain);
        _entry_builder.store(target, value);
    }
    for (auto &&field : _frame.designated_fields) {
        if (!load_set.contains(field.value, {})) { continue; }
        auto slot = ensure_slot(field.value);
        auto value = read_frame_field(_entry_builder, field.frame_index, field.type);
        _entry_builder.store(slot, value);
    }
}

void ScopeMaterializer::suspend_scope(uint32_t target_token, XIRBuilder &builder) noexcept {
    auto edge = _transition.nodes.at(_scope).outlets.at(target_token);
    AccessSet save_set{edge.states_to_save};
    write_frame_field(builder, 1u, Type::of<uint>(), create_u32_constant(module(), target_token));
    for (auto &&field : _frame.fields) {
        if (!save_set.contains(field.root, to_signed_chain(field.chain))) { continue; }
        auto slot = ensure_slot(field.root);
        auto source = access_chain_lvalue(builder, slot, field.type, field.chain);
        auto value = source->is_lvalue() ? static_cast<Value *>(builder.load(field.type, source)) : source;
        write_frame_field(builder, field.frame_index, field.type, value);
    }
    for (auto &&field : _frame.designated_fields) {
        if (!save_set.contains(field.value, {})) { continue; }
        auto slot = ensure_slot(field.value);
        auto value = slot->is_lvalue() ? static_cast<Value *>(builder.load(field.type, slot)) : slot;
        write_frame_field(builder, field.frame_index, field.type, value);
    }
    builder.return_void();
}

void ScopeMaterializer::terminate_scope(XIRBuilder &builder) noexcept {
    constexpr auto terminate_token = 0x8000'0000u;
    write_frame_field(builder, 1u, Type::of<uint>(), create_u32_constant(module(), terminate_token));
    builder.return_void();
}

void ScopeMaterializer::make_first_flag() noexcept {
    if (_first_flag != nullptr) { return; }
    _first_flag = _entry_builder.alloca_local(Type::of<bool>());
    _entry_builder.store(_first_flag, create_bool_constant(module(), false));
}

bool ScopeMaterializer::emit_source_instruction(const Instruction *inst, XIRBuilder &builder) noexcept {
    switch (inst->derived_instruction_tag()) {
        case DerivedInstructionTag::ALLOCA:
            static_cast<void>(ensure_slot(inst));
            return false;
        case DerivedInstructionTag::CORO_ID:
            _uses_coro_id = true;
            define_value(inst, read_coro_id(builder), builder);
            return false;
        case DerivedInstructionTag::CORO_TOKEN:
            define_value(inst, current_coro_token(), builder);
            return false;
        case DerivedInstructionTag::CORO_REGISTER:
            return false;
        default: break;
    }
    ScopeValueResolver resolver{*this, builder};
    auto cloned = inst->clone_with_metadata(builder, resolver);
    if (inst->type() != nullptr) { define_value(inst, cloned, builder); }
    return cloned->is_terminator();
}

bool ScopeMaterializer::emit_sequence(const luisa::vector<CoroInstructionRef> &instructions,
                                      XIRBuilder &builder,
                                      const luisa::vector<LoopTargets> &loop_stack) noexcept {
    for (auto ref : instructions) {
        auto &&instr = _graph.instructions.at(ref);
        switch (instr.tag) {
            case CoroInstructionTag::SIMPLE:
                if (instr.simple != nullptr && emit_source_instruction(instr.simple, builder)) { return true; }
                break;
            case CoroInstructionTag::CONDITION_STACK_REPLAY:
                for (auto &&item : instr.condition_stack) {
                    define_value(item.value,
                                 create_selected_value_constant(module(), item.value->type(), item.selected_value),
                                 builder);
                }
                break;
            case CoroInstructionTag::MAKE_FIRST_FLAG:
                make_first_flag();
                break;
            case CoroInstructionTag::SKIP_IF_FIRST_FLAG: {
                LUISA_ASSERT(_first_flag != nullptr, "Coroutine first flag must be initialized before use.");
                auto if_inst = builder.if_(static_cast<Value *>(builder.load(Type::of<bool>(), _first_flag)));
                auto true_block = if_inst->create_true_block();
                auto false_block = if_inst->create_false_block();
                auto merge_block = if_inst->create_merge_block();
                XIRBuilder true_builder;
                true_builder.set_insertion_point(true_block);
                if (!emit_sequence(instr.body, true_builder, loop_stack)) { true_builder.br(merge_block); }
                XIRBuilder false_builder;
                false_builder.set_insertion_point(false_block);
                false_builder.br(merge_block);
                builder.set_insertion_point(merge_block);
                break;
            }
            case CoroInstructionTag::CLEAR_FIRST_FLAG:
                builder.store(_first_flag, create_bool_constant(module(), true));
                break;
            case CoroInstructionTag::IF: {
                auto if_inst = builder.if_(value_or_load(instr.condition, builder));
                auto true_block = if_inst->create_true_block();
                auto false_block = if_inst->create_false_block();
                auto merge_block = if_inst->create_merge_block();
                XIRBuilder true_builder;
                true_builder.set_insertion_point(true_block);
                if (!emit_sequence(instr.true_branch, true_builder, loop_stack)) { true_builder.br(merge_block); }
                XIRBuilder false_builder;
                false_builder.set_insertion_point(false_block);
                if (!emit_sequence(instr.false_branch, false_builder, loop_stack)) { false_builder.br(merge_block); }
                builder.set_insertion_point(merge_block);
                break;
            }
            case CoroInstructionTag::SWITCH: {
                auto switch_inst = builder.switch_(value_or_load(instr.condition, builder));
                auto merge_block = switch_inst->create_merge_block();
                for (auto &&c : instr.cases) {
                    auto case_block = switch_inst->create_case_block(c.value);
                    XIRBuilder case_builder;
                    case_builder.set_insertion_point(case_block);
                    if (!emit_sequence(c.body, case_builder, loop_stack)) { case_builder.br(merge_block); }
                }
                auto default_block = switch_inst->create_default_block();
                XIRBuilder default_builder;
                default_builder.set_insertion_point(default_block);
                if (!emit_sequence(instr.default_body, default_builder, loop_stack)) { default_builder.br(merge_block); }
                builder.set_insertion_point(merge_block);
                break;
            }
            case CoroInstructionTag::SIMPLE_LOOP: {
                auto loop_inst = builder.simple_loop();
                auto body_block = loop_inst->create_body_block();
                auto merge_block = loop_inst->create_merge_block();
                auto nested_stack = loop_stack;
                nested_stack.emplace_back(LoopTargets{.continue_target = body_block, .break_target = merge_block});
                XIRBuilder body_builder;
                body_builder.set_insertion_point(body_block);
                if (!emit_sequence(instr.body, body_builder, nested_stack)) { body_builder.br(body_block); }
                builder.set_insertion_point(merge_block);
                break;
            }
            case CoroInstructionTag::LOOP_CONTINUE:
                LUISA_ASSERT(!loop_stack.empty(), "Loop continue must appear inside a loop.");
                builder.continue_(loop_stack.back().continue_target);
                return true;
            case CoroInstructionTag::LOOP_BREAK:
                LUISA_ASSERT(!loop_stack.empty(), "Loop break must appear inside a loop.");
                builder.break_(loop_stack.back().break_target);
                return true;
            case CoroInstructionTag::SUSPEND:
                suspend_scope(instr.token, builder);
                return true;
            case CoroInstructionTag::TERMINATE:
                terminate_scope(builder);
                return true;
            default: break;
        }
    }
    return false;
}

CallableFunction *ScopeMaterializer::materialize() noexcept {
    if (_token.has_value()) { resume_scope(); }
    XIRBuilder body_builder;
    body_builder.set_insertion_point(_generated->body_block());
    if (!emit_sequence(_graph.scopes.at(_scope).instructions, body_builder, {})) { body_builder.return_void(); }
    return _generated;
}

Value *ScopeValueResolver::resolve(const Value *value) noexcept {
    if (value == nullptr) { return nullptr; }
    if (value->isa<Function>() || value->isa<Constant>() || value->isa<Undefined>()) {
        return const_cast<Value *>(value);
    }
    return value->is_lvalue() ? _materializer.ref_or_local(value, _builder) :
                                _materializer.value_or_load(value, _builder);
}

}// namespace detail

MaterializeCoroResult materialize_coro_pass_run_on_function(CallableFunction *function) noexcept {
    static_cast<void>(Canoinicalize_Control_Flow_pass_run_on_Function(function));
    auto graph = compute_coro_graph(function);
    detail::dump_coro_graph_if_requested(graph);
    auto use_def = compute_coro_use_def(graph);
    auto transition = compute_coro_transition_graph(graph, use_def);
    auto frame = compute_coro_frame(graph, transition);
    detail::dump_coro_analysis(graph, use_def, transition, frame);
    auto base_name = detail::materialized_base_name(function);

    MaterializeCoroResult result;
    {
        auto entry_name = luisa::string{base_name}.append(".coro.entry");
        auto entry_materializer = detail::ScopeMaterializer{
            function, graph, transition, frame, graph.entry, {}, entry_name};
        result.entry = entry_materializer.materialize();
        auto node_iter = transition.nodes.find(graph.entry);
        result.entry_input_fields = node_iter != transition.nodes.end() ?
                                        detail::collect_frame_indices(frame,
                                                                     node_iter->second.union_states_to_load,
                                                                     entry_materializer.uses_coro_id()) :
                                        luisa::vector<uint32_t>{1u};
        result.entry_output_fields = node_iter != transition.nodes.end() ?
                                         detail::collect_frame_indices(frame,
                                                                      node_iter->second.union_states_to_save,
                                                                      false) :
                                         luisa::vector<uint32_t>{1u};
        if (node_iter != transition.nodes.end()) {
            for (auto &&[target_token, _] : node_iter->second.outlets) { result.entry_target_tokens.emplace_back(target_token); }
        }
    }
    result.frame_interface_type = frame.interface_type;
    result.frame_fields = frame.fields;
    result.designated_fields = frame.designated_fields;
    result.named_tokens = detail::collect_named_tokens(function);

    for (auto &&[token, scope_ref] : graph.tokens) {
        auto resume_name = luisa::string{base_name}.append(".coro.resume.").append(std::to_string(token));
        auto materializer = detail::ScopeMaterializer{
            function, graph, transition, frame, scope_ref, token, resume_name};
        auto generated = materializer.materialize();
        auto node_iter = transition.nodes.find(scope_ref);
        auto input_fields = node_iter != transition.nodes.end() ?
                                detail::collect_frame_indices(frame,
                                                             node_iter->second.union_states_to_load,
                                                             materializer.uses_coro_id()) :
                                luisa::vector<uint32_t>{1u};
        auto output_fields = node_iter != transition.nodes.end() ?
                                 detail::collect_frame_indices(frame,
                                                              node_iter->second.union_states_to_save,
                                                              false) :
                                 luisa::vector<uint32_t>{1u};
        luisa::vector<uint32_t> target_tokens;
        if (node_iter != transition.nodes.end()) {
            for (auto &&[target_token, _] : node_iter->second.outlets) { target_tokens.emplace_back(target_token); }
        }
        result.scopes.emplace_back(MaterializedCoroScope{
            .token = token,
            .function = generated,
            .input_fields = std::move(input_fields),
            .output_fields = std::move(output_fields),
            .target_tokens = std::move(target_tokens)});
    }
    return result;
}

} // namespace luisa::compute::xir
