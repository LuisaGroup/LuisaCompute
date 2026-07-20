#include <mutex>

#include <luisa/ast/type.h>
#include <luisa/xir/argument.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/passes/scalar_evolution.h>
#include <luisa/xir/special_register.h>
#include <luisa/xir/undefined.h>

namespace luisa::compute::xir {

SCEVUnknown::SCEVUnknown(Value *value) noexcept : _value{value} {}
const Type *SCEVUnknown::type() const noexcept { return _value == nullptr ? nullptr : _value->type(); }

SCEVConstant::SCEVConstant(Constant *c) noexcept : _constant{c} {}
const Type *SCEVConstant::type() const noexcept { return _constant == nullptr ? nullptr : _constant->type(); }

SCEVAddRec::SCEVAddRec(const SCEV *start, const SCEV *stride, LoopInst *loop) noexcept
    : _start{start}, _stride{stride}, _loop{loop} {}
const Type *SCEVAddRec::type() const noexcept { return _start == nullptr ? nullptr : _start->type(); }

SCEVAddExpr::SCEVAddExpr(luisa::vector<const SCEV *> ops) noexcept : _operands{std::move(ops)} {}
const Type *SCEVAddExpr::type() const noexcept { return _operands.empty() ? nullptr : _operands.front()->type(); }

SCEVMulExpr::SCEVMulExpr(luisa::vector<const SCEV *> ops) noexcept : _operands{std::move(ops)} {}
const Type *SCEVMulExpr::type() const noexcept { return _operands.empty() ? nullptr : _operands.front()->type(); }

struct SCEVAnalysis::Impl {
    struct InstructionSnapshot {
        Instruction *inst;
        BasicBlock *parent;
        const Type *type;
        DerivedInstructionTag tag;
        luisa::vector<Value *> operands;
        luisa::vector<BasicBlock *> phi_blocks;
        ArithmeticOp arithmetic_op{};
        BasicBlock *loop_body{nullptr};
        BasicBlock *loop_update{nullptr};
        BasicBlock *merge{nullptr};
    };

    FunctionDefinition *def{nullptr};
    luisa::vector<luisa::unique_ptr<SCEV>> allocated;
    luisa::unordered_map<Value *, const SCEV *> cache;
    luisa::unordered_map<Instruction *, const SCEV *> results;
    luisa::unordered_set<Value *> active;
    luisa::unordered_set<Value *> cyclic;
    luisa::unordered_set<BasicBlock *> owned_blocks;
    luisa::unordered_set<BasicBlock *> reachable_blocks;
    luisa::unordered_map<BasicBlock *, luisa::unordered_set<BasicBlock *>> predecessors;
    luisa::unordered_set<BasicBlock *> loop_blocks;
    luisa::vector<BasicBlock *> ordered_blocks;
    LoopInst *current_loop{nullptr};
    SCEVInfo info;
    BasicBlock *snapshot_body{nullptr};
    luisa::weak_ptr<uint8_t> lifetime_token;
    bool snapshot_valid{false};
    luisa::vector<Argument *> snapshot_arguments;
    luisa::vector<BasicBlock *> snapshot_blocks;
    luisa::vector<InstructionSnapshot> snapshot_instructions;

    void clear() noexcept {
        def = nullptr;
        allocated.clear();
        cache.clear();
        results.clear();
        active.clear();
        cyclic.clear();
        owned_blocks.clear();
        reachable_blocks.clear();
        predecessors.clear();
        loop_blocks.clear();
        ordered_blocks.clear();
        current_loop = nullptr;
        info = {};
        snapshot_body = nullptr;
        lifetime_token.reset();
        snapshot_valid = false;
        snapshot_arguments.clear();
        snapshot_blocks.clear();
        snapshot_instructions.clear();
    }

    void capture_snapshot() noexcept {
        lifetime_token = def->lifetime_token();
        snapshot_body = def->body_block();
        for (auto *argument : def->arguments()) { snapshot_arguments.emplace_back(argument); }
        for (auto *block : def->basic_blocks()) {
            snapshot_blocks.emplace_back(block);
            for (auto *inst : block->instructions()) {
                InstructionSnapshot snapshot{
                    .inst = inst,
                    .parent = inst->parent_block(),
                    .type = inst->type(),
                    .tag = inst->derived_instruction_tag()};
                for (size_t i = 0u; i < inst->operand_count(); ++i) {
                    snapshot.operands.emplace_back(inst->operand(i));
                }
                if (inst->isa<PhiInst>()) {
                    auto *phi = static_cast<PhiInst *>(inst);
                    for (auto *incoming_block : phi->incoming_blocks()) {
                        snapshot.phi_blocks.emplace_back(incoming_block);
                    }
                }
                if (inst->isa<ArithmeticInst>()) {
                    snapshot.arithmetic_op = static_cast<ArithmeticInst *>(inst)->op();
                }
                if (inst->isa<LoopInst>()) {
                    auto *loop = static_cast<LoopInst *>(inst);
                    snapshot.loop_body = loop->body_block();
                    snapshot.loop_update = loop->update_block();
                }
                if (auto *merge = inst->control_flow_merge()) {
                    snapshot.merge = merge->merge_block();
                }
                snapshot_instructions.emplace_back(std::move(snapshot));
            }
        }
        snapshot_valid = true;
    }

    [[nodiscard]] bool is_current() const noexcept {
        if (!snapshot_valid || def == nullptr || lifetime_token.expired() ||
            def->body_block() != snapshot_body) {
            return false;
        }
        size_t argument_index = 0u;
        for (auto *argument : def->arguments()) {
            if (argument_index >= snapshot_arguments.size() || snapshot_arguments[argument_index] != argument) { return false; }
            ++argument_index;
        }
        if (argument_index != snapshot_arguments.size()) { return false; }
        size_t block_index = 0u;
        size_t instruction_index = 0u;
        for (auto *block : def->basic_blocks()) {
            if (block_index >= snapshot_blocks.size() || snapshot_blocks[block_index] != block) { return false; }
            ++block_index;
            for (auto *inst : block->instructions()) {
                if (instruction_index >= snapshot_instructions.size()) { return false; }
                auto &snapshot = snapshot_instructions[instruction_index++];
                if (snapshot.inst != inst || snapshot.parent != inst->parent_block() ||
                    snapshot.type != inst->type() || snapshot.tag != inst->derived_instruction_tag() ||
                    snapshot.operands.size() != inst->operand_count()) {
                    return false;
                }
                for (size_t i = 0u; i < inst->operand_count(); ++i) {
                    if (snapshot.operands[i] != inst->operand(i)) { return false; }
                }
                if (inst->isa<PhiInst>()) {
                    auto blocks = static_cast<PhiInst *>(inst)->incoming_blocks();
                    if (snapshot.phi_blocks.size() != blocks.size()) { return false; }
                    for (size_t i = 0u; i < blocks.size(); ++i) {
                        if (snapshot.phi_blocks[i] != blocks[i]) { return false; }
                    }
                } else if (!snapshot.phi_blocks.empty()) {
                    return false;
                }
                if (inst->isa<ArithmeticInst>() &&
                    snapshot.arithmetic_op != static_cast<ArithmeticInst *>(inst)->op()) {
                    return false;
                }
                if (inst->isa<LoopInst>()) {
                    auto *loop = static_cast<LoopInst *>(inst);
                    if (snapshot.loop_body != loop->body_block() || snapshot.loop_update != loop->update_block()) {
                        return false;
                    }
                }
                auto *merge = inst->control_flow_merge();
                if (snapshot.merge != (merge == nullptr ? nullptr : merge->merge_block())) { return false; }
            }
        }
        return block_index == snapshot_blocks.size() && instruction_index == snapshot_instructions.size();
    }

    [[nodiscard]] const SCEV *make_unknown(Value *value) noexcept {
        auto scev = luisa::make_unique<SCEVUnknown>(value);
        auto *ptr = scev.get();
        allocated.emplace_back(std::move(scev));
        return ptr;
    }

    [[nodiscard]] bool initialize_function(FunctionDefinition *function) noexcept {
        def = function;
        if (def == nullptr || def->body_block() == nullptr) { return false; }
        for (auto *block : def->basic_blocks()) {
            if (block == nullptr || block->parent_function() != def) { return false; }
            owned_blocks.emplace(block);
            ordered_blocks.emplace_back(block);
        }
        if (!owned_blocks.contains(def->body_block())) { return false; }
        luisa::vector<BasicBlock *> worklist{def->body_block()};
        reachable_blocks.emplace(def->body_block());
        while (!worklist.empty()) {
            auto *block = worklist.back();
            worklist.pop_back();
            if (!block->is_terminated()) { return false; }
            auto *terminator = block->terminator();
            for (auto *inst : block->instructions()) {
                if (inst == nullptr || inst->parent_block() != block ||
                    (inst->is_terminator() && inst != terminator)) {
                    return false;
                }
            }
            for (auto *use : terminator->operand_uses()) {
                auto *value = use == nullptr ? nullptr : use->value();
                if (value == nullptr || !value->isa<BasicBlock>()) { continue; }
                auto *successor = static_cast<BasicBlock *>(value);
                if (!owned_blocks.contains(successor)) { return false; }
                predecessors[successor].emplace(block);
                if (reachable_blocks.emplace(successor).second) {
                    worklist.emplace_back(successor);
                }
            }
        }
        return true;
    }

    [[nodiscard]] bool value_belongs_to_function(Value *value) const noexcept {
        if (value == nullptr || def == nullptr) { return false; }
        switch (value->derived_value_tag()) {
            case DerivedValueTag::CONSTANT:
                return static_cast<Constant *>(value)->parent_module() == def->parent_module();
            case DerivedValueTag::SPECIAL_REGISTER:
                return static_cast<SpecialRegister *>(value)->parent_module() == def->parent_module();
            case DerivedValueTag::UNDEFINED:
                return static_cast<Undefined *>(value)->parent_module() == def->parent_module();
            case DerivedValueTag::ARGUMENT:
                return static_cast<Argument *>(value)->parent_function() == def;
            case DerivedValueTag::INSTRUCTION: {
                auto *inst = static_cast<Instruction *>(value);
                auto *block = inst->parent_block();
                return block != nullptr && owned_blocks.contains(block) && reachable_blocks.contains(block);
            }
            default: return false;
        }
    }

    [[nodiscard]] const SCEV *get_scev(Value *value) noexcept {
        if (value == nullptr || !value_belongs_to_function(value)) { return nullptr; }
        if (auto iter = cache.find(value); iter != cache.end()) { return iter->second; }
        if (!active.emplace(value).second) {
            cyclic.emplace(value);
            return make_unknown(value);
        }
        const SCEV *result = nullptr;
        if (value->isa<Constant>()) {
            auto scev = luisa::make_unique<SCEVConstant>(static_cast<Constant *>(value));
            result = scev.get();
            allocated.emplace_back(std::move(scev));
        } else if (!value->isa<Instruction>()) {
            result = make_unknown(value);
        } else {
            result = build_scev(static_cast<Instruction *>(value));
        }
        active.erase(value);
        if (cyclic.erase(value) != 0u) { result = make_unknown(value); }
        cache.emplace(value, result);
        return result;
    }

    [[nodiscard]] const SCEV *build_scev(Instruction *inst) noexcept {
        if (inst->isa<PhiInst>()) { return build_phi_scev(static_cast<PhiInst *>(inst)); }
        if (inst->isa<ArithmeticInst>()) { return build_arithmetic_scev(static_cast<ArithmeticInst *>(inst)); }
        return make_unknown(inst);
    }

    [[nodiscard]] bool is_loop_invariant(Value *value) noexcept {
        luisa::unordered_set<Value *> visited;
        return is_loop_invariant(value, visited);
    }

    [[nodiscard]] bool is_loop_invariant(Value *value, luisa::unordered_set<Value *> &visited) noexcept {
        if (value == nullptr || !value_belongs_to_function(value)) { return false; }
        if (value->isa<Constant>()) { return true; }
        if (value->isa<Argument>() || value->isa<SpecialRegister>()) { return true; }
        if (!value->isa<Instruction>()) { return false; }
        auto *inst = static_cast<Instruction *>(value);
        auto *block = inst->parent_block();
        if (!loop_blocks.contains(block)) { return true; }
        switch (inst->derived_instruction_tag()) {
            case DerivedInstructionTag::ARITHMETIC:
            case DerivedInstructionTag::CAST:
            case DerivedInstructionTag::GEP: break;
            default: return false;
        }
        if (!visited.emplace(value).second) { return false; }
        for (size_t i = 0u; i < inst->operand_count(); ++i) {
            if (!is_loop_invariant(inst->operand(i), visited)) {
                visited.erase(value);
                return false;
            }
        }
        visited.erase(value);
        return true;
    }

    [[nodiscard]] const SCEV *build_phi_scev(PhiInst *phi) noexcept {
        if (current_loop == nullptr || phi->parent_block() != current_loop->prepare_block() ||
            phi->incoming_count() != 2u || phi->type() == nullptr) {
            return make_unknown(phi);
        }
        auto *preheader = current_loop->parent_block();
        auto *update = current_loop->update_block();
        Value *start_value = nullptr;
        Value *recur_value = nullptr;
        for (size_t i = 0u; i < phi->incoming_count(); ++i) {
            auto incoming = phi->incoming(i);
            if (incoming.value == nullptr || incoming.block == nullptr ||
                !owned_blocks.contains(incoming.block) ||
                !predecessors[phi->parent_block()].contains(incoming.block)) {
                return make_unknown(phi);
            }
            if (incoming.block == preheader) {
                if (start_value != nullptr) { return make_unknown(phi); }
                start_value = incoming.value;
            } else if (incoming.block == update) {
                if (recur_value != nullptr) { return make_unknown(phi); }
                recur_value = incoming.value;
            } else {
                return make_unknown(phi);
            }
        }
        if (start_value == nullptr || recur_value == nullptr || start_value == phi ||
            start_value->type() != phi->type() || recur_value->type() != phi->type() ||
            !recur_value->isa<ArithmeticInst>()) {
            return make_unknown(phi);
        }
        auto *arithmetic = static_cast<ArithmeticInst *>(recur_value);
        if (arithmetic->op() != ArithmeticOp::BINARY_ADD || arithmetic->operand_count() != 2u) {
            return make_unknown(phi);
        }
        Value *stride_value = nullptr;
        size_t phi_operand_count = 0u;
        for (size_t i = 0u; i < arithmetic->operand_count(); ++i) {
            auto *operand = arithmetic->operand(i);
            if (operand == phi) {
                ++phi_operand_count;
            } else {
                stride_value = operand;
            }
        }
        if (phi_operand_count != 1u || stride_value == nullptr ||
            stride_value->type() != phi->type() || !is_loop_invariant(stride_value)) {
            return make_unknown(phi);
        }
        auto *start = get_scev(start_value);
        auto *stride = get_scev(stride_value);
        if (start == nullptr || stride == nullptr || start->type() != phi->type() || stride->type() != phi->type()) {
            return make_unknown(phi);
        }
        auto scev = luisa::make_unique<SCEVAddRec>(start, stride, current_loop);
        auto *ptr = scev.get();
        allocated.emplace_back(std::move(scev));
        return simplify(ptr);
    }

    [[nodiscard]] const SCEV *build_arithmetic_scev(ArithmeticInst *inst) noexcept {
        auto op = inst->op();
        if ((op != ArithmeticOp::BINARY_ADD && op != ArithmeticOp::BINARY_MUL) ||
            inst->operand_count() != 2u || inst->type() == nullptr) {
            return make_unknown(inst);
        }
        luisa::vector<const SCEV *> operands;
        for (size_t i = 0u; i < inst->operand_count(); ++i) {
            auto *operand = inst->operand(i);
            if (operand == nullptr || operand->type() != inst->type()) { return make_unknown(inst); }
            auto *scev = get_scev(operand);
            if (scev == nullptr || scev->type() != inst->type()) { return make_unknown(inst); }
            auto expected_kind = op == ArithmeticOp::BINARY_ADD ? SCEV::Kind::ADD : SCEV::Kind::MUL;
            if (scev->kind() == expected_kind) {
                auto nested = expected_kind == SCEV::Kind::ADD ?
                                  static_cast<const SCEVAddExpr *>(scev)->operands() :
                                  static_cast<const SCEVMulExpr *>(scev)->operands();
                for (auto *nested_operand : nested) { operands.emplace_back(nested_operand); }
            } else {
                operands.emplace_back(scev);
            }
        }
        if (op == ArithmeticOp::BINARY_ADD) {
            auto scev = luisa::make_unique<SCEVAddExpr>(std::move(operands));
            auto *ptr = scev.get();
            allocated.emplace_back(std::move(scev));
            return simplify(ptr);
        }
        auto scev = luisa::make_unique<SCEVMulExpr>(std::move(operands));
        auto *ptr = scev.get();
        allocated.emplace_back(std::move(scev));
        return simplify(ptr);
    }

    [[nodiscard]] static bool constant_is_zero(const SCEVConstant *scev) noexcept {
        auto *constant = scev == nullptr ? nullptr : scev->constant();
        auto *type = constant == nullptr ? nullptr : constant->type();
        if (type == nullptr) { return false; }
        if (type->is_int8()) { return constant->as<int8_t>() == 0; }
        if (type->is_uint8()) { return constant->as<uint8_t>() == 0u; }
        if (type->is_int16()) { return constant->as<int16_t>() == 0; }
        if (type->is_uint16()) { return constant->as<uint16_t>() == 0u; }
        if (type->is_int32()) { return constant->as<int32_t>() == 0; }
        if (type->is_uint32()) { return constant->as<uint32_t>() == 0u; }
        if (type->is_int64()) { return constant->as<int64_t>() == 0; }
        if (type->is_uint64()) { return constant->as<uint64_t>() == 0u; }
        return false;
    }

    [[nodiscard]] static bool constant_is_one(const SCEVConstant *scev) noexcept {
        auto *constant = scev == nullptr ? nullptr : scev->constant();
        auto *type = constant == nullptr ? nullptr : constant->type();
        if (type == nullptr) { return false; }
        if (type->is_int8()) { return constant->as<int8_t>() == 1; }
        if (type->is_uint8()) { return constant->as<uint8_t>() == 1u; }
        if (type->is_int16()) { return constant->as<int16_t>() == 1; }
        if (type->is_uint16()) { return constant->as<uint16_t>() == 1u; }
        if (type->is_int32()) { return constant->as<int32_t>() == 1; }
        if (type->is_uint32()) { return constant->as<uint32_t>() == 1u; }
        if (type->is_int64()) { return constant->as<int64_t>() == 1; }
        if (type->is_uint64()) { return constant->as<uint64_t>() == 1u; }
        if (type->is_float32()) { return constant->as<float>() == 1.0f; }
        if (type->is_float64()) { return constant->as<double>() == 1.0; }
        return false;
    }

    [[nodiscard]] const SCEV *simplify(const SCEV *scev) noexcept {
        if (scev == nullptr) { return nullptr; }
        if (scev->kind() == SCEV::Kind::ADD) {
            auto operands = static_cast<const SCEVAddExpr *>(scev)->operands();
            return operands.size() == 1u ? operands.front() : scev;
        }
        if (scev->kind() == SCEV::Kind::MUL) {
            auto operands = static_cast<const SCEVMulExpr *>(scev)->operands();
            luisa::vector<const SCEV *> filtered;
            for (auto *operand : operands) {
                if (operand->kind() == SCEV::Kind::CONSTANT) {
                    auto *constant = static_cast<const SCEVConstant *>(operand);
                    if (constant_is_zero(constant)) { return operand; }
                    if (constant_is_one(constant)) { continue; }
                }
                filtered.emplace_back(operand);
            }
            if (filtered.empty()) { return operands.empty() ? scev : operands.front(); }
            if (filtered.size() == 1u) { return filtered.front(); }
            if (filtered.size() != operands.size()) {
                auto replacement = luisa::make_unique<SCEVMulExpr>(std::move(filtered));
                auto *ptr = replacement.get();
                allocated.emplace_back(std::move(replacement));
                return ptr;
            }
            return scev;
        }
        if (scev->kind() == SCEV::Kind::ADD_REC) {
            auto *recurrence = static_cast<const SCEVAddRec *>(scev);
            if (recurrence->stride() != nullptr && recurrence->stride()->kind() == SCEV::Kind::CONSTANT &&
                constant_is_zero(static_cast<const SCEVConstant *>(recurrence->stride()))) {
                return recurrence->start();
            }
        }
        return scev;
    }

    [[nodiscard]] bool collect_loop_blocks(LoopInst *loop) noexcept {
        loop_blocks.clear();
        auto *parent = loop == nullptr ? nullptr : loop->parent_block();
        auto *prepare = loop == nullptr ? nullptr : loop->prepare_block();
        auto *body = loop == nullptr ? nullptr : loop->body_block();
        auto *update = loop == nullptr ? nullptr : loop->update_block();
        auto *merge = loop == nullptr ? nullptr : loop->merge_block();
        if (parent == nullptr || prepare == nullptr || body == nullptr || update == nullptr || merge == nullptr ||
            !reachable_blocks.contains(parent) || !reachable_blocks.contains(prepare) ||
            !reachable_blocks.contains(body) || !reachable_blocks.contains(update) ||
            !reachable_blocks.contains(merge) || parent == prepare || parent == body ||
            parent == update || parent == merge || prepare == body || prepare == update ||
            prepare == merge || body == update || body == merge || update == merge ||
            !predecessors[prepare].contains(parent) || !predecessors[prepare].contains(update)) {
            return false;
        }
        luisa::vector<BasicBlock *> worklist{prepare, body, update};
        loop_blocks.emplace(prepare);
        loop_blocks.emplace(body);
        loop_blocks.emplace(update);
        luisa::unordered_set<BasicBlock *> processed;
        while (!worklist.empty()) {
            auto *block = worklist.back();
            worklist.pop_back();
            if (!processed.emplace(block).second) { continue; }
            if (!block->is_terminated()) { return false; }
            for (auto *use : block->terminator()->operand_uses()) {
                auto *value = use == nullptr ? nullptr : use->value();
                if (value == nullptr || !value->isa<BasicBlock>()) { continue; }
                auto *successor = static_cast<BasicBlock *>(value);
                if (!owned_blocks.contains(successor)) { return false; }
                if (successor == merge) { continue; }
                if (successor == parent) { return false; }
                if (loop_blocks.emplace(successor).second) {
                    worklist.emplace_back(successor);
                }
            }
        }
        return true;
    }

    [[nodiscard]] bool analyze_loop(LoopInst *loop) noexcept {
        if (!collect_loop_blocks(loop)) { return false; }
        current_loop = loop;
        cache.clear();
        active.clear();
        cyclic.clear();
        for (auto *block : ordered_blocks) {
            if (!loop_blocks.contains(block)) { continue; }
            for (auto *inst : block->instructions()) {
                static_cast<void>(get_scev(inst));
            }
        }
        for (auto &&[value, scev] : cache) {
            if (value->isa<Instruction>()) {
                results[static_cast<Instruction *>(value)] = scev;
            }
        }
        current_loop = nullptr;
        return true;
    }

    [[nodiscard]] SCEVInfo run(FunctionDefinition *function) noexcept {
        clear();
        if (!initialize_function(function)) {
            info.invalid_function_count = 1u;
            return info;
        }
        for (auto *block : ordered_blocks) {
            if (!reachable_blocks.contains(block) || !block->is_terminated()) { continue; }
            auto *terminator = block->terminator();
            if (!terminator->isa<LoopInst>()) { continue; }
            if (analyze_loop(static_cast<LoopInst *>(terminator))) {
                ++info.analyzed_loop_count;
            } else {
                ++info.rejected_loop_count;
            }
        }
        capture_snapshot();
        return info;
    }
};

SCEVAnalysis::SCEVAnalysis() noexcept : _impl{luisa::make_unique<Impl>()} {}
SCEVAnalysis::~SCEVAnalysis() noexcept = default;
SCEVAnalysis::SCEVAnalysis(SCEVAnalysis &&) noexcept = default;
SCEVAnalysis &SCEVAnalysis::operator=(SCEVAnalysis &&) noexcept = default;

void SCEVAnalysis::clear() noexcept {
    if (_impl != nullptr) { _impl->clear(); }
}

SCEVInfo SCEVAnalysis::analyze(FunctionDefinition *def) noexcept {
    if (_impl == nullptr) { _impl = luisa::make_unique<Impl>(); }
    return _impl->run(def);
}

const SCEV *SCEVAnalysis::get(Instruction *inst) const noexcept {
    if (_impl == nullptr || inst == nullptr || !_impl->is_current()) { return nullptr; }
    return _get_unchecked(inst);
}

const SCEV *SCEVAnalysis::_get_unchecked(Instruction *inst) const noexcept {
    if (_impl == nullptr || inst == nullptr) { return nullptr; }
    auto iter = _impl->results.find(inst);
    return iter == _impl->results.end() ? nullptr : iter->second;
}

FunctionDefinition *SCEVAnalysis::function() const noexcept {
    return _impl == nullptr || _impl->lifetime_token.expired() ? nullptr : _impl->def;
}

bool SCEVAnalysis::is_current() const noexcept {
    return _impl != nullptr && _impl->is_current();
}

namespace {

struct SCEVLegacyStorage {
    std::mutex mutex;
    luisa::unordered_map<Function *, luisa::shared_ptr<SCEVAnalysis>> function_analyses;
    luisa::unordered_map<Function *, luisa::vector<Instruction *>> function_values;
    luisa::unordered_map<Instruction *, luisa::shared_ptr<SCEVAnalysis>> value_analyses;
};

[[nodiscard]] SCEVLegacyStorage &legacy_storage() noexcept {
    static SCEVLegacyStorage storage;
    return storage;
}

thread_local luisa::shared_ptr<SCEVAnalysis> legacy_query_hold;

void invalidate_locked(SCEVLegacyStorage &storage, Function *function) noexcept {
    if (auto iter = storage.function_values.find(function); iter != storage.function_values.end()) {
        for (auto *inst : iter->second) { storage.value_analyses.erase(inst); }
        storage.function_values.erase(iter);
    }
    storage.function_analyses.erase(function);
}

}// namespace

namespace detail {

void scev_register_function(Function *function) noexcept {
    static_cast<void>(function);
    static_cast<void>(legacy_storage());
}

void scev_invalidate_function(Function *function) noexcept {
    if (function == nullptr) { return; }
    auto &storage = legacy_storage();
    {
        std::lock_guard lock{storage.mutex};
        invalidate_locked(storage, function);
    }
    if (legacy_query_hold != nullptr && legacy_query_hold->function() == function) {
        legacy_query_hold.reset();
    }
}

}// namespace detail

SCEVInfo scev_pass_run_on_function(FunctionDefinition *def) noexcept {
    auto analysis = luisa::make_shared<SCEVAnalysis>();
    auto info = analysis->analyze(def);
    if (def == nullptr) { return info; }
    auto &storage = legacy_storage();
    {
        std::lock_guard lock{storage.mutex};
        invalidate_locked(storage, def);
        auto &values = storage.function_values[def];
        for (auto *block : def->basic_blocks()) {
            for (auto *inst : block->instructions()) {
                if (analysis->_get_unchecked(inst) != nullptr) {
                    storage.value_analyses.emplace(inst, analysis);
                    values.emplace_back(inst);
                }
            }
        }
        storage.function_analyses.emplace(def, analysis);
    }
    legacy_query_hold = std::move(analysis);
    return info;
}

SCEVInfo scev_pass_run_on_module(Module *module, PassReport *report) noexcept {
    SCEVInfo info;
    if (module == nullptr) {
        info.invalid_function_count = 1u;
    } else {
        for (auto *function : module->function_list()) {
            if (auto *def = function->definition()) {
                auto function_info = scev_pass_run_on_function(def);
                info.analyzed_loop_count += function_info.analyzed_loop_count;
                info.rejected_loop_count += function_info.rejected_loop_count;
                info.invalid_function_count += function_info.invalid_function_count;
            }
        }
    }
    if (report != nullptr) {
        report->set("analyzed_loop", info.analyzed_loop_count);
        report->set("rejected_loop", info.rejected_loop_count);
        report->set("invalid_function", info.invalid_function_count);
    }
    return info;
}

const SCEV *scev_get_for_value(Instruction *inst) noexcept {
    if (inst == nullptr) { return nullptr; }
    auto &storage = legacy_storage();
    luisa::shared_ptr<SCEVAnalysis> analysis;
    {
        std::lock_guard lock{storage.mutex};
        if (auto iter = storage.value_analyses.find(inst); iter != storage.value_analyses.end()) {
            analysis = iter->second;
        }
    }
    legacy_query_hold = std::move(analysis);
    if (legacy_query_hold == nullptr) { return nullptr; }
    if (!legacy_query_hold->is_current()) {
        auto *function = legacy_query_hold->function();
        detail::scev_invalidate_function(function);
        return nullptr;
    }
    return legacy_query_hold->_get_unchecked(inst);
}

}// namespace luisa::compute::xir
