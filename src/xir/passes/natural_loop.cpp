#include "natural_loop.h"

#include <limits>
#include <luisa/ast/type.h>
#include <luisa/core/stl/algorithm.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/phi.h>

namespace luisa::compute::xir {

namespace {

[[nodiscard]] luisa::vector<BasicBlock *> block_successors(BasicBlock *block) noexcept {
    luisa::vector<BasicBlock *> successors;
    if (block == nullptr || !block->is_terminated()) { return successors; }
    block->traverse_successors(false, [&](BasicBlock *succ) noexcept {
        successors.emplace_back(succ);
    });
    return successors;
}

[[nodiscard]] luisa::vector<BasicBlock *> block_predecessors(BasicBlock *block) noexcept {
    luisa::vector<BasicBlock *> predecessors;
    if (block == nullptr) { return predecessors; }
    block->traverse_predecessors(false, [&](BasicBlock *pred) noexcept {
        predecessors.emplace_back(pred);
    });
    return predecessors;
}

[[nodiscard]] bool decode_constant_int(const Value *value, int64_t &out) noexcept {
    if (value == nullptr || !value->isa<Constant>()) { return false; }
    auto constant = static_cast<const Constant *>(value);
    auto type = constant->type();
    if (type == nullptr) { return false; }
    if (type->is_int8()) {
        out = constant->as<int8_t>();
    } else if (type->is_uint8()) {
        out = constant->as<uint8_t>();
    } else if (type->is_int16()) {
        out = constant->as<int16_t>();
    } else if (type->is_uint16()) {
        out = constant->as<uint16_t>();
    } else if (type->is_int32()) {
        out = constant->as<int32_t>();
    } else if (type->is_uint32()) {
        out = constant->as<uint32_t>();
    } else if (type->is_int64()) {
        out = constant->as<int64_t>();
    } else if (type->is_uint64()) {
        auto unsigned_value = constant->as<uint64_t>();
        if (unsigned_value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) { return false; }
        out = static_cast<int64_t>(unsigned_value);
    } else {
        return false;
    }
    return true;
}

[[nodiscard]] bool is_less_than_comparison(ArithmeticOp op) noexcept {
    switch (op) {
        case ArithmeticOp::BINARY_LESS:
        case ArithmeticOp::BINARY_LESS_EQUAL:
        case ArithmeticOp::BINARY_GREATER:
        case ArithmeticOp::BINARY_GREATER_EQUAL:
        case ArithmeticOp::BINARY_EQUAL:
        case ArithmeticOp::BINARY_NOT_EQUAL: return true;
        default: return false;
    }
}

[[nodiscard]] bool positive_recurrence_stays_in_range(
    const Type *type, int64_t start, uint64_t trip_count,
    uint64_t stride) noexcept {
    if (trip_count == 0u) { return true; }
    if (type == nullptr || stride == 0u ||
        !(type->is_int() || type->is_uint())) {
        return false;
    }
    auto width = static_cast<uint32_t>(type->size() * 8u);
    if (width == 0u || width > 64u) { return false; }
    uint64_t maximum = 0u;
    if (type->is_uint()) {
        maximum = width == 64u ?
                      std::numeric_limits<uint64_t>::max() :
                      (uint64_t{1u} << width) - 1u;
        if (start < 0) { return false; }
    } else {
        maximum = width == 64u ?
                      static_cast<uint64_t>(
                          std::numeric_limits<int64_t>::max()) :
                      (uint64_t{1u} << (width - 1u)) - 1u;
    }
    // `maximum - start` in unsigned arithmetic is the exact mathematical
    // room to the type's maximum, including INT64_MIN -> INT64_MAX.
    auto room = maximum - static_cast<uint64_t>(start);
    return trip_count <= room / stride;
}

}// namespace

bool NaturalLoop::contains(BasicBlock *block) const noexcept {
    if (block == header) { return true; }
    for (auto *b : body_blocks) {
        if (b == block) { return true; }
    }
    return false;
}

luisa::vector<NaturalLoop> discover_natural_loops(
    FunctionDefinition *def, const DomTree &dom_tree) noexcept {
    luisa::vector<NaturalLoop> loops;
    if (def == nullptr) { return loops; }

    // Step 1: find back-edges (A -> B where B dominates A) and group by header.
    luisa::unordered_map<BasicBlock *, luisa::vector<BasicBlock *>> back_edges_by_header;
    for (auto *block : def->basic_blocks()) {
        for (auto *succ : block_successors(block)) {
            if (dom_tree.contains(block) && dom_tree.contains(succ) &&
                dom_tree.dominates(succ, block)) {
                back_edges_by_header[succ].emplace_back(block);
            }
        }
    }

    // Step 2: build the loop body for each header.
    for (auto &[header, latches] : back_edges_by_header) {
        NaturalLoop loop;
        loop.header = header;
        luisa::unordered_set<BasicBlock *> body;
        body.emplace(header);
        // Reverse traversal from each latch, stopping at the header: every
        // block that can reach a latch without passing through the header is
        // part of the loop body.
        luisa::vector<BasicBlock *> worklist;
        for (auto *latch : latches) {
            loop.latches.emplace_back(latch);
            loop.back_edges.emplace_back(latch, header);
            if (body.emplace(latch).second) {
                worklist.emplace_back(latch);
            }
        }
        while (!worklist.empty()) {
            auto *block = worklist.back();
            worklist.pop_back();
            for (auto *pred : block_predecessors(block)) {
                if (pred == header) { continue; }
                if (!dom_tree.contains(pred)) { continue; }
                if (!dom_tree.dominates(header, pred)) { continue; }
                if (body.emplace(pred).second) {
                    worklist.emplace_back(pred);
                }
            }
        }
        body.erase(header);
        for (auto *block : body) { loop.body_blocks.emplace_back(block); }

        // Step 3: exit blocks — successors of in-loop blocks outside the loop.
        {
            luisa::unordered_set<BasicBlock *> exits;
            auto collect_exits = [&](BasicBlock *block) noexcept {
                for (auto *succ : block_successors(block)) {
                    if (succ != header && !body.contains(succ)) {
                        exits.emplace(succ);
                        auto duplicate = false;
                        for (auto &&edge : loop.exit_edges) {
                            if (edge.first == block && edge.second == succ) {
                                duplicate = true;
                                break;
                            }
                        }
                        if (!duplicate) {
                            loop.exit_edges.emplace_back(block, succ);
                        }
                    }
                }
            };
            collect_exits(header);
            for (auto *block : loop.body_blocks) { collect_exits(block); }
            for (auto *exit : exits) { loop.exit_blocks.emplace_back(exit); }
        }

        // Step 4: preheader — the unique out-of-loop predecessor of the
        // header, provided it branches nowhere else.
        {
            BasicBlock *candidate = nullptr;
            auto outside_count = 0u;
            for (auto *pred : block_predecessors(header)) {
                if (pred == header || body.contains(pred)) { continue; }
                candidate = pred;
                outside_count++;
            }
            if (outside_count == 1u && candidate != nullptr) {
                auto successors = block_successors(candidate);
                if (successors.size() == 1u && successors.front() == header) {
                    loop.preheader = candidate;
                }
            }
        }
        loops.emplace_back(std::move(loop));
    }

    // Step 5: order inner loops before outer loops (ascending body size).
    std::sort(loops.begin(), loops.end(), [](const NaturalLoop &a, const NaturalLoop &b) noexcept {
        return a.body_blocks.size() < b.body_blocks.size();
    });
    return loops;
}

LoopBoundsInfo analyze_loop_bounds(const NaturalLoop &loop) noexcept {
    LoopBoundsInfo info;
    if (loop.header == nullptr || loop.preheader == nullptr ||
        loop.latches.size() != 1u) {
        return info;
    }
    // Canonical consumers need one executable exit edge owned by the header,
    // not merely one distinct destination reached from arbitrary loop blocks.
    if (loop.exit_edges.size() != 1u ||
        loop.exit_edges.front().first != loop.header) {
        return info;
    }
    auto *latch = loop.latches.front();

    // Find the induction phi: one incoming from the preheader, one from the
    // latch, and the latch incoming is phi + constant stride.
    for (auto *inst : loop.header->instructions()) {
        if (!inst->isa<PhiInst>()) { break; }
        auto *phi = static_cast<PhiInst *>(inst);
        auto *type = phi->type();
        if (type == nullptr || !(type->is_int() || type->is_uint())) { continue; }
        if (phi->incoming_count() != 2u) { continue; }
        Value *start = nullptr;
        Value *recurrence = nullptr;
        for (auto i = 0u; i < 2u; ++i) {
            auto incoming = phi->incoming(i);
            if (incoming.block == loop.preheader) {
                start = incoming.value;
            } else if (incoming.block == latch) {
                recurrence = incoming.value;
            }
        }
        if (start == nullptr || recurrence == nullptr) { continue; }
        if (!recurrence->isa<ArithmeticInst>()) { continue; }
        auto *add = static_cast<ArithmeticInst *>(recurrence);
        if (add->op() != ArithmeticOp::BINARY_ADD ||
            add->operand_count() != 2u || add->type() != phi->type()) {
            continue;
        }
        Value *stride_value = nullptr;
        if (add->operand(0u) == phi) {
            stride_value = add->operand(1u);
        } else if (add->operand(1u) == phi) {
            stride_value = add->operand(0u);
        } else {
            continue;
        }
        int64_t stride = 0;
        auto stride_is_constant = decode_constant_int(stride_value, stride);
        if (stride_is_constant && stride == 0) { continue; }
        info.induction_phi = phi;
        info.start_value = start;
        info.stride = stride;
        info.stride_is_constant = stride_is_constant;
        break;
    }
    if (info.induction_phi == nullptr) { return info; }

    // Extract the bound from the header's conditional branch condition:
    // comparison(iv, bound) or comparison(bound, iv).
    auto *terminator = loop.header->terminator();
    if (terminator == nullptr ||
        !terminator->isa<ConditionalBranchInst>()) {
        info.induction_phi = nullptr;
        return info;
    }
    auto *condition = static_cast<ConditionalBranchInst *>(terminator)->condition();
    if (condition == nullptr || !condition->isa<ArithmeticInst>()) {
        info.induction_phi = nullptr;
        return info;
    }
    auto *cmp = static_cast<ArithmeticInst *>(condition);
    if (!is_less_than_comparison(cmp->op()) || cmp->operand_count() != 2u) {
        info.induction_phi = nullptr;
        return info;
    }
    info.comparison_inst = cmp;
    if (cmp->operand(0u) == info.induction_phi) {
        info.bound_value = cmp->operand(1u);
        info.induction_is_lhs = true;
    } else if (cmp->operand(1u) == info.induction_phi) {
        info.bound_value = cmp->operand(0u);
        info.induction_is_lhs = false;
    } else {
        info.induction_phi = nullptr;
        return info;
    }
    info.comparison = cmp->op();

    auto *branch = static_cast<ConditionalBranchInst *>(terminator);
    auto true_is_inside = loop.contains(branch->true_block());
    auto false_is_inside = loop.contains(branch->false_block());
    if (true_is_inside == false_is_inside) {
        info.induction_phi = nullptr;
        return info;
    }
    info.continue_on_true = true_is_inside;
    info.body_entry = true_is_inside ?
                          branch->true_block() :
                          branch->false_block();
    info.exit_block = true_is_inside ?
                          branch->false_block() :
                          branch->true_block();
    if (info.exit_block != loop.exit_edges.front().second) {
        info.induction_phi = nullptr;
        return info;
    }

    // Normalize the executable continuation predicate, accounting for both
    // operand order and branch polarity. Only this strict-less form has the
    // simple positive-stride trip-count formula used by vectorization.
    if (info.induction_is_lhs) {
        info.normalized_strict_less =
            (info.continue_on_true &&
             info.comparison == ArithmeticOp::BINARY_LESS) ||
            (!info.continue_on_true &&
             info.comparison == ArithmeticOp::BINARY_GREATER_EQUAL);
    } else {
        info.normalized_strict_less =
            (info.continue_on_true &&
             info.comparison == ArithmeticOp::BINARY_GREATER) ||
            (!info.continue_on_true &&
             info.comparison == ArithmeticOp::BINARY_LESS_EQUAL);
    }

    // Constant trip count when start, bound, and stride are all constants
    // and the comparison is a strict less-than: ceil((bound-start)/stride).
    int64_t start = 0;
    int64_t bound = 0;
    if (info.stride_is_constant &&
        decode_constant_int(info.start_value, start) &&
        decode_constant_int(info.bound_value, bound) &&
        info.normalized_strict_less &&
        info.stride > 0) {
        if (bound <= start) {
            info.constant_trip_count = 0u;
            info.trip_count_is_constant = true;
        } else {
            // Unsigned subtraction yields the exact positive mathematical
            // difference even when the signed endpoints straddle zero.
            auto span = static_cast<uint64_t>(bound) -
                        static_cast<uint64_t>(start);
            auto stride = static_cast<uint64_t>(info.stride);
            auto trip_count =
                span / stride + static_cast<uint64_t>(span % stride != 0u);
            // The closed-form count is only valid if the recurrence reaches
            // the first failing check without wrapping the IV type. For
            // example, `uint8 iv = 0; iv < 255; iv += 200` cycles after 200
            // instead of terminating after ceil(255 / 200) iterations.
            if (positive_recurrence_stays_in_range(
                    info.induction_phi->type(), start, trip_count, stride)) {
                info.constant_trip_count = trip_count;
                info.trip_count_is_constant = true;
            }
        }
    }
    return info;
}

}// namespace luisa::compute::xir
