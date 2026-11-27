#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstring>
#include <functional>
#include <utility>

#include <luisa/core/stl/format.h>
#include <luisa/core/stl/queue.h>
#include "translator_impl.h"

namespace luisa::compute::osl::detail {

bool StringLattice::join(const StringLattice &other, uint32_t max_size) noexcept {
    if (other.is_bot() || is_top()) { return false; }
    if (is_bot()) {
        *this = other;
        return true;
    }
    if (other.is_top()) {
        kind = Kind::TOP;
        tokens.clear();
        return true;
    }
    luisa::vector<StringToken> merged;
    merged.reserve(tokens.size() + other.tokens.size());
    std::set_union(tokens.begin(), tokens.end(),
                   other.tokens.begin(), other.tokens.end(),
                   std::back_inserter(merged));
    if (merged.size() > max_size) {
        kind = Kind::TOP;
        tokens.clear();
        return true;
    }
    if (merged == tokens) { return false; }
    tokens = std::move(merged);
    return true;
}

bool ScalarLattice::operator==(const ScalarLattice &rhs) const noexcept {
    if (kind != rhs.kind) { return false; }
    if (kind == Kind::INT) { return i == rhs.i; }
    if (kind == Kind::FLOAT) {
        uint32_t a = 0u, b = 0u;
        std::memcpy(&a, &f, sizeof(a));
        std::memcpy(&b, &rhs.f, sizeof(b));
        return a == b;
    }
    return true;
}

bool ScalarLattice::join(const ScalarLattice &other) noexcept {
    if (other.is_bot() || is_top()) { return false; }
    if (is_bot()) {
        *this = other;
        return true;
    }
    if (other.is_top()) {
        kind = Kind::TOP;
        return true;
    }
    if (*this == other) { return false; }
    kind = Kind::TOP;
    return true;
}

namespace {

[[nodiscard]] const Symbol *find_symbol_in_shader(const Shader &shader, luisa::string_view name) noexcept {
    for (auto &&sym : shader.symbols()) {
        if (sym->identifier() == name) {
            return sym.get();
        }
    }
    return nullptr;
}

class CFGBuilder {

private:
    const Shader &_shader;
    OSOCodeRange &_range;
    luisa::vector<TranslationError> &_errors;
    luisa::string_view _layer_name;
    luisa::unordered_map<uint32_t, uint32_t> _jump_pc_to_block;

    [[nodiscard]] uint32_t _new_block() noexcept {
        auto id = static_cast<uint32_t>(_range.blocks.size());
        _range.blocks.emplace_back(OSOBasicBlock{.id = id});
        return id;
    }

    [[nodiscard]] uint32_t _block_for_pc(uint32_t pc) noexcept {
        if (auto it = _jump_pc_to_block.find(pc); it != _jump_pc_to_block.end()) {
            return it->second;
        }
        return _new_block();
    }

    void _fail(uint32_t pc, luisa::string_view opcode, luisa::string msg) noexcept {
        _errors.emplace_back(TranslationError{
            .layer_name = luisa::string{_layer_name},
            .instruction_index = pc,
            .opcode = luisa::string{opcode},
            .message = std::move(msg)});
    }

public:
    CFGBuilder(const Shader &shader, OSOCodeRange &range,
               luisa::string_view layer_name,
               luisa::vector<TranslationError> &errors) noexcept
        : _shader{shader}, _range{range}, _errors{errors}, _layer_name{layer_name} {}

    struct CtrlCtx {
        uint32_t break_blk{~0u};
        uint32_t continue_blk{~0u};
        uint32_t return_blk{~0u};
    };

    [[nodiscard]] bool build() noexcept {
        const auto &insts = _shader.instructions();
        _range.entry_block = _new_block();
        _range.exit_block = _new_block();
        _range.blocks[_range.exit_block].term = OSOTermKind::UNREACHABLE;// caller overrides after layer

        // Pre-register explicit jump targets
        for (uint32_t pc = _range.begin_pc; pc < _range.end_pc; ++pc) {
            const auto &inst = *insts[pc];
            if (inst.opcode() == "jump") {
                if (inst.jump_targets().size() != 1u) {
                    _fail(pc, "jump", "Expected 1 jump target for 'jump'.");
                    return false;
                }
                int t = inst.jump_targets()[0];
                if (t < static_cast<int>(_range.begin_pc) || t > static_cast<int>(_range.end_pc)) {
                    _fail(pc, "jump", luisa::format("Jump target {} out of range [{}, {}].", t, _range.begin_pc, _range.end_pc));
                    return false;
                }
                auto ut = static_cast<uint32_t>(t);
                if (ut == _range.end_pc) {
                    _jump_pc_to_block.emplace(ut, _range.exit_block);
                } else {
                    if (!_jump_pc_to_block.contains(ut)) {
                        _jump_pc_to_block.emplace(ut, _new_block());
                    }
                }
            }
        }

        uint32_t start_blk = _range.entry_block;
        if (auto it = _jump_pc_to_block.find(_range.begin_pc); it != _jump_pc_to_block.end()) {
            _range.blocks[start_blk].term = OSOTermKind::BR;
            _range.blocks[start_blk].succ[0] = it->second;
            start_blk = it->second;
        }
        return _build_range(_range.begin_pc, _range.end_pc, start_blk, _range.exit_block,
                            CtrlCtx{.break_blk = ~0u, .continue_blk = ~0u, .return_blk = _range.exit_block});
    }

private:
    [[nodiscard]] bool _build_range(
        uint32_t begin_pc, uint32_t end_pc,
        uint32_t cur_blk, uint32_t fallthrough_blk,
        const CtrlCtx &ctrl) noexcept {
        const auto &insts = _shader.instructions();
        uint32_t pc = begin_pc;
        while (pc < end_pc) {
            // If an explicit jump targets pc (and we didn't just start cur_blk as that jump block)
            if (pc > begin_pc) {
                if (auto it = _jump_pc_to_block.find(pc); it != _jump_pc_to_block.end()) {
                    if (cur_blk != ~0u && cur_blk != it->second) {
                        _range.blocks[cur_blk].term = OSOTermKind::BR;
                        _range.blocks[cur_blk].succ[0] = it->second;
                    }
                    cur_blk = it->second;
                }
            }

            const auto &inst = *insts[pc];
            auto op = inst.opcode();

            auto ensure_cur = [&]() noexcept -> uint32_t {
                if (cur_blk == ~0u) {
                    cur_blk = _new_block();
                }
                return cur_blk;
            };

            if (op == "if") {
                if (inst.args().size() < 1u || inst.jump_targets().size() != 2u) {
                    _fail(pc, op, "Malformed 'if' instruction.");
                    return false;
                }
                int t0 = inst.jump_targets()[0];
                int t1 = inst.jump_targets()[1];
                if (t0 < static_cast<int>(pc + 1u) || t1 < t0 || t1 > static_cast<int>(end_pc)) {
                    _fail(pc, op, luisa::format("Invalid 'if' jump targets ({}, {}) at pc={}.", t0, t1, pc));
                    return false;
                }
                auto ut0 = static_cast<uint32_t>(t0);
                auto ut1 = static_cast<uint32_t>(t1);
                uint32_t after_bb = (ut1 == end_pc) ? fallthrough_blk : _block_for_pc(ut1);
                uint32_t then_bb = (pc + 1u == ut0) ? after_bb : _block_for_pc(pc + 1u);
                uint32_t else_bb = (ut0 == ut1) ? after_bb : _block_for_pc(ut0);

                uint32_t b = ensure_cur();
                _range.blocks[b].term = OSOTermKind::COND_BR;
                _range.blocks[b].cond_pc = pc;
                _range.blocks[b].cond_symbol = inst.args()[0];
                _range.blocks[b].succ[0] = then_bb;
                _range.blocks[b].succ[1] = else_bb;

                if (pc + 1u < ut0) {
                    if (!_build_range(pc + 1u, ut0, then_bb, after_bb, ctrl)) { return false; }
                }
                if (ut0 < ut1) {
                    if (!_build_range(ut0, ut1, else_bb, after_bb, ctrl)) { return false; }
                }
                cur_blk = after_bb;
                pc = ut1;
                if (ut1 == end_pc) { return true; }
                continue;
            }

            if (op == "for" || op == "while" || op == "dowhile") {
                if (inst.args().size() < 1u || inst.jump_targets().size() != 4u) {
                    _fail(pc, op, luisa::format("Malformed '{}' instruction.", op));
                    return false;
                }
                int t0 = inst.jump_targets()[0];
                int t1 = inst.jump_targets()[1];
                int t2 = inst.jump_targets()[2];
                int t3 = inst.jump_targets()[3];
                if (t0 < static_cast<int>(pc + 1u) || t1 < t0 || t2 < t1 || t3 < t2 || t3 > static_cast<int>(end_pc)) {
                    _fail(pc, op, luisa::format("Invalid '{}' jump targets ({}, {}, {}, {}) at pc={}.", op, t0, t1, t2, t3, pc));
                    return false;
                }
                auto ut0 = static_cast<uint32_t>(t0);
                auto ut1 = static_cast<uint32_t>(t1);
                auto ut2 = static_cast<uint32_t>(t2);
                auto ut3 = static_cast<uint32_t>(t3);
                bool is_dowhile = (op == "dowhile");

                uint32_t after_bb = (ut3 == end_pc) ? fallthrough_blk : _block_for_pc(ut3);
                uint32_t cond_bb = _block_for_pc(ut0);
                uint32_t cond_eval_bb = _new_block();
                uint32_t step_bb = (ut2 == ut3) ? cond_bb : _block_for_pc(ut2);
                uint32_t body_bb = (ut1 == ut2) ? step_bb : _block_for_pc(ut1);
                uint32_t first_target = is_dowhile ? body_bb : cond_bb;
                uint32_t init_bb = (pc + 1u == ut0) ? first_target : _block_for_pc(pc + 1u);

                uint32_t b = ensure_cur();
                _range.blocks[b].term = OSOTermKind::BR;
                _range.blocks[b].succ[0] = init_bb;

                if (pc + 1u < ut0) {
                    if (!_build_range(pc + 1u, ut0, init_bb, first_target, ctrl)) { return false; }
                }

                // Condition range [ut0, ut1) falls through to cond_eval_bb
                if (ut0 < ut1) {
                    if (!_build_range(ut0, ut1, cond_bb, cond_eval_bb, ctrl)) { return false; }
                } else {
                    _range.blocks[cond_bb].term = OSOTermKind::BR;
                    _range.blocks[cond_bb].succ[0] = cond_eval_bb;
                }
                _range.blocks[cond_eval_bb].term = OSOTermKind::COND_BR;
                _range.blocks[cond_eval_bb].cond_pc = pc;
                _range.blocks[cond_eval_bb].cond_symbol = inst.args()[0];
                _range.blocks[cond_eval_bb].succ[0] = body_bb;
                _range.blocks[cond_eval_bb].succ[1] = after_bb;

                CtrlCtx loop_ctrl{
                    .break_blk = after_bb,
                    .continue_blk = step_bb,
                    .return_blk = ctrl.return_blk};
                if (ut1 < ut2) {
                    if (!_build_range(ut1, ut2, body_bb, step_bb, loop_ctrl)) { return false; }
                }
                if (ut2 < ut3) {
                    if (!_build_range(ut2, ut3, step_bb, cond_bb, ctrl)) { return false; }
                }

                cur_blk = after_bb;
                pc = ut3;
                if (ut3 == end_pc) { return true; }
                continue;
            }

            if (op == "functioncall") {
                if (inst.jump_targets().size() != 1u) {
                    _fail(pc, op, "Malformed 'functioncall' instruction.");
                    return false;
                }
                int t0 = inst.jump_targets()[0];
                if (t0 < static_cast<int>(pc + 1u) || t0 > static_cast<int>(end_pc)) {
                    _fail(pc, op, luisa::format("Invalid 'functioncall' target {} at pc={}.", t0, pc));
                    return false;
                }
                auto ut0 = static_cast<uint32_t>(t0);
                uint32_t after_bb = (ut0 == end_pc) ? fallthrough_blk : _block_for_pc(ut0);
                uint32_t body_bb = (pc + 1u == ut0) ? after_bb : _block_for_pc(pc + 1u);

                uint32_t b = ensure_cur();
                _range.blocks[b].term = OSOTermKind::BR;
                _range.blocks[b].succ[0] = body_bb;

                if (pc + 1u < ut0) {
                    CtrlCtx call_ctrl{.break_blk = ~0u, .continue_blk = ~0u, .return_blk = after_bb};
                    if (!_build_range(pc + 1u, ut0, body_bb, after_bb, call_ctrl)) { return false; }
                }
                cur_blk = after_bb;
                pc = ut0;
                if (ut0 == end_pc) { return true; }
                continue;
            }

            if (op == "break") {
                if (ctrl.break_blk == ~0u) {
                    _fail(pc, op, "'break' outside of a loop.");
                    return false;
                }
                uint32_t b = ensure_cur();
                _range.blocks[b].term = OSOTermKind::BR;
                _range.blocks[b].succ[0] = ctrl.break_blk;
                cur_blk = ~0u;
                ++pc;
                continue;
            }

            if (op == "continue") {
                if (ctrl.continue_blk == ~0u) {
                    _fail(pc, op, "'continue' outside of a loop.");
                    return false;
                }
                uint32_t b = ensure_cur();
                _range.blocks[b].term = OSOTermKind::BR;
                _range.blocks[b].succ[0] = ctrl.continue_blk;
                cur_blk = ~0u;
                ++pc;
                continue;
            }

            if (op == "return") {
                uint32_t b = ensure_cur();
                _range.blocks[b].term = OSOTermKind::BR;
                _range.blocks[b].succ[0] = ctrl.return_blk;
                cur_blk = ~0u;
                ++pc;
                continue;
            }

            if (op == "exit") {
                uint32_t b = ensure_cur();
                _range.blocks[b].term = OSOTermKind::EXIT_LAYER;
                _range.blocks[b].succ[0] = ~0u;
                cur_blk = ~0u;
                ++pc;
                continue;
            }

            if (op == "jump") {
                uint32_t target = _block_for_pc(static_cast<uint32_t>(inst.jump_targets()[0]));
                uint32_t b = ensure_cur();
                _range.blocks[b].term = OSOTermKind::BR;
                _range.blocks[b].succ[0] = target;
                cur_blk = ~0u;
                ++pc;
                continue;
            }

            if (cur_blk != ~0u) {
                _range.blocks[cur_blk].pcs.emplace_back(pc);
            }
            ++pc;
        }

        if (cur_blk != ~0u && cur_blk != fallthrough_blk) {
            _range.blocks[cur_blk].term = OSOTermKind::BR;
            _range.blocks[cur_blk].succ[0] = fallthrough_blk;
        }
        return true;
    }
};

struct AbstractState {
    bool reachable{false};
    luisa::unordered_map<const Symbol *, StringLattice> strings;
    luisa::unordered_map<const Symbol *, luisa::vector<StringLattice>> string_arrays;
    luisa::unordered_map<const Symbol *, ScalarLattice> scalars;
    luisa::unordered_map<const Symbol *, luisa::float3> triples; // exact constant triples
    luisa::unordered_set<const Symbol *> non_const_triples;
    luisa::unordered_map<const Symbol *, luisa::float4x4> matrices;
    luisa::unordered_set<const Symbol *> non_const_matrices;
    luisa::unordered_map<StringToken, StringLattice> msg_strings;
    luisa::unordered_set<StringToken> definitely_set_msgs;
    bool msg_strings_tainted{false};

    bool join_from(const AbstractState &other, uint32_t max_size) noexcept {
        if (!other.reachable) { return false; }
        bool first_reach = !reachable;
        bool changed = first_reach;
        reachable = true;
        for (const auto &[sym, lat] : other.strings) {
            changed |= strings[sym].join(lat, max_size);
        }
        for (const auto &[sym, arr] : other.string_arrays) {
            auto &dst_arr = string_arrays[sym];
            if (dst_arr.size() < arr.size()) {
                dst_arr.resize(arr.size());
                changed = true;
            }
            for (size_t i = 0u; i < arr.size(); ++i) {
                changed |= dst_arr[i].join(arr[i], max_size);
            }
        }
        for (const auto &[sym, lat] : other.scalars) {
            changed |= scalars[sym].join(lat);
        }
        for (const Symbol *sym : other.non_const_triples) {
            triples.erase(sym);
            if (non_const_triples.emplace(sym).second) { changed = true; }
        }
        for (const auto &[sym, v] : other.triples) {
            if (non_const_triples.contains(sym)) { continue; }
            auto it = triples.find(sym);
            if (it == triples.end()) {
                triples.emplace(sym, v);
                changed = true;
            } else if (it->second.x != v.x || it->second.y != v.y || it->second.z != v.z) {
                triples.erase(it);
                non_const_triples.emplace(sym);
                changed = true;
            }
        }
        for (const Symbol *sym : other.non_const_matrices) {
            matrices.erase(sym);
            if (non_const_matrices.emplace(sym).second) { changed = true; }
        }
        for (const auto &[sym, m] : other.matrices) {
            if (non_const_matrices.contains(sym)) { continue; }
            auto it = matrices.find(sym);
            if (it == matrices.end()) {
                matrices.emplace(sym, m);
                changed = true;
            } else {
                bool eq = true;
                for (uint32_t c = 0u; c < 4u; ++c) {
                    for (uint32_t r = 0u; r < 4u; ++r) {
                        if (it->second[c][r] != m[c][r]) { eq = false; }
                    }
                }
                if (!eq) {
                    matrices.erase(it);
                    non_const_matrices.emplace(sym);
                    changed = true;
                }
            }
        }
        if (other.msg_strings_tainted && !msg_strings_tainted) {
            msg_strings_tainted = true;
            changed = true;
        }
        for (const auto &[tok, lat] : other.msg_strings) {
            changed |= msg_strings[tok].join(lat, max_size);
        }
        if (first_reach) {
            definitely_set_msgs = other.definitely_set_msgs;
        } else {
            for (auto it = definitely_set_msgs.begin(); it != definitely_set_msgs.end();) {
                if (!other.definitely_set_msgs.contains(*it)) {
                    it = definitely_set_msgs.erase(it);
                    changed = true;
                } else {
                    ++it;
                }
            }
        }
        return changed;
    }
};

void build_layer_ranges(LinkedLayer &ll, luisa::vector<TranslationError> &errors) noexcept {
    const auto &markers = ll.shader->code_markers();
    const auto &insts = ll.shader->instructions();
    if (insts.empty()) { return; }
    if (markers.empty()) {
        errors.emplace_back(TranslationError{
            .layer_name = ll.layer->name,
            .message = "Shader contains instructions without any 'code' marker."});
        return;
    }
    if (markers.front().instruction != 0u) {
        errors.emplace_back(TranslationError{
            .layer_name = ll.layer->name,
            .message = "First 'code' marker does not start at instruction 0."});
        return;
    }
    ll.ranges.reserve(markers.size());
    for (size_t i = 0u; i < markers.size(); ++i) {
        uint32_t begin_pc = markers[i].instruction;
        uint32_t end_pc = (i + 1u < markers.size()) ? markers[i + 1u].instruction : static_cast<uint32_t>(insts.size());
        if (end_pc < begin_pc || end_pc > insts.size()) {
            errors.emplace_back(TranslationError{
                .layer_name = ll.layer->name,
                .message = luisa::format("Invalid code marker range [{}, {}) for '{}'.",
                                         begin_pc, end_pc, markers[i].identifier)});
            return;
        }
        OSOCodeRange range;
        range.identifier = markers[i].identifier;
        range.begin_pc = begin_pc;
        range.end_pc = end_pc;
        range.is_main = (range.identifier == "___main___");
        if (!range.is_main) {
            range.param_symbol = find_symbol_in_shader(*ll.shader, range.identifier);
        }
        CFGBuilder cfg_builder{*ll.shader, range, ll.layer->name, errors};
        if (!cfg_builder.build()) {
            return;
        }
        ll.ranges.emplace_back(std::move(range));
    }
}

}// namespace

bool is_dynamic_input_symbol(const LinkedLayer &layer, const Symbol *sym) noexcept {
    if (sym == nullptr || layer.dynamic_inputs.empty()) { return false; }
    if (layer.dynamic_inputs.contains(sym)) { return true; }
    for (const Symbol *child : sym->children()) {
        if (is_dynamic_input_symbol(layer, child)) { return true; }
    }
    return false;
}

void analyze_group_strings_and_constants(
    const LinkedGroup &group,
    StringTable &strings,
    uint32_t max_lattice_size,
    GroupStringAnalysis &out_analysis,
    luisa::vector<TranslationError> &errors) noexcept {

    // Build CFGs for every active layer
    auto &mutable_layers = const_cast<LinkedGroup &>(group).layers;
    for (auto &ll : mutable_layers) {
        if (ll.ranges.empty() && !ll.shader->instructions().empty()) {
            build_layer_ranges(ll, errors);
            if (!errors.empty()) { return; }
        }
    }

    out_analysis.layers.clear();
    out_analysis.layers.resize(group.layers.size());

    luisa::unordered_map<StringToken, StringLattice> group_msg_strings;
    luisa::unordered_set<StringToken> group_definitely_set_msgs;
    bool group_msg_tainted = false;

    for (size_t li = 0u; li < group.layers.size(); ++li) {
        const auto &ll = group.layers[li];
        auto &la = out_analysis.layers[li];
        la.instructions.resize(ll.shader->instructions().size());

        // Initialize state from symbols, overrides, and incoming connections
        AbstractState init_state;
        init_state.reachable = true;
        init_state.msg_strings = group_msg_strings;
        init_state.definitely_set_msgs = group_definitely_set_msgs;
        init_state.msg_strings_tainted = group_msg_tainted;

        auto init_scalar_or_string_sym = [&](const Symbol *sym, const ParameterValue *ov) {
            if (sym->type() == nullptr || sym->type()->tag() != Type::Tag::SIMPLE) { return; }
            auto prim = static_cast<const SimpleType *>(sym->type())->primitive();
            int len = 0;
            if (auto it = ll.effective_array_length.find(sym); it != ll.effective_array_length.end()) {
                len = it->second;
            }
            if (sym->tag() == Symbol::Tag::SYM_GLOBAL) {
                if (prim == SimpleType::Primitive::STRING) {
                    init_state.strings[sym] = StringLattice::top();
                } else {
                    init_state.scalars[sym] = ScalarLattice::top();
                }
                return;
            }
            if (ll.dynamic_inputs.contains(sym)) {
                if (prim == SimpleType::Primitive::STRING) {
                    if (sym->is_array()) {
                        luisa::vector<StringLattice> arr(static_cast<size_t>(std::max(0, len)), StringLattice::top());
                        init_state.string_arrays.emplace(sym, std::move(arr));
                    } else {
                        init_state.strings[sym] = StringLattice::top();
                    }
                } else {
                    init_state.scalars[sym] = ScalarLattice::top();
                    init_state.triples.erase(sym);
                    init_state.non_const_triples.emplace(sym);
                    init_state.matrices.erase(sym);
                    init_state.non_const_matrices.emplace(sym);
                }
                return;
            }

            if (sym->is_array()) {
                if (prim == SimpleType::Primitive::STRING) {
                    luisa::vector<StringLattice> arr(static_cast<size_t>(std::max(0, len)), StringLattice::singleton(empty_string_token));
                    if (ov != nullptr) {
                        if (auto *vs = luisa::get_if<luisa::vector<luisa::string>>(&ov->value)) {
                            for (size_t k = 0u; k < std::min(arr.size(), vs->size()); ++k) {
                                arr[k] = StringLattice::singleton(strings.intern((*vs)[k]));
                            }
                        }
                    } else {
                        for (size_t k = 0u; k < std::min(arr.size(), sym->initial_values().size()); ++k) {
                            if (sym->initial_values()[k].is_string()) {
                                arr[k] = StringLattice::singleton(strings.intern(sym->initial_values()[k].as_string()));
                            }
                        }
                    }
                    init_state.string_arrays.emplace(sym, std::move(arr));
                }
                return;
            }

            if (prim == SimpleType::Primitive::STRING) {
                StringToken tok = empty_string_token;
                if (ov != nullptr) {
                    if (auto *ps = luisa::get_if<luisa::string>(&ov->value)) {
                        tok = strings.intern(*ps);
                    }
                } else if (!sym->initial_values().empty() && sym->initial_values()[0].is_string()) {
                    tok = strings.intern(sym->initial_values()[0].as_string());
                }
                init_state.strings[sym] = StringLattice::singleton(tok);
            } else if (prim == SimpleType::Primitive::INT) {
                int32_t v = 0;
                if (ov != nullptr) {
                    if (auto *pi = luisa::get_if<int32_t>(&ov->value)) { v = *pi; }
                    else if (auto *pf = luisa::get_if<float>(&ov->value)) { v = static_cast<int32_t>(*pf); }
                } else if (!sym->initial_values().empty() && sym->initial_values()[0].is_number()) {
                    v = static_cast<int32_t>(sym->initial_values()[0].as_int());
                }
                init_state.scalars[sym] = ScalarLattice::make_int(v);
            } else if (prim == SimpleType::Primitive::FLOAT) {
                float v = 0.0f;
                if (ov != nullptr) {
                    if (auto *pf = luisa::get_if<float>(&ov->value)) { v = *pf; }
                    else if (auto *pi = luisa::get_if<int32_t>(&ov->value)) { v = static_cast<float>(*pi); }
                } else if (!sym->initial_values().empty() && sym->initial_values()[0].is_number()) {
                    v = sym->initial_values()[0].as_float();
                }
                init_state.scalars[sym] = ScalarLattice::make_float(v);
            } else if (is_triple_type(sym->type())) {
                luisa::float3 v{0.0f};
                if (ov != nullptr) {
                    if (auto *p3 = luisa::get_if<luisa::float3>(&ov->value)) { v = *p3; }
                    else if (auto *pf = luisa::get_if<float>(&ov->value)) { v = luisa::float3{*pf}; }
                    else if (auto *pi = luisa::get_if<int32_t>(&ov->value)) { v = luisa::float3{static_cast<float>(*pi)}; }
                } else if (sym->initial_values().size() >= 3u) {
                    v = luisa::float3{sym->initial_values()[0].as_float(),
                                      sym->initial_values()[1].as_float(),
                                      sym->initial_values()[2].as_float()};
                } else if (sym->initial_values().size() == 1u) {
                    v = luisa::float3{sym->initial_values()[0].as_float()};
                }
                init_state.triples[sym] = v;
            } else if (prim == SimpleType::Primitive::MATRIX) {
                luisa::float4x4 m = luisa::make_float4x4(1.0f);
                if (ov != nullptr) {
                    if (auto *pm = luisa::get_if<luisa::float4x4>(&ov->value)) { m = *pm; }
                    else if (auto *pf = luisa::get_if<float>(&ov->value)) { m = luisa::make_float4x4(*pf); }
                    else if (auto *pi = luisa::get_if<int32_t>(&ov->value)) { m = luisa::make_float4x4(static_cast<float>(*pi)); }
                } else if (sym->initial_values().size() >= 16u) {
                    float raw[16];
                    for (size_t k = 0u; k < 16u; ++k) { raw[k] = sym->initial_values()[k].as_float(); }
                    m = osl_row_major_to_xir_matrix(raw);
                } else if (sym->initial_values().size() == 1u) {
                    m = luisa::make_float4x4(sym->initial_values()[0].as_float());
                } else {
                    m = luisa::make_float4x4(0.0f);
                }
                init_state.matrices[sym] = m;
            }
        };

        for (auto &&sym_ptr : ll.shader->symbols()) {
            const Symbol *sym = sym_ptr.get();
            const ParameterValue *ov = nullptr;
            if (auto it = ll.param_overrides.find(sym); it != ll.param_overrides.end()) {
                ov = it->second;
            }
            init_scalar_or_string_sym(sym, ov);
        }

        // Apply incoming connections from already-analyzed upstream layers
        for (uint32_t ci : ll.incoming_connections) {
            const auto &conn = group.connections[ci];
            const auto &src_la = out_analysis.layers[conn.src_layer];
            std::function<void(const Symbol *, const Symbol *)> apply_conn =
                [&](const Symbol *src_s, const Symbol *dst_s) {
                    if (src_s->type()->tag() == Type::Tag::STRUCT && dst_s->type()->tag() == Type::Tag::STRUCT) {
                        for (size_t k = 0u; k < std::min(src_s->children().size(), dst_s->children().size()); ++k) {
                            apply_conn(src_s->children()[k], dst_s->children()[k]);
                        }
                        return;
                    }
                    if (is_simple_type(dst_s->type(), SimpleType::Primitive::STRING)) {
                        if (auto it = src_la.output_strings.find(src_s); it != src_la.output_strings.end()) {
                            init_state.strings[dst_s] = it->second;
                        } else {
                            init_state.strings[dst_s] = StringLattice::top();
                        }
                    } else {
                        if (auto it = src_la.output_scalars.find(src_s); it != src_la.output_scalars.end()) {
                            init_state.scalars[dst_s] = it->second;
                        } else {
                            init_state.scalars[dst_s] = ScalarLattice::top();
                        }
                        init_state.triples.erase(dst_s);
                        init_state.non_const_triples.emplace(dst_s);
                        init_state.matrices.erase(dst_s);
                        init_state.non_const_matrices.emplace(dst_s);
                    }
                };
            apply_conn(conn.src_symbol, conn.dst_symbol);
        }

        auto get_str = [&](const AbstractState &st, const Symbol *s) -> StringLattice {
            if (s->tag() == Symbol::Tag::SYM_CONST && !s->initial_values().empty() && s->initial_values()[0].is_string()) {
                return StringLattice::singleton(strings.intern(s->initial_values()[0].as_string()));
            }
            if (auto it = st.strings.find(s); it != st.strings.end()) { return it->second; }
            return StringLattice::bot();
        };

        auto get_scalar = [&](const AbstractState &st, const Symbol *s) -> ScalarLattice {
            if (s->tag() == Symbol::Tag::SYM_CONST && !s->initial_values().empty() && s->initial_values()[0].is_number()) {
                if (is_simple_type(s->type(), SimpleType::Primitive::INT)) {
                    return ScalarLattice::make_int(static_cast<int32_t>(s->initial_values()[0].as_int()));
                }
                if (is_simple_type(s->type(), SimpleType::Primitive::FLOAT)) {
                    return ScalarLattice::make_float(s->initial_values()[0].as_float());
                }
            }
            if (auto it = st.scalars.find(s); it != st.scalars.end()) { return it->second; }
            return ScalarLattice::top();
        };

        auto kill_sym = [&](AbstractState &st, const Symbol *s) {
            if (s == nullptr) { return; }
            if (s->is_array()) {
                if (is_simple_type(s->type(), SimpleType::Primitive::STRING)) {
                    for (auto &el : st.string_arrays[s]) { el = StringLattice::top(); }
                }
                return;
            }
            if (is_simple_type(s->type(), SimpleType::Primitive::STRING)) {
                st.strings[s] = StringLattice::top();
            } else {
                st.scalars[s] = ScalarLattice::top();
                st.triples.erase(s);
                st.non_const_triples.emplace(s);
                st.matrices.erase(s);
                st.non_const_matrices.emplace(s);
            }
            for (const Symbol *c : s->children()) {
                st.strings[c] = StringLattice::top();
                st.scalars[c] = ScalarLattice::top();
                st.triples.erase(c);
                st.non_const_triples.emplace(c);
                st.matrices.erase(c);
                st.non_const_matrices.emplace(c);
            }
        };

        auto transfer_inst = [&](uint32_t pc, AbstractState &st) {
            const auto &inst = *ll.shader->instructions()[pc];
            auto &ia = la.instructions[pc];
            ia.reachable = true;
            const auto args = inst.args();
            if (ia.arg_strings.size() != args.size()) {
                ia.arg_strings.assign(args.size(), StringLattice::bot());
                ia.arg_scalars.assign(args.size(), ScalarLattice::bot());
            }
            for (size_t i = 0u; i < args.size(); ++i) {
                if (is_simple_type(args[i]->type(), SimpleType::Primitive::STRING)) {
                    if (args[i]->is_array()) {
                        StringLattice joined = StringLattice::bot();
                        if (auto it = st.string_arrays.find(args[i]); it != st.string_arrays.end()) {
                            for (const auto &el : it->second) { joined.join(el, max_lattice_size); }
                        }
                        ia.arg_strings[i].join(joined, max_lattice_size);
                    } else {
                        ia.arg_strings[i].join(get_str(st, args[i]), max_lattice_size);
                    }
                } else {
                    ia.arg_scalars[i].join(get_scalar(st, args[i]));
                }
            }

            auto op = inst.opcode();
            auto set_int = [&](const Symbol *dst, int32_t v) {
                st.scalars[dst] = ScalarLattice::make_int(v);
                st.triples.erase(dst);
                st.matrices.erase(dst);
                ia.folded_result.join(st.scalars[dst]);
            };
            auto set_float = [&](const Symbol *dst, float v) {
                st.scalars[dst] = ScalarLattice::make_float(v);
                st.triples.erase(dst);
                st.matrices.erase(dst);
                ia.folded_result.join(st.scalars[dst]);
            };

            if (op == "assign" && args.size() >= 2u) {
                const Symbol *dst = args[0];
                const Symbol *src = args[1];
                if (dst->is_array()) {
                    if (is_simple_type(dst->type(), SimpleType::Primitive::STRING) &&
                        is_simple_type(src->type(), SimpleType::Primitive::STRING)) {
                        if (auto it = st.string_arrays.find(src); it != st.string_arrays.end()) {
                            auto &d = st.string_arrays[dst];
                            for (size_t k = 0u; k < std::min(d.size(), it->second.size()); ++k) {
                                d[k] = it->second[k];
                            }
                        } else {
                            kill_sym(st, dst);
                        }
                    }
                    return;
                }
                if (dst->type()->tag() == Type::Tag::STRUCT && src->type()->tag() == Type::Tag::STRUCT) {
                    for (size_t k = 0u; k < std::min(dst->children().size(), src->children().size()); ++k) {
                        const Symbol *dc = dst->children()[k];
                        const Symbol *sc = src->children()[k];
                        if (is_simple_type(dc->type(), SimpleType::Primitive::STRING)) {
                            st.strings[dc] = get_str(st, sc);
                        } else {
                            st.scalars[dc] = get_scalar(st, sc);
                        }
                    }
                    return;
                }
                if (is_simple_type(dst->type(), SimpleType::Primitive::STRING)) {
                    st.strings[dst] = get_str(st, src);
                    return;
                }
                auto sv = get_scalar(st, src);
                if (is_simple_type(dst->type(), SimpleType::Primitive::INT)) {
                    if (sv.is_int()) { set_int(dst, sv.i); }
                    else if (sv.is_float()) { set_int(dst, static_cast<int32_t>(sv.f)); }
                    else { kill_sym(st, dst); }
                } else if (is_simple_type(dst->type(), SimpleType::Primitive::FLOAT)) {
                    if (sv.is_float()) { set_float(dst, sv.f); }
                    else if (sv.is_int()) { set_float(dst, static_cast<float>(sv.i)); }
                    else { kill_sym(st, dst); }
                } else if (is_triple_type(dst->type())) {
                    st.scalars[dst] = ScalarLattice::top();
                    if (auto it = st.triples.find(src); it != st.triples.end()) {
                        st.triples[dst] = it->second;
                        st.non_const_triples.erase(dst);
                    } else if (sv.is_float()) {
                        st.triples[dst] = luisa::float3{sv.f};
                        st.non_const_triples.erase(dst);
                    } else if (sv.is_int()) {
                        st.triples[dst] = luisa::float3{static_cast<float>(sv.i)};
                        st.non_const_triples.erase(dst);
                    } else {
                        kill_sym(st, dst);
                    }
                } else if (is_simple_type(dst->type(), SimpleType::Primitive::MATRIX)) {
                    st.scalars[dst] = ScalarLattice::top();
                    if (auto it = st.matrices.find(src); it != st.matrices.end()) {
                        st.matrices[dst] = it->second;
                        st.non_const_matrices.erase(dst);
                    } else if (sv.is_float()) {
                        st.matrices[dst] = luisa::make_float4x4(sv.f);
                        st.non_const_matrices.erase(dst);
                    } else if (sv.is_int()) {
                        st.matrices[dst] = luisa::make_float4x4(static_cast<float>(sv.i));
                        st.non_const_matrices.erase(dst);
                    } else {
                        kill_sym(st, dst);
                    }
                } else {
                    kill_sym(st, dst);
                }
                return;
            }

            if (op == "concat" && args.size() >= 3u) {
                const Symbol *dst = args[0];
                StringLattice cur = get_str(st, args[1]);
                for (size_t k = 2u; k < args.size(); ++k) {
                    StringLattice rhs = get_str(st, args[k]);
                    if (cur.is_bot() || rhs.is_bot()) {
                        cur = StringLattice::bot();
                        break;
                    }
                    if (cur.is_top() || rhs.is_top() ||
                        cur.tokens.size() * rhs.tokens.size() > max_lattice_size) {
                        cur = StringLattice::top();
                        break;
                    }
                    StringLattice next;
                    for (StringToken ta : cur.tokens) {
                        for (StringToken tb : rhs.tokens) {
                            luisa::string combined{strings.lookup(ta)};
                            combined.append(strings.lookup(tb));
                            next.join(StringLattice::singleton(strings.intern(combined)), max_lattice_size);
                        }
                    }
                    cur = std::move(next);
                }
                st.strings[dst] = std::move(cur);
                return;
            }

            if (op == "format" && args.size() >= 2u) {
                const Symbol *dst = args[0];
                StringLattice fmt_lat = get_str(st, args[1]);
                if (!fmt_lat.is_set()) {
                    st.strings[dst] = StringLattice::top();
                    return;
                }
                auto count_specs = [](luisa::string_view f) noexcept -> luisa::optional<size_t> {
                    size_t cnt = 0u;
                    for (size_t i = 0u; i < f.size(); ++i) {
                        if (f[i] != '%') { continue; }
                        if (i + 1u >= f.size()) { return luisa::nullopt; }
                        if (f[i + 1u] == '%') { ++i; continue; }
                        size_t j = i + 1u;
                        while (j < f.size() && (f[j] == '-' || f[j] == '+' || f[j] == ' ' || f[j] == '#' || f[j] == '0')) { ++j; }
                        while (j < f.size() && std::isdigit(static_cast<unsigned char>(f[j]))) { ++j; }
                        if (j < f.size() && f[j] == '.') {
                            ++j;
                            while (j < f.size() && std::isdigit(static_cast<unsigned char>(f[j]))) { ++j; }
                        }
                        if (j >= f.size()) { return luisa::nullopt; }
                        ++cnt;
                        i = j;
                    }
                    return cnt;
                };
                size_t max_specs = 0u;
                bool specs_valid = !fmt_lat.tokens.empty();
                for (StringToken ftok : fmt_lat.tokens) {
                    auto c = count_specs(strings.lookup(ftok));
                    if (!c.has_value()) { specs_valid = false; break; }
                    max_specs = std::max(max_specs, *c);
                }
                size_t arg_limit = specs_valid ? std::min(args.size(), 2u + max_specs) : args.size();
                // Gather Cartesian options for needed format arguments
                luisa::vector<luisa::vector<HostFormatArg>> arg_choices;
                size_t prod = fmt_lat.tokens.size();
                bool ok = specs_valid && !fmt_lat.is_bot();
                for (size_t k = 2u; k < arg_limit && ok; ++k) {
                    const Symbol *a = args[k];
                    luisa::vector<HostFormatArg> choices;
                    if (is_simple_type(a->type(), SimpleType::Primitive::STRING)) {
                        auto sl = get_str(st, a);
                        if (!sl.is_set()) { ok = false; break; }
                        for (StringToken t : sl.tokens) { choices.emplace_back(strings.lookup(t)); }
                    } else if (is_simple_type(a->type(), SimpleType::Primitive::INT)) {
                        auto sc = get_scalar(st, a);
                        if (!sc.is_int()) { ok = false; break; }
                        choices.emplace_back(sc.i);
                    } else if (is_simple_type(a->type(), SimpleType::Primitive::FLOAT)) {
                        auto sc = get_scalar(st, a);
                        if (!sc.is_float()) { ok = false; break; }
                        choices.emplace_back(sc.f);
                    } else if (is_triple_type(a->type())) {
                        auto it = st.triples.find(a);
                        if (it == st.triples.end()) { ok = false; break; }
                        choices.emplace_back(it->second);
                    } else if (is_simple_type(a->type(), SimpleType::Primitive::MATRIX)) {
                        auto it = st.matrices.find(a);
                        if (it == st.matrices.end()) { ok = false; break; }
                        choices.emplace_back(it->second);
                    } else {
                        ok = false;
                        break;
                    }
                    prod *= choices.size();
                    if (prod > max_lattice_size) { ok = false; break; }
                    arg_choices.emplace_back(std::move(choices));
                }
                if (!ok) {
                    st.strings[dst] = StringLattice::top();
                    return;
                }
                StringLattice result;
                luisa::vector<HostFormatArg> cur_args(arg_choices.size());
                std::function<void(luisa::string_view, size_t)> enumerate = [&](luisa::string_view fmt, size_t depth) {
                    if (result.is_top()) { return; }
                    if (depth == arg_choices.size()) {
                        auto res = eval_osl_format(fmt, cur_args);
                        if (!res.has_value()) {
                            result = StringLattice::top();
                            return;
                        }
                        result.join(StringLattice::singleton(strings.intern(*res)), max_lattice_size);
                        return;
                    }
                    for (const auto &c : arg_choices[depth]) {
                        cur_args[depth] = c;
                        enumerate(fmt, depth + 1u);
                    }
                };
                for (StringToken ftok : fmt_lat.tokens) {
                    enumerate(strings.lookup(ftok), 0u);
                }
                st.strings[dst] = std::move(result);
                return;
            }

            if (op == "substr" && args.size() >= 3u) {
                const Symbol *dst = args[0];
                StringLattice s_lat = get_str(st, args[1]);
                ScalarLattice start_lat = get_scalar(st, args[2]);
                bool has_len = args.size() >= 4u;
                ScalarLattice len_lat = has_len ? get_scalar(st, args[3]) : ScalarLattice::make_int(INT32_MAX);
                if (!s_lat.is_set() || !start_lat.is_int() || (has_len && !len_lat.is_int())) {
                    st.strings[dst] = StringLattice::top();
                    return;
                }
                StringLattice res;
                for (StringToken t : s_lat.tokens) {
                    luisa::string_view sv = strings.lookup(t);
                    int32_t n = has_len ? len_lat.i : static_cast<int32_t>(sv.size());
                    auto sub = eval_osl_substr(sv, start_lat.i, n);
                    res.join(StringLattice::singleton(strings.intern(sub)), max_lattice_size);
                }
                st.strings[dst] = std::move(res);
                return;
            }

            if ((op == "eq" || op == "neq" || op == "lt" || op == "gt" || op == "le" || op == "ge") &&
                args.size() >= 3u && is_simple_type(args[1]->type(), SimpleType::Primitive::STRING)) {
                const Symbol *dst = args[0];
                StringLattice a = get_str(st, args[1]);
                StringLattice b = get_str(st, args[2]);
                if (op == "eq" || op == "neq") {
                    bool is_eq = (op == "eq");
                    if (a.is_singleton() && b.is_singleton()) {
                        bool same = (*a.as_singleton() == *b.as_singleton());
                        set_int(dst, (same == is_eq) ? 1 : 0);
                    } else if (a.is_set() && b.is_set()) {
                        bool any_overlap = false;
                        for (StringToken ta : a.tokens) {
                            if (std::binary_search(b.tokens.begin(), b.tokens.end(), ta)) {
                                any_overlap = true;
                                break;
                            }
                        }
                        if (!any_overlap) {
                            set_int(dst, is_eq ? 0 : 1);
                        } else {
                            kill_sym(st, dst);
                        }
                    } else {
                        kill_sym(st, dst);
                    }
                    return;
                }
                if (a.is_set() && b.is_set() && !a.tokens.empty() && !b.tokens.empty() &&
                    a.tokens.size() * b.tokens.size() <= 256u) {
                    bool saw_t = false, saw_f = false;
                    for (StringToken ta : a.tokens) {
                        luisa::string_view sa = strings.lookup(ta);
                        for (StringToken tb : b.tokens) {
                            luisa::string_view sb = strings.lookup(tb);
                            bool r = (op == "lt") ? (sa < sb) : (op == "gt") ? (sa > sb) : (op == "le") ? (sa <= sb) : (sa >= sb);
                            if (r) { saw_t = true; } else { saw_f = true; }
                        }
                    }
                    if (saw_t != saw_f) {
                        set_int(dst, saw_t ? 1 : 0);
                        return;
                    }
                }
                kill_sym(st, dst);
                return;
            }

            if ((op == "startswith" || op == "endswith") && args.size() >= 3u) {
                const Symbol *dst = args[0];
                StringLattice s = get_str(st, args[1]);
                StringLattice p = get_str(st, args[2]);
                if (s.is_set() && p.is_set()) {
                    bool saw_true = false, saw_false = false;
                    for (StringToken ts : s.tokens) {
                        for (StringToken tp : p.tokens) {
                            bool m = (op == "startswith") ?
                                         strings.lookup(ts).starts_with(strings.lookup(tp)) :
                                         strings.lookup(ts).ends_with(strings.lookup(tp));
                            if (m) { saw_true = true; }
                            else { saw_false = true; }
                        }
                    }
                    if (saw_true != saw_false) {
                        set_int(dst, saw_true ? 1 : 0);
                        return;
                    }
                }
                kill_sym(st, dst);
                return;
            }

            if (op == "strlen" && args.size() >= 2u) {
                StringLattice s = get_str(st, args[1]);
                if (s.is_set() && !s.tokens.empty()) {
                    size_t len0 = strings.lookup(s.tokens[0]).size();
                    bool all_same = std::all_of(s.tokens.begin(), s.tokens.end(), [&](StringToken t) {
                        return strings.lookup(t).size() == len0;
                    });
                    if (all_same) {
                        set_int(args[0], static_cast<int32_t>(len0));
                        return;
                    }
                }
                kill_sym(st, args[0]);
                return;
            }

            if (op == "getchar" && args.size() >= 3u) {
                StringLattice s = get_str(st, args[1]);
                ScalarLattice idx = get_scalar(st, args[2]);
                if (s.is_set() && !s.tokens.empty() && idx.is_int()) {
                    int32_t c0 = eval_osl_getchar(strings.lookup(s.tokens[0]), idx.i);
                    bool all_same = std::all_of(s.tokens.begin(), s.tokens.end(), [&](StringToken t) {
                        return eval_osl_getchar(strings.lookup(t), idx.i) == c0;
                    });
                    if (all_same) {
                        set_int(args[0], c0);
                        return;
                    }
                }
                kill_sym(st, args[0]);
                return;
            }

            if (op == "stoi" && args.size() >= 2u) {
                StringLattice s = get_str(st, args[1]);
                if (s.is_set() && !s.tokens.empty()) {
                    int32_t v0 = eval_osl_stoi(strings.lookup(s.tokens[0]));
                    bool all_same = std::all_of(s.tokens.begin(), s.tokens.end(), [&](StringToken t) {
                        return eval_osl_stoi(strings.lookup(t)) == v0;
                    });
                    if (all_same) {
                        set_int(args[0], v0);
                        return;
                    }
                }
                kill_sym(st, args[0]);
                return;
            }

            if (op == "stof" && args.size() >= 2u) {
                StringLattice s = get_str(st, args[1]);
                if (s.is_set() && !s.tokens.empty()) {
                    float v0 = eval_osl_stof(strings.lookup(s.tokens[0]));
                    bool all_same = std::all_of(s.tokens.begin(), s.tokens.end(), [&](StringToken t) {
                        return eval_osl_stof(strings.lookup(t)) == v0;
                    });
                    if (all_same) {
                        set_float(args[0], v0);
                        return;
                    }
                }
                kill_sym(st, args[0]);
                return;
            }

            if (op == "hash" && args.size() == 2u &&
                is_simple_type(args[1]->type(), SimpleType::Primitive::STRING)) {
                StringLattice s = get_str(st, args[1]);
                if (s.is_set() && !s.tokens.empty()) {
                    int32_t h0 = eval_osl_string_hash(strings.lookup(s.tokens[0]));
                    bool all_same = std::all_of(s.tokens.begin(), s.tokens.end(), [&](StringToken t) {
                        return eval_osl_string_hash(strings.lookup(t)) == h0;
                    });
                    if (all_same) {
                        set_int(args[0], h0);
                        return;
                    }
                }
                kill_sym(st, args[0]);
                return;
            }

            if (op == "aref" && args.size() >= 3u) {
                const Symbol *dst = args[0];
                const Symbol *arr = args[1];
                if (is_simple_type(dst->type(), SimpleType::Primitive::STRING) && arr->is_array()) {
                    auto it = st.string_arrays.find(arr);
                    if (it == st.string_arrays.end() || it->second.empty()) {
                        st.strings[dst] = StringLattice::top();
                        return;
                    }
                    ScalarLattice idx = get_scalar(st, args[2]);
                    if (idx.is_int()) {
                        int32_t ci = std::clamp<int32_t>(idx.i, 0, static_cast<int32_t>(it->second.size()) - 1);
                        st.strings[dst] = it->second[static_cast<size_t>(ci)];
                    } else {
                        StringLattice joined;
                        for (const auto &el : it->second) { joined.join(el, max_lattice_size); }
                        st.strings[dst] = std::move(joined);
                    }
                    return;
                }
                kill_sym(st, dst);
                return;
            }

            if (op == "aassign" && args.size() >= 3u) {
                const Symbol *arr = args[0];
                if (is_simple_type(arr->type(), SimpleType::Primitive::STRING) && arr->is_array()) {
                    auto &vec = st.string_arrays[arr];
                    if (!vec.empty()) {
                        StringLattice val = get_str(st, args[2]);
                        ScalarLattice idx = get_scalar(st, args[1]);
                        if (idx.is_int()) {
                            int32_t ci = std::clamp<int32_t>(idx.i, 0, static_cast<int32_t>(vec.size()) - 1);
                            vec[static_cast<size_t>(ci)] = val;
                        } else {
                            for (auto &el : vec) { el.join(val, max_lattice_size); }
                        }
                    }
                }
                return;
            }

            if (op == "arraylength" && args.size() >= 2u) {
                int len = 0;
                if (auto it = ll.effective_array_length.find(args[1]); it != ll.effective_array_length.end()) {
                    len = it->second;
                }
                set_int(args[0], len);
                return;
            }

            if (op == "isconnected" && args.size() >= 2u) {
                const Symbol *sym = args[1];
                const Symbol *root = sym->root();
                bool conn = false;
                for (uint32_t ci : ll.incoming_connections) {
                    const auto &c = group.connections[ci];
                    if (c.dst_symbol == sym || c.dst_symbol == root) {
                        conn = true;
                        break;
                    }
                }
                if (!conn) {
                    conn = is_dynamic_input_symbol(ll, sym);
                }
                set_int(args[0], conn ? 1 : 0);
                return;
            }

            if (op == "split" && args.size() >= 3u) {
                const Symbol *dst_count = args[0];
                const Symbol *out_arr = args[2];
                StringLattice s_lat = get_str(st, args[1]);
                StringLattice sep_lat = (args.size() >= 4u)
                                            ? get_str(st, args[3])
                                            : StringLattice::singleton(empty_string_token);
                ScalarLattice ms_lat = (args.size() >= 5u)
                                           ? get_scalar(st, args[4])
                                           : ScalarLattice::make_int(-1);
                int arr_len = 0;
                if (auto it = ll.effective_array_length.find(out_arr); it != ll.effective_array_length.end()) {
                    arr_len = it->second;
                }
                if (out_arr->is_array() && is_simple_type(out_arr->type(), SimpleType::Primitive::STRING) &&
                    s_lat.is_set() && sep_lat.is_set() && ms_lat.is_int() &&
                    s_lat.tokens.size() * sep_lat.tokens.size() <= max_lattice_size &&
                    arr_len >= 0) {
                    auto &arr_vec = st.string_arrays[out_arr];
                    if (arr_vec.size() < static_cast<size_t>(arr_len)) {
                        arr_vec.resize(static_cast<size_t>(arr_len), StringLattice::singleton(empty_string_token));
                    }
                    luisa::vector<StringLattice> old_vec = arr_vec;
                    if (s_lat.is_singleton() && sep_lat.is_singleton()) {
                        auto pieces = eval_osl_split(strings.lookup(*s_lat.as_singleton()),
                                                     strings.lookup(*sep_lat.as_singleton()), ms_lat.i);
                        int n = static_cast<int>(std::min<size_t>(static_cast<size_t>(arr_len), pieces.size()));
                        for (int k = 0; k < n; ++k) {
                            arr_vec[static_cast<size_t>(k)] = StringLattice::singleton(strings.intern(pieces[static_cast<size_t>(k)]));
                        }
                        set_int(dst_count, n);
                    } else {
                        luisa::vector<StringLattice> next_vec(static_cast<size_t>(arr_len), StringLattice::bot());
                        bool have_c = false, same_c = true;
                        int first_c = 0;
                        for (StringToken ts : s_lat.tokens) {
                            for (StringToken tsp : sep_lat.tokens) {
                                auto pieces = eval_osl_split(strings.lookup(ts), strings.lookup(tsp), ms_lat.i);
                                int n = static_cast<int>(std::min<size_t>(static_cast<size_t>(arr_len), pieces.size()));
                                if (!have_c) { first_c = n; have_c = true; }
                                else if (first_c != n) { same_c = false; }
                                for (int k = 0; k < arr_len; ++k) {
                                    StringLattice elem = (k < n)
                                                             ? StringLattice::singleton(strings.intern(pieces[static_cast<size_t>(k)]))
                                                             : old_vec[static_cast<size_t>(k)];
                                    next_vec[static_cast<size_t>(k)].join(elem, max_lattice_size);
                                }
                            }
                        }
                        arr_vec = std::move(next_vec);
                        if (have_c && same_c) { set_int(dst_count, first_c); }
                        else { kill_sym(st, dst_count); }
                    }
                } else {
                    kill_sym(st, dst_count);
                    kill_sym(st, out_arr);
                }
                return;
            }

            if (op == "setmessage" && args.size() == 2u) {
                StringLattice k_lat = get_str(st, args[0]);
                if (!k_lat.is_set() || k_lat.tokens.empty()) {
                    st.msg_strings_tainted = true;
                    return;
                }
                if (!is_simple_type(args[1]->type(), SimpleType::Primitive::STRING)) {
                    for (StringToken t : k_lat.tokens) {
                        st.msg_strings.erase(t);
                        st.definitely_set_msgs.erase(t);
                    }
                    return;
                }
                StringLattice v_lat = get_str(st, args[1]);
                if (k_lat.is_singleton()) {
                    StringToken k = *k_lat.as_singleton();
                    st.msg_strings[k] = v_lat;
                    st.definitely_set_msgs.emplace(k);
                } else {
                    for (StringToken t : k_lat.tokens) {
                        st.msg_strings[t].join(v_lat, max_lattice_size);
                    }
                }
                return;
            }

            if (op == "getmessage" && args.size() == 3u) {
                const Symbol *ok_dst = args[0];
                const Symbol *out_dst = args[2];
                if (is_simple_type(out_dst->type(), SimpleType::Primitive::STRING)) {
                    StringLattice k_lat = get_str(st, args[1]);
                    bool all_known = !st.msg_strings_tainted && k_lat.is_set() && !k_lat.tokens.empty();
                    if (all_known) {
                        for (StringToken t : k_lat.tokens) {
                            if (!st.msg_strings.contains(t)) {
                                all_known = false;
                                break;
                            }
                        }
                    }
                    if (all_known) {
                        if (k_lat.is_singleton() && st.definitely_set_msgs.contains(*k_lat.as_singleton())) {
                            st.strings[out_dst] = st.msg_strings[*k_lat.as_singleton()];
                        } else {
                            StringLattice joined = get_str(st, out_dst);
                            for (StringToken t : k_lat.tokens) {
                                joined.join(st.msg_strings[t], max_lattice_size);
                            }
                            st.strings[out_dst] = std::move(joined);
                        }
                        kill_sym(st, ok_dst);
                        return;
                    }
                }
                kill_sym(st, ok_dst);
                kill_sym(st, out_dst);
                return;
            }

            // Fold basic scalar arithmetic / comparisons for accurate branch pruning
            if (args.size() >= 3u &&
                (is_simple_type(args[0]->type(), SimpleType::Primitive::INT))) {
                ScalarLattice a = get_scalar(st, args[1]);
                ScalarLattice b = get_scalar(st, args[2]);
                auto to_f = [](const ScalarLattice &s) { return s.is_int() ? static_cast<float>(s.i) : s.f; };
                if ((a.is_int() || a.is_float()) && (b.is_int() || b.is_float())) {
                    if (a.is_int() && b.is_int()) {
                        if (op == "add") { set_int(args[0], static_cast<int32_t>(static_cast<uint32_t>(a.i) + static_cast<uint32_t>(b.i))); return; }
                        if (op == "sub") { set_int(args[0], static_cast<int32_t>(static_cast<uint32_t>(a.i) - static_cast<uint32_t>(b.i))); return; }
                        if (op == "mul") { set_int(args[0], static_cast<int32_t>(static_cast<uint32_t>(a.i) * static_cast<uint32_t>(b.i))); return; }
                        if (op == "div") {
                            int32_t res = (b.i == 0) ? 0 : ((a.i == INT32_MIN && b.i == -1) ? INT32_MIN : (a.i / b.i));
                            set_int(args[0], res);
                            return;
                        }
                        if (op == "mod") {
                            int32_t res = (b.i == 0 || (a.i == INT32_MIN && b.i == -1)) ? 0 : (a.i % b.i);
                            set_int(args[0], res);
                            return;
                        }
                        if (op == "bitand") { set_int(args[0], a.i & b.i); return; }
                        if (op == "bitor") { set_int(args[0], a.i | b.i); return; }
                        if (op == "xor") { set_int(args[0], a.i ^ b.i); return; }
                        if (op == "shl") { set_int(args[0], static_cast<int32_t>(static_cast<uint32_t>(a.i) << (b.i & 31))); return; }
                        if (op == "shr") { set_int(args[0], a.i >> (b.i & 31)); return; }
                        if (op == "and") { set_int(args[0], (a.i != 0 && b.i != 0) ? 1 : 0); return; }
                        if (op == "or") { set_int(args[0], (a.i != 0 || b.i != 0) ? 1 : 0); return; }
                    }
                    float fa = to_f(a), fb = to_f(b);
                    if (op == "eq") { set_int(args[0], fa == fb ? 1 : 0); return; }
                    if (op == "neq") { set_int(args[0], fa != fb ? 1 : 0); return; }
                    if (op == "lt") { set_int(args[0], fa < fb ? 1 : 0); return; }
                    if (op == "gt") { set_int(args[0], fa > fb ? 1 : 0); return; }
                    if (op == "le") { set_int(args[0], fa <= fb ? 1 : 0); return; }
                    if (op == "ge") { set_int(args[0], fa >= fb ? 1 : 0); return; }
                }
            }
            if (args.size() >= 2u && is_simple_type(args[0]->type(), SimpleType::Primitive::INT)) {
                ScalarLattice a = get_scalar(st, args[1]);
                if (a.is_int()) {
                    if (op == "neg") { set_int(args[0], static_cast<int32_t>(-static_cast<uint32_t>(a.i))); return; }
                    if (op == "compl") { set_int(args[0], ~a.i); return; }
                    if (op == "not") { set_int(args[0], a.i == 0 ? 1 : 0); return; }
                }
            }

            // Default kill: mark any written operands as TOP via shared classify_inst_rw
            auto rw = classify_inst_rw(inst);
            for (size_t i = 0u; i < args.size(); ++i) {
                if (rw.writes_arg(i)) {
                    kill_sym(st, args[i]);
                }
            }
        };

        // Run worklist across all executed CodeRanges in order
        AbstractState current_state = init_state;
        AbstractState layer_early_exit_state;
        for (const auto &range : ll.ranges) {
            if (!range.is_main) {
                if (range.param_symbol != nullptr &&
                    ll.skip_init_params.contains(range.param_symbol->root())) {
                    continue;
                }
            }
            if (range.blocks.empty()) { continue; }
            luisa::vector<AbstractState> in_states(range.blocks.size());
            in_states[range.entry_block] = current_state;
            luisa::queue<uint32_t> wl;
            luisa::vector<bool> in_wl(range.blocks.size(), false);
            wl.push(range.entry_block);
            in_wl[range.entry_block] = true;

            while (!wl.empty()) {
                uint32_t bid = wl.front();
                wl.pop();
                in_wl[bid] = false;
                AbstractState st = in_states[bid];
                if (!st.reachable) { continue; }
                const auto &blk = range.blocks[bid];
                for (uint32_t pc : blk.pcs) {
                    transfer_inst(pc, st);
                }

                auto propagate = [&](uint32_t succ_id) {
                    if (succ_id == ~0u || succ_id >= range.blocks.size()) { return; }
                    if (in_states[succ_id].join_from(st, max_lattice_size)) {
                        if (!in_wl[succ_id]) {
                            wl.push(succ_id);
                            in_wl[succ_id] = true;
                        }
                    }
                };

                if (blk.term == OSOTermKind::BR) {
                    propagate(blk.succ[0]);
                } else if (blk.term == OSOTermKind::EXIT_LAYER) {
                    layer_early_exit_state.join_from(st, max_lattice_size);
                } else if (blk.term == OSOTermKind::COND_BR) {
                    ScalarLattice c = get_scalar(st, blk.cond_symbol);
                    auto &ia = la.instructions[blk.cond_pc];
                    ia.reachable = true;
                    if (c.is_int() || c.is_float()) {
                        bool take_true = c.is_int() ? (c.i != 0) : (c.f != 0.0f);
                        if (!ia.branch_mixed) {
                            if (!ia.folded_branch.has_value()) {
                                ia.folded_branch = take_true;
                            } else if (*ia.folded_branch != take_true) {
                                ia.folded_branch.reset();
                                ia.branch_mixed = true;
                            }
                        }
                        propagate(take_true ? blk.succ[0] : blk.succ[1]);
                    } else {
                        ia.folded_branch.reset();
                        ia.branch_mixed = true;
                        propagate(blk.succ[0]);
                        propagate(blk.succ[1]);
                    }
                }
            }
            if (in_states[range.exit_block].reachable) {
                current_state = std::move(in_states[range.exit_block]);
            } else {
                current_state.reachable = false;
                break;
            }
        }
        if (!current_state.reachable) {
            current_state = std::move(layer_early_exit_state);
        } else {
            current_state.join_from(layer_early_exit_state, max_lattice_size);
        }

        group_msg_strings = current_state.msg_strings;
        group_definitely_set_msgs = current_state.definitely_set_msgs;
        group_msg_tainted = current_state.msg_strings_tainted;

        // Also clear folded_branch if both branches were dynamically visited across iterations
        // (checked conservatively: if cond_symbol was non-const on any reachable iteration, reset)
        for (const auto &range : ll.ranges) {
            for (const auto &blk : range.blocks) {
                if (blk.term == OSOTermKind::COND_BR) {
                    auto &ia = la.instructions[blk.cond_pc];
                    // Notice: if a loop runs cond multiple times, earlier iteration may set folded_branch=true,
                    // later iteration sets false (or TOP!). Ensure TOP also resets folded_branch:
                    // Wait: during worklist, the final visit with TOP hits `else` branch and resets folded_branch.
                }
            }
        }

        // Record output lattices and which string symbols ever hold non-singleton values
        la.output_strings = std::move(current_state.strings);
        la.output_scalars = std::move(current_state.scalars);
        for (const auto &[sym, lat] : la.output_strings) {
            if (!lat.is_singleton()) { la.dynamic_string_symbols.emplace(sym); }
        }
        for (uint32_t pc = 0u; pc < ll.shader->instructions().size(); ++pc) {
            const auto &inst = *ll.shader->instructions()[pc];
            const auto &ia = la.instructions[pc];
            if (!ia.reachable) { continue; }
            for (size_t i = 0u; i < inst.args().size(); ++i) {
                if (is_simple_type(inst.args()[i]->type(), SimpleType::Primitive::STRING) &&
                    !ia.arg_strings[i].is_singleton()) {
                    la.dynamic_string_symbols.emplace(inst.args()[i]);
                }
            }
        }
    }
}

}// namespace luisa::compute::osl::detail
