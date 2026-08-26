#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/string_scratch.h>
#include <luisa/ast/type.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/special_register.h>
#include <luisa/xir/undefined.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/assert.h>
#include <luisa/xir/instructions/assume.h>
#include <luisa/xir/instructions/atomic.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/break.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/instructions/cast.h>
#include <luisa/xir/instructions/clock.h>
#include <luisa/xir/instructions/continue.h>
#include <luisa/xir/instructions/coro.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/autodiff.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/outline.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/print.h>
#include <luisa/xir/instructions/debug_break.h>
#include <luisa/xir/instructions/ray_query.h>
#include <luisa/xir/instructions/raster_discard.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/resource.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/instructions/thread_group.h>
#include <luisa/xir/instructions/unreachable.h>
#include <luisa/xir/metadata/comment.h>
#include <luisa/xir/metadata/location.h>
#include <luisa/xir/metadata/name.h>
#include <luisa/xir/metadata/curve_basis.h>
#include <luisa/xir/metadata/reg2mem_spill.h>
#include <luisa/xir/passes/dom_tree.h>
#include <luisa/xir/translators/xir2text.h>

namespace luisa::compute::xir {

class XIR2TextTranslator final {

private:
    StringScratch _prelude;
    StringScratch _main;
    luisa::unordered_map<const Value *, uint32_t> _value_uid_map;
    luisa::unordered_map<const Type *, uint32_t> _struct_uid_map;
    luisa::unordered_set<const BasicBlock *> _emitted_blocks;
    bool _debug_info{false};
    bool _flat_blocks{false};

private:
    [[nodiscard]] auto _value_uid(const Value *value) noexcept {
        LUISA_ASSERT(value != nullptr, "Value must not be null.");
        auto next_uid = static_cast<uint32_t>(_value_uid_map.size());
        return _value_uid_map.try_emplace(value, next_uid).first->second;
    }

    [[nodiscard]] auto _value_ident(const Value *value) noexcept {
        auto uid = _value_uid(value);
        if (value->isa<SpecialRegister>()) {
            auto r = static_cast<const SpecialRegister *>(value);
            auto name = xir::to_string(r->derived_special_register_tag());
            return luisa::format("%{}.{}", uid, name);
        }
        if (value->isa<Undefined>()) {
            return luisa::format("%{}.undefined", uid);
        }
        return luisa::format("%{}", uid);
    }

    [[nodiscard]] auto _struct_uid(const Type *type) noexcept {
        LUISA_ASSERT(type != nullptr, "Type must not be null.");
        // custom
        if (type->is_custom()) {
            if (auto iter = _struct_uid_map.find(type); iter != _struct_uid_map.end()) {
                return iter->second;
            }
            auto next_uid = static_cast<uint32_t>(_struct_uid_map.size());
            _prelude << "type T" << next_uid << " = opaque \"" << type->description() << "\";\n\n";
            _struct_uid_map.emplace(type, next_uid);
            return next_uid;
        }
        // structure
        LUISA_ASSERT(type->is_structure(), "Type must be a structure.");
        if (auto iter = _struct_uid_map.find(type); iter != _struct_uid_map.end()) {
            return iter->second;
        }
        luisa::string desc;
        for (auto elem : type->members()) {
            desc.append(_type_ident(elem)).append(", ");
        }
        if (!type->members().empty()) {
            desc.pop_back();
            desc.pop_back();
        }
        auto next_uid = static_cast<uint32_t>(_struct_uid_map.size());
        _prelude << "type T" << next_uid << " = struct { " << desc << " };\n\n";
        _struct_uid_map.emplace(type, next_uid);
        return next_uid;
    }

    void _traverse_value_in_instruction(const Instruction *inst) noexcept {
        static_cast<void>(_value_uid(inst));
        for (auto &use : inst->operand_uses()) {
            if (auto value = use->value();
                value != nullptr && value->isa<BasicBlock>()) {
                _traverse_values_in_basic_block(static_cast<const BasicBlock *>(value));
            }
        }
    }

    void _traverse_values_in_basic_block(const BasicBlock *bb) noexcept {
        if (!_value_uid_map.contains(bb)) {// to avoid infinite recursion in loops
            static_cast<void>(_value_uid(bb));
            for (auto inst : bb->instructions()) {
                _traverse_value_in_instruction(inst);
            }
        }
    }

    void _traverse_values_in_function(const Function *f) noexcept {
        for (auto arg : f->arguments()) {
            static_cast<void>(_value_uid(arg));
        }
        if (auto definition = f->definition()) {
            _traverse_values_in_basic_block(definition->body_block());
        }
    }

    void _traverse_values_in_module(const Module *module) noexcept {
        for (auto c : module->constant_list()) {
            static_cast<void>(_value_uid(c));
        }
        for (auto f : module->function_list()) {
            static_cast<void>(_value_uid(f));
            _traverse_values_in_function(f);
        }
    }

    [[nodiscard]] luisa::string _type_ident(const Type *type) noexcept {
        if (type == nullptr) { return "void"; }
        switch (type->tag()) {
            case Type::Tag::BOOL: return "bool";
            case Type::Tag::INT8: return "i8";
            case Type::Tag::UINT8: return "u8";
            case Type::Tag::INT16: return "i16";
            case Type::Tag::UINT16: return "u16";
            case Type::Tag::INT32: return "i32";
            case Type::Tag::UINT32: return "u32";
            case Type::Tag::INT64: return "i64";
            case Type::Tag::UINT64: return "u64";
            case Type::Tag::FLOAT16: return "f16";
            case Type::Tag::FLOAT32: return "f32";
            case Type::Tag::FLOAT64: return "f64";
            case Type::Tag::FLOAT8_E4M3: return "f8e4m3";
            case Type::Tag::FLOAT8_E5M2: return "f8e5m2";
            case Type::Tag::VECTOR: return luisa::format("vector<{}, {}>", _type_ident(type->element()), type->dimension());
            case Type::Tag::MATRIX: return luisa::format("matrix<{}, {}>", _type_ident(type->element()), type->dimension());
            case Type::Tag::ARRAY: return luisa::format("array<{}, {}>", _type_ident(type->element()), type->dimension());
            case Type::Tag::STRUCTURE: return luisa::format("T{}", _struct_uid(type));
            case Type::Tag::BUFFER: return luisa::format("buffer<{}>", _type_ident(type->element()));
            case Type::Tag::TEXTURE: return luisa::format("texture{}d<{}>", type->dimension(), _type_ident(type->element()));
            case Type::Tag::BINDLESS_ARRAY: return "bindless_array";
            case Type::Tag::ACCEL: return "accel";
            case Type::Tag::CUSTOM: return luisa::format("T{}", _struct_uid(type));
            case Type::Tag::COOPERATIVE_VECTOR: return luisa::string{type->description()};
            case Type::Tag::COOPERATIVE_VECTOR_REF: return "u32";
            case Type::Tag::COOPERATIVE_MATRIX_REF: return "u32";
        }
        LUISA_ERROR_WITH_LOCATION("Unknown type tag.");
    }

    void _emit_indent(int indent) noexcept {
        for (int i = 0; i < indent; i++) { _main << "    "; }
    }

    void _emit_basic_block_ref(const BasicBlock *b) noexcept {
        if (b == nullptr) {
            _main << "null";
        } else {
            _main << _value_ident(b);
        }
    }

    void _emit_use_debug_info(StringScratch &ss, const UseList &uses) noexcept {
        if (_debug_info && !uses.empty()) {
            ss << "// users:";
            for (auto &&u : uses) {
                ss << " " << _value_ident(u->user());
            }
        }
    }

    void _emit_basic_block_use_and_pred_debug_info(StringScratch &ss, const BasicBlock *bb) noexcept {
        _emit_use_debug_info(ss, bb->use_list());
        if (_debug_info) {
            if (bb->use_list().empty()) {
                ss << "// ";
            } else {
                ss << ", ";
            }
            ss << "preds:";
            bb->traverse_predecessors(false, [&](const BasicBlock *pred) noexcept {
                ss << " " << _value_ident(pred);
            });
        }
    }

    void _emit_constant(const Constant *c) noexcept {
        auto t = _type_ident(c->type());
        auto v = _value_ident(c);
        if (!c->metadata_list().empty()) {
            _emit_metadata_list(_prelude, c->metadata_list());
            _prelude << "\n";
        }
        _prelude << "const " << v << ": " << t << " = ";
        auto size = c->type()->size();
        for (auto i = 0u; i < size; i++) {
            auto x = static_cast<const uint8_t *>(c->data())[i];
            _prelude << luisa::format("{:02x}", static_cast<uint32_t>(x));
        }
        _prelude << ";";
        _emit_use_debug_info(_prelude, c->use_list());
        _prelude << "\n\n";
    }

    static void _emit_string_escaped(StringScratch &ss, luisa::string_view s) noexcept {
        luisa::format_to(std::back_inserter(ss.string()), "{:?}", s);
    }

    void _emit_operands(const Instruction *inst) noexcept {
        for (auto &&o : inst->operand_uses()) {
            _main << _value_ident(o->value()) << ", ";
        }
        if (!inst->operand_uses().empty()) {
            _main.pop_back();
            _main.pop_back();
        }
    }

    void _emit_break_inst(const BreakInst *inst) noexcept {
        _main << "break " << _value_ident(inst->target_block());
    }

    void _emit_continue_inst(const ContinueInst *inst) noexcept {
        _main << "continue " << _value_ident(inst->target_block());
    }

    void _emit_unreachable_inst(const UnreachableInst *inst) noexcept {
        _main << "unreachable";
        if (!inst->message().empty()) {
            _main << " ";
            _emit_string_escaped(_main, inst->message());
        }
    }

    void _emit_coro_suspend_inst(const CoroSuspendInst *inst) noexcept {
        _main << "coro_suspend " << inst->token() << " ";
        _emit_string_escaped(_main, inst->name());
        _main << " " << _value_ident(inst->frame());
        for (size_t i = 0u;
             i < inst->frame_export_count(); ++i) {
            _main << ", export ";
            _emit_string_escaped(
                _main, inst->frame_export_name(i));
            _main << "="
                  << _value_ident(inst->frame_export_value(i));
        }
    }

    void _emit_coro_resume_inst(const CoroResumeInst *inst) noexcept {
        _main << "coro_resume " << inst->token() << " " << _value_ident(inst->frame());
    }

    void _emit_coro_terminate_inst(const CoroTerminateInst *inst [[maybe_unused]]) noexcept {
        _main << "coro_terminate";
    }

    void _emit_assert_inst(const AssertInst *inst) noexcept {
        _main << "assert";
        if (!inst->message().empty()) {
            _main << " ";
            _emit_string_escaped(_main, inst->message());
        }
        _main << " " << _value_ident(inst->condition());
    }

    void _emit_assume_inst(const AssumeInst *inst) noexcept {
        _main << "assume";
        if (!inst->message().empty()) {
            _main << " ";
            _emit_string_escaped(_main, inst->message());
        }
        _main << " " << _value_ident(inst->condition());
    }

    void _emit_clock_inst(const ClockInst *inst [[maybe_unused]]) noexcept {
        _main << "clock";
    }

    void _emit_if_inst(const IfInst *inst, int indent) noexcept {
        _main << "if " << _value_ident(inst->condition()) << ", then ";
        _flat_blocks ? _emit_basic_block_ref(inst->true_block()) : _emit_basic_block(inst->true_block(), indent);
        _main << ", else ";
        _flat_blocks ? _emit_basic_block_ref(inst->false_block()) : _emit_basic_block(inst->false_block(), indent);
        _main << ", merge ";
        _flat_blocks ? _emit_basic_block_ref(inst->merge_block()) : _emit_basic_block(inst->merge_block(), indent);
    }

    void _emit_autodiff_scope_inst(const AutodiffScopeInst *inst, int indent) noexcept {
        if (inst->is_forward()) {
            _main << "autodiff forward " << inst->n_forward_grads() << " entry ";
        } else {
            _main << "autodiff entry ";
        }
        _flat_blocks ? _emit_basic_block_ref(inst->entry_block()) : _emit_basic_block(inst->entry_block(), indent);
        _main << ", merge ";
        _flat_blocks ? _emit_basic_block_ref(inst->merge_block()) : _emit_basic_block(inst->merge_block(), indent);
    }

    void _emit_indexed_branch_edges(
        const IndexedBranchTerminatorInstruction *inst,
        int indent) noexcept {
        _main << _value_ident(inst->value()) << ", ";
        for (auto i = 0u; i < inst->case_count(); i++) {
            auto value = inst->case_value(i);
            _main << "case ";
            switch (inst->value()->type()->tag()) {
                case Type::Tag::INT8:
                    _main << luisa::format(
                        "{}", static_cast<int64_t>(luisa::bit_cast<int8_t>(
                                  static_cast<uint8_t>(value))));
                    break;
                case Type::Tag::INT16:
                    _main << luisa::format(
                        "{}", static_cast<int64_t>(luisa::bit_cast<int16_t>(
                                  static_cast<uint16_t>(value))));
                    break;
                case Type::Tag::INT32:
                    _main << luisa::format(
                        "{}", static_cast<int64_t>(luisa::bit_cast<int32_t>(
                                  static_cast<uint32_t>(value))));
                    break;
                case Type::Tag::INT64:
                    _main << luisa::format(
                        "{}", luisa::bit_cast<int64_t>(value));
                    break;
                default: _main << luisa::format("{}", value); break;
            }
            _main << " ";
            _flat_blocks ? _emit_basic_block_ref(inst->case_block(i)) : _emit_basic_block(inst->case_block(i), indent);
            _main << ", ";
        }
        _main << "default ";
        _flat_blocks ? _emit_basic_block_ref(inst->default_block()) : _emit_basic_block(inst->default_block(), indent);
    }

    void _emit_switch_inst(const SwitchInst *inst, int indent) noexcept {
        _main << "switch ";
        _emit_indexed_branch_edges(inst, indent);
        _main << ", merge ";
        _flat_blocks ? _emit_basic_block_ref(inst->merge_block()) : _emit_basic_block(inst->merge_block(), indent);
    }

    void _emit_indexed_branch_inst(
        const IndexedBranchInst *inst, int indent) noexcept {
        _main << "indexed_branch ";
        _emit_indexed_branch_edges(inst, indent);
    }

    void _emit_loop_inst(const LoopInst *inst, int indent) noexcept {
        _main << "loop prepare ";
        _flat_blocks ? _emit_basic_block_ref(inst->prepare_block()) : _emit_basic_block(inst->prepare_block(), indent);
        _main << ", body ";
        _flat_blocks ? _emit_basic_block_ref(inst->body_block()) : _emit_basic_block(inst->body_block(), indent);
        _main << ", update ";
        _flat_blocks ? _emit_basic_block_ref(inst->update_block()) : _emit_basic_block(inst->update_block(), indent);
        _main << ", merge ";
        _flat_blocks ? _emit_basic_block_ref(inst->merge_block()) : _emit_basic_block(inst->merge_block(), indent);
    }

    void _emit_simple_loop_inst(const SimpleLoopInst *inst, int indent) noexcept {
        _main << "simple_loop body ";
        _flat_blocks ? _emit_basic_block_ref(inst->body_block()) : _emit_basic_block(inst->body_block(), indent);
        _main << ", merge ";
        _flat_blocks ? _emit_basic_block_ref(inst->merge_block()) : _emit_basic_block(inst->merge_block(), indent);
    }

    void _emit_outline_inst(const OutlineInst *inst, int indent) noexcept {
        _main << "outline body ";
        _flat_blocks ? _emit_basic_block_ref(inst->target_block()) : _emit_basic_block(inst->target_block(), indent);
        _main << ", merge ";
        _flat_blocks ? _emit_basic_block_ref(inst->merge_block()) : _emit_basic_block(inst->merge_block(), indent);
    }

    void _emit_ray_query_dispatch_inst(const RayQueryDispatchInst *inst, int indent) noexcept {
        _main << "ray_query_dispatch " << _value_ident(inst->query_object())
              << ", exit " << _value_ident(inst->exit_block())
              << ", on_surface_candidate ";
        _flat_blocks ? _emit_basic_block_ref(inst->on_surface_candidate_block()) : _emit_basic_block(inst->on_surface_candidate_block(), indent);
        _main << ", on_procedural_candidate ";
        _flat_blocks ? _emit_basic_block_ref(inst->on_procedural_candidate_block()) : _emit_basic_block(inst->on_procedural_candidate_block(), indent);
    }

    void _emit_ray_query_loop_inst(const RayQueryLoopInst *inst, int indent) noexcept {
        _main << "ray_query_loop dispatch ";
        _flat_blocks ? _emit_basic_block_ref(inst->dispatch_block()) : _emit_basic_block(inst->dispatch_block(), indent);
        _main << ", merge ";
        _flat_blocks ? _emit_basic_block_ref(inst->merge_block()) : _emit_basic_block(inst->merge_block(), indent);
    }

    void _emit_ray_query_object_read_inst(const RayQueryObjectReadInst *inst) noexcept {
        _main << "ray_query_object_read " << xir::to_string(inst->op()) << " ";
        _emit_operands(inst);
    }

    void _emit_ray_query_object_write_inst(const RayQueryObjectWriteInst *inst) noexcept {
        _main << "ray_query_object_write " << xir::to_string(inst->op()) << " ";
        _emit_operands(inst);
    }

    void _emit_ray_query_pipeline_inst(const RayQueryPipelineInst *inst) noexcept {
        _main << "ray_query_pipeline ";
        _emit_operands(inst);
    }

    void _emit_return_inst(const ReturnInst *inst) noexcept {
        if (auto ret = inst->return_value()) {
            _main << "return " << _value_ident(ret);
        } else {
            _main << "return";
        }
    }

    void _emit_raster_discard_inst(const RasterDiscardInst *inst [[maybe_unused]]) noexcept {
        _main << "raster_discard";
    }

    void _emit_phi_inst(const PhiInst *inst) noexcept {
        _main << "phi";
        for (auto i = 0u; i < inst->incoming_count(); i++) {
            auto incoming = inst->incoming(i);
            _main << " (" << _value_ident(incoming.value) << ", "
                  << _value_ident(incoming.block) << "),";
        }
        if (inst->incoming_count() != 0u) { _main.pop_back(); }
    }

    void _emit_alloca_inst(const AllocaInst *inst) noexcept {
        _main << "alloca ";
        switch (inst->op()) {
            case AllocaOp::LOCAL: _main << "local"; break;
            case AllocaOp::SHARED: _main << "shared"; break;
        }
    }

    void _emit_load_inst(const LoadInst *inst) noexcept {
        _main << "load ";
        _emit_operands(inst);
    }

    void _emit_store_inst(const StoreInst *inst) noexcept {
        _main << "store ";
        _emit_operands(inst);
    }

    void _emit_gep_inst(const GEPInst *inst) noexcept {
        _main << "getelementptr ";
        _emit_operands(inst);
    }

    void _emit_atomic_inst(const AtomicInst *inst) noexcept {
        _main << "atomic " << xir::to_string(inst->op()) << " ";
        _emit_operands(inst);
    }

    void _emit_thread_group_inst(const ThreadGroupInst *inst) noexcept {
        _main << "thread_group " << xir::to_string(inst->op()) << " ";
        _emit_operands(inst);
    }

    void _emit_call_inst(const CallInst *inst) noexcept {
        _main << "call ";
        _emit_operands(inst);
    }

    void _emit_autodiff_intrinsic_inst(const AutodiffIntrinsicInst *inst) noexcept {
        _main << "@" << to_string(inst->op());
        if (!inst->operand_uses().empty()) {
            _main << " ";
            _emit_operands(inst);
        }
    }

    void _emit_cast_inst(const CastInst *inst) noexcept {
        _main << "cast " << xir::to_string(inst->op()) << " ";
        _emit_operands(inst);
    }

    void _emit_print_inst(const PrintInst *inst) noexcept {
        _main << "print ";
        _emit_string_escaped(_main, inst->format());
        _main << " ";
        _emit_operands(inst);
    }

    void _emit_debug_break_inst(const DebugBreakInst *inst) noexcept {
        _main << "debug_break ";
        _emit_operands(inst);
    }

    void _emit_branch_inst(const BranchInst *inst) noexcept {
        LUISA_DEBUG_ASSERT(inst->target_block() != nullptr,
                           "Branch target block must not be null.");
        _main << "br " << _value_ident(inst->target_block());
    }

    void _emit_arithmetic_inst(const ArithmeticInst *inst) noexcept {
        _main << "arithmetic " << xir::to_string(inst->op()) << " ";
        _emit_operands(inst);
    }

    void _emit_bindless_access(BindlessResourceAccess access) noexcept {
        if (access.is_default()) { return; }
        _main << " [";
        if (access.typed) { _main << "typed"; }
        if (access.typed && access.uniform) { _main << ", "; }
        if (access.uniform) { _main << "uniform"; }
        _main << "]";
    }

    void _emit_resource_query_inst(const ResourceQueryInst *inst) noexcept {
        _main << "resource_query " << xir::to_string(inst->op());
        _emit_bindless_access(inst->bindless_access());
        _main << " ";
        _emit_operands(inst);
    }

    void _emit_resource_read_inst(const ResourceReadInst *inst) noexcept {
        _main << "resource_read " << xir::to_string(inst->op());
        _emit_bindless_access(inst->bindless_access());
        _main << " ";
        _emit_operands(inst);
    }

    void _emit_resource_write_inst(const ResourceWriteInst *inst) noexcept {
        _main << "resource_write " << xir::to_string(inst->op());
        _emit_bindless_access(inst->bindless_access());
        _main << " ";
        _emit_operands(inst);
    }

    void _emit_conditional_branch_inst(const ConditionalBranchInst *inst) noexcept {
        LUISA_DEBUG_ASSERT(inst->true_block() != nullptr && inst->false_block() != nullptr,
                           "Conditional branch target blocks must not be null.");
        _main << "cond_br "
              << _value_ident(inst->condition()) << ", "
              << _value_ident(inst->true_block()) << ", "
              << _value_ident(inst->false_block());
    }

    void _emit_instruction(const Instruction *inst, int indent) noexcept {
        if (!inst->metadata_list().empty()) {
            _emit_indent(indent);
            _emit_metadata_list(_main, inst->metadata_list());
            _main << "\n";
        }
        _emit_indent(indent);
        _main << _value_ident(inst) << ": " << _type_ident(inst->type()) << " = ";
        switch (inst->derived_instruction_tag()) {
            case DerivedInstructionTag::UNREACHABLE:
                _emit_unreachable_inst(static_cast<const UnreachableInst *>(inst));
                break;
            case DerivedInstructionTag::CORO_SUSPEND:
                _emit_coro_suspend_inst(static_cast<const CoroSuspendInst *>(inst));
                break;
            case DerivedInstructionTag::CORO_RESUME:
                _emit_coro_resume_inst(static_cast<const CoroResumeInst *>(inst));
                break;
            case DerivedInstructionTag::CORO_TERMINATE:
                _emit_coro_terminate_inst(static_cast<const CoroTerminateInst *>(inst));
                break;
            case DerivedInstructionTag::IF:
                _emit_if_inst(static_cast<const IfInst *>(inst), indent);
                break;
            case DerivedInstructionTag::SWITCH:
                _emit_switch_inst(static_cast<const SwitchInst *>(inst), indent);
                break;
            case DerivedInstructionTag::INDEXED_BRANCH:
                _emit_indexed_branch_inst(
                    static_cast<const IndexedBranchInst *>(inst), indent);
                break;
            case DerivedInstructionTag::LOOP:
                _emit_loop_inst(static_cast<const LoopInst *>(inst), indent);
                break;
            case DerivedInstructionTag::SIMPLE_LOOP:
                _emit_simple_loop_inst(static_cast<const SimpleLoopInst *>(inst), indent);
                break;
            case DerivedInstructionTag::BREAK:
                _emit_break_inst(static_cast<const BreakInst *>(inst));
                break;
            case DerivedInstructionTag::CONTINUE:
                _emit_continue_inst(static_cast<const ContinueInst *>(inst));
                break;
            case DerivedInstructionTag::RETURN:
                _emit_return_inst(static_cast<const ReturnInst *>(inst));
                break;
            case DerivedInstructionTag::RASTER_DISCARD:
                _emit_raster_discard_inst(static_cast<const RasterDiscardInst *>(inst));
                break;
            case DerivedInstructionTag::PHI:
                _emit_phi_inst(static_cast<const PhiInst *>(inst));
                break;
            case DerivedInstructionTag::ALLOCA:
                _emit_alloca_inst(static_cast<const AllocaInst *>(inst));
                break;
            case DerivedInstructionTag::LOAD:
                _emit_load_inst(static_cast<const LoadInst *>(inst));
                break;
            case DerivedInstructionTag::STORE:
                _emit_store_inst(static_cast<const StoreInst *>(inst));
                break;
            case DerivedInstructionTag::GEP:
                _emit_gep_inst(static_cast<const GEPInst *>(inst));
                break;
            case DerivedInstructionTag::CALL:
                _emit_call_inst(static_cast<const CallInst *>(inst));
                break;
            case DerivedInstructionTag::CAST:
                _emit_cast_inst(static_cast<const CastInst *>(inst));
                break;
            case DerivedInstructionTag::PRINT:
                _emit_print_inst(static_cast<const PrintInst *>(inst));
                break;
            case DerivedInstructionTag::OUTLINE:
                _emit_outline_inst(static_cast<const OutlineInst *>(inst), indent);
                break;
            case DerivedInstructionTag::AUTODIFF_SCOPE:
                _emit_autodiff_scope_inst(static_cast<const AutodiffScopeInst *>(inst), indent);
                break;
            case DerivedInstructionTag::AUTODIFF_INTRINSIC:
                _emit_autodiff_intrinsic_inst(static_cast<const AutodiffIntrinsicInst *>(inst));
                break;
            case DerivedInstructionTag::RAY_QUERY_LOOP:
                _emit_ray_query_loop_inst(static_cast<const RayQueryLoopInst *>(inst), indent);
                break;
            case DerivedInstructionTag::RAY_QUERY_DISPATCH:
                _emit_ray_query_dispatch_inst(static_cast<const RayQueryDispatchInst *>(inst), indent);
                break;
            case DerivedInstructionTag::RAY_QUERY_OBJECT_READ:
                _emit_ray_query_object_read_inst(static_cast<const RayQueryObjectReadInst *>(inst));
                break;
            case DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE:
                _emit_ray_query_object_write_inst(static_cast<const RayQueryObjectWriteInst *>(inst));
                break;
            case DerivedInstructionTag::RAY_QUERY_PIPELINE:
                _emit_ray_query_pipeline_inst(static_cast<const RayQueryPipelineInst *>(inst));
                break;
            case DerivedInstructionTag::BRANCH:
                _emit_branch_inst(static_cast<const BranchInst *>(inst));
                break;
            case DerivedInstructionTag::CONDITIONAL_BRANCH:
                _emit_conditional_branch_inst(static_cast<const ConditionalBranchInst *>(inst));
                break;
            case DerivedInstructionTag::ASSERT:
                _emit_assert_inst(static_cast<const AssertInst *>(inst));
                break;
            case DerivedInstructionTag::ASSUME:
                _emit_assume_inst(static_cast<const AssumeInst *>(inst));
                break;
            case DerivedInstructionTag::CLOCK:
                _emit_clock_inst(static_cast<const ClockInst *>(inst));
                break;
            case DerivedInstructionTag::ATOMIC:
                _emit_atomic_inst(static_cast<const AtomicInst *>(inst));
                break;
            case DerivedInstructionTag::THREAD_GROUP:
                _emit_thread_group_inst(static_cast<const ThreadGroupInst *>(inst));
                break;
            case DerivedInstructionTag::ARITHMETIC:
                _emit_arithmetic_inst(static_cast<const ArithmeticInst *>(inst));
                break;
            case DerivedInstructionTag::RESOURCE_QUERY:
                _emit_resource_query_inst(static_cast<const ResourceQueryInst *>(inst));
                break;
            case DerivedInstructionTag::RESOURCE_READ:
                _emit_resource_read_inst(static_cast<const ResourceReadInst *>(inst));
                break;
            case DerivedInstructionTag::RESOURCE_WRITE:
                _emit_resource_write_inst(static_cast<const ResourceWriteInst *>(inst));
                break;
            case DerivedInstructionTag::DEBUG_BREAK:
                _emit_debug_break_inst(static_cast<const DebugBreakInst *>(inst));
                break;
        }
        _main << ";";
        _emit_use_debug_info(_main, inst->use_list());
        _main << "\n";
    }

    void _emit_basic_block(const BasicBlock *b, int indent) noexcept {
        if (b == nullptr) {
            _main << "null";
        } else if (!_emitted_blocks.insert(b).second) {
            // already emitted (e.g., shared merge target referenced by a flat branch);
            // print just the label reference so the dump stays compact.
            _main << _value_ident(b);
        } else {
            if (!b->metadata_list().empty()) {
                _emit_metadata_list(_main, b->metadata_list());
                _main << " ";
            }
            _main << _value_ident(b) << ": {";
            _emit_basic_block_use_and_pred_debug_info(_main, b);
            _main << "\n";
            for (auto &&inst : b->instructions()) {
                _emit_instruction(inst, indent + 1);
            }
            _emit_indent(indent);
            _main << "}";
        }
    }

    void _emit_flat_basic_block(const BasicBlock *b) noexcept {
        if (b == nullptr) { return; }
        if (!b->metadata_list().empty()) {
            _emit_metadata_list(_main, b->metadata_list());
            _main << "\n";
        }
        _main << _value_ident(b) << ":";
        _emit_basic_block_use_and_pred_debug_info(_main, b);
        _main << "\n";
        for (auto &&inst : b->instructions()) {
            _emit_instruction(inst, 1);
        }
    }

    void _emit_function(const Function *f) noexcept {
        if (!f->metadata_list().empty()) {
            _emit_metadata_list(_main, f->metadata_list());
            _main << "\n";
        }
        switch (f->derived_function_tag()) {
            case DerivedFunctionTag::KERNEL: _main << "kernel " << _value_ident(f); break;
            case DerivedFunctionTag::CALLABLE: _main << "callable " << _value_ident(f) << ": " << _type_ident(f->type()); break;
            case DerivedFunctionTag::RASTER_STAGE: {
                auto raster = static_cast<const RasterStageFunction *>(f);
                _main << "raster_" << to_string(raster->stage()) << " "
                      << _value_ident(f) << ": " << _type_ident(f->type());
                break;
            }
            case DerivedFunctionTag::EXTERNAL: _main << "external " << _value_ident(f) << ": " << _type_ident(f->type()); break;
        }
        _main << " (";
        // TODO: metadata
        if (!f->arguments().empty()) { _main << "\n"; }
        for (auto arg : f->arguments()) {
            if (!arg->metadata_list().empty()) {
                _emit_indent(1);
                _emit_metadata_list(_main, arg->metadata_list());
                _main << "\n";
            }
            _emit_indent(1);
            _main << _value_ident(arg) << ": ";
            if (arg->isa<ReferenceArgument>()) {
                _main << "&";
            }
            _main << _type_ident(arg->type()) << ";";
            _emit_use_debug_info(_main, arg->use_list());
            _main << "\n";
        }
        _main << ")";
        if (auto definition = f->definition()) {
            if (_flat_blocks) {
                _main << " = define {\n";
                const_cast<FunctionDefinition *>(definition)
                    ->traverse_basic_blocks([&](BasicBlock *block) noexcept {
                        _emit_flat_basic_block(block);
                    });
                _main << "}";
            } else {
                _main << " = define ";
                _emitted_blocks.clear();
                _emit_basic_block(definition->body_block(), 0);
                const_cast<FunctionDefinition *>(definition)
                    ->traverse_basic_blocks([&](BasicBlock *block) noexcept {
                        if (_emitted_blocks.contains(block)) { return; }
                        _main << "\n";
                        _emit_basic_block(block, 0);
                    });
            }
        }
        _main << ";";
        _emit_use_debug_info(_main, f->use_list());
        _main << "\n\n";
        if (auto definition = f->definition();
            definition != nullptr && _debug_info) {
            _emit_control_flow_graph_debug_info(
                const_cast<FunctionDefinition *>(definition));
        }
    }

    void _emit_control_flow_graph_debug_info(FunctionDefinition *f) noexcept {
        // CFG Nodes
        _main << R"(// CFG = {"function": ")" << _value_ident(f) << "\", ";
        {
            _main << "\"nodes\": [";
            f->traverse_basic_blocks([&](auto block) noexcept {
                _main << "\"" << _value_ident(block) << "\", ";
            });
            _main.pop_back();
            _main.pop_back();
            _main << "], ";
        }
        // CFG Edges
        {
            _main << "\"edges\": {";
            f->traverse_basic_blocks([&](auto block) noexcept {
                _main << "\"" << _value_ident(block) << "\": [";
                bool any_succ = false;
                block->traverse_successors(false, [&](auto succ) noexcept {
                    any_succ = true;
                    _main << "\"" << _value_ident(succ) << "\", ";
                });
                if (any_succ) {
                    _main.pop_back();
                    _main.pop_back();
                }
                _main << "], ";
            });
            _main.pop_back();
            _main.pop_back();
            _main << "}, ";
        }
        // CFG Terminators
        {
            _main << "\"terminators\": {";
            f->traverse_basic_blocks([&](auto block) noexcept {
                _main << "\"" << _value_ident(block) << "\": ";
                if (auto term = block->terminator()) {
                    _main << "\"" << xir::to_string(term->derived_instruction_tag()) << "\", ";
                }
            });
            _main.pop_back();
            _main.pop_back();
            _main << "}, ";
        }
        // CFG Control Merges
        {
            _main << "\"merges\": {";
            auto any_merge = false;
            f->traverse_basic_blocks([&](auto block) noexcept {
                auto terminator = block->terminator();
                if (auto merge = terminator->control_flow_merge();
                    merge != nullptr && merge->merge_block() != nullptr) {
                    any_merge = true;
                    _main << "\"" << _value_ident(block) << "\": \""
                          << _value_ident(merge->merge_block()) << "\", ";
                }
            });
            if (any_merge) {
                _main.pop_back();
                _main.pop_back();
            }
            _main << "}, ";
        }
        // Dominance Tree
        {
            auto dom_tree = compute_dom_tree(f);
            _main << "\"dominance_tree\": {";
            for (auto &&[b, node] : dom_tree.nodes()) {
                _main << "\"" << _value_ident(b) << "\": [";
                for (auto &&child : node->children()) {
                    _main << "\"" << _value_ident(child->block()) << "\", ";
                }
                if (!node->children().empty()) {
                    _main.pop_back();
                    _main.pop_back();
                }
                _main << "], ";
            }
            _main.pop_back();
            _main.pop_back();
            // Dominance Frontiers
            _main << "}, \"dominance_frontiers\": {";
            for (auto &&[b, node] : dom_tree.nodes()) {
                _main << "\"" << _value_ident(b) << "\": [";
                for (auto &&frontier : node->frontiers()) {
                    _main << "\"" << _value_ident(frontier->block()) << "\", ";
                }
                if (!node->frontiers().empty()) {
                    _main.pop_back();
                    _main.pop_back();
                }
                _main << "], ";
            }
            _main.pop_back();
            _main.pop_back();
            _main << "}";
        }
        _main << "}\n\n";
    }

    void _emit_module(const Module *module) noexcept {
        _traverse_values_in_module(module);
        if (!module->metadata_list().empty()) {
            _emit_metadata_list(_prelude, module->metadata_list());
            _prelude << "\n";
        }
        _prelude << "module;\n\n";// TODO: metadata
        for (auto c : module->constant_list()) { _emit_constant(c); }
        for (auto f : module->function_list()) { _emit_function(f); }
    }

    static void _emit_name_metadata(StringScratch &s, const NameMD *m) noexcept {
        s << "name = " << m->name();
    }

    static void _emit_location_metadata(StringScratch &s, const LocationMD *m) noexcept {
        s << "location = (";
        _emit_string_escaped(s, m->file().string());
        s << ", " << m->line() << ")";
    }

    static void _emit_comment_metadata(StringScratch &s, const CommentMD *m) noexcept {
        s << "comment = ";
        _emit_string_escaped(s, m->comment());
    }

    static void _emit_curve_basis_metadata(StringScratch &s, const CurveBasisMD *m) noexcept {
        s << "curve_basis = {";
        auto any_basis = false;
        if (m->curve_basis_set().test(CurveBasis::PIECEWISE_LINEAR)) {
            any_basis = true;
            s << "piecewise_linear, ";
        }
        if (m->curve_basis_set().test(CurveBasis::CUBIC_BSPLINE)) {
            any_basis = true;
            s << "cubic_bspline, ";
        }
        if (m->curve_basis_set().test(CurveBasis::CATMULL_ROM)) {
            any_basis = true;
            s << "catmull_rom, ";
        }
        if (m->curve_basis_set().test(CurveBasis::BEZIER)) {
            any_basis = true;
            s << "bezier, ";
        }
        if (any_basis) {
            s.pop_back();
            s.pop_back();
        }
        s << "}";
    }

    template<typename T>
    static void _emit_metadata_list(StringScratch &s, const T &m) noexcept {
        s << "[";
        for (auto item : m) {
            switch (item->derived_metadata_tag()) {
                case DerivedMetadataTag::NAME:
                    _emit_name_metadata(s, static_cast<const NameMD *>(item));
                    break;
                case DerivedMetadataTag::LOCATION:
                    _emit_location_metadata(s, static_cast<const LocationMD *>(item));
                    break;
                case DerivedMetadataTag::COMMENT:
                    _emit_comment_metadata(s, static_cast<const CommentMD *>(item));
                    break;
                case DerivedMetadataTag::CURVE_BASIS:
                    _emit_curve_basis_metadata(s, static_cast<const CurveBasisMD *>(item));
                    break;
                case DerivedMetadataTag::SIGNATURE_CONSTRAINT:
                    s << "signature_constraint";
                    break;
                case DerivedMetadataTag::REG2MEM_SPILL:
                    s << "reg2mem_spill = "
                      << to_string(static_cast<const Reg2MemSpillMD *>(item)->kind());
                    break;
            }
            s << ", ";
        }
        if (!m.empty()) {
            s.pop_back();
            s.pop_back();
        }
        s << "]";
    }

public:
    XIR2TextTranslator() noexcept : _prelude{1_k}, _main{4_k} {}

    [[nodiscard]] luisa::string emit(const Module *module, bool debug_info, bool flat_blocks) noexcept {
        _prelude.clear();
        _main.clear();
        _value_uid_map.clear();
        _struct_uid_map.clear();
        _emitted_blocks.clear();
        _debug_info = debug_info;
        _flat_blocks = flat_blocks;
        _emit_module(module);
        return _prelude.string() + _main.string();
    }
};

luisa::string xir_to_text_translate(const Module *module, bool debug_info) noexcept {
    return XIR2TextTranslator{}.emit(module, debug_info, false);
}

luisa::string xir_to_flat_text_translate(const Module *module, bool debug_info) noexcept {
    return XIR2TextTranslator{}.emit(module, debug_info, true);
}

}// namespace luisa::compute::xir
