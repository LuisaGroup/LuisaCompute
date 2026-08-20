//
// Created by mike on 4/1/25.
//

#ifdef LUISA_ENABLE_XIR

#include <luisa/core/stl/algorithm.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/rtx/ray.h>
#include <luisa/runtime/rtx/hit.h>
#include <luisa/runtime/dispatch_buffer.h>
#include <luisa/dsl/rtx/ray_query.h>
#include <luisa/xir/module.h>
#include <luisa/xir/function.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/metadata/curve_basis.h>
#include <luisa/xir/metadata/name.h>
#include <luisa/xir/metadata/comment.h>
#include <luisa/xir/metadata/location.h>
#include <luisa/xir/metadata/reg2mem_spill.h>

#include "cuda_texture.h"
#include "cuda_codegen_xir.h"

namespace luisa::compute::cuda {

CUDACodegenXIR::CUDACodegenXIR(StringScratch &scratch, bool allow_indirect) noexcept
    : _scratch{scratch},
      _allow_indirect_dispatch{allow_indirect},
      _ray_type{Type::of<Ray>()},
      _triangle_hit_type{Type::of<TriangleHit>()},
      _procedural_hit_type{Type::of<ProceduralHit>()},
      _committed_hit_type{Type::of<CommittedHit>()},
      _ray_query_all_type{Type::of<RayQueryAll>()},
      _ray_query_any_type{Type::of<RayQueryAny>()},
      _indirect_buffer_type{Type::of<IndirectDispatchBuffer>()},
      _motion_srt_type{Type::of<MotionInstanceTransformSRT>()} {}

CUDACodegenXIR::~CUDACodegenXIR() noexcept = default;

void CUDACodegenXIR::_analyze_instruction_usage(const xir::Function *f, InstructionUsageAnalysis &analysis,
                                                luisa::unordered_set<const xir::Function *> &visited) noexcept {

    // skip if already visited
    if (!visited.emplace(f).second) { return; }

    // collect types from function arguments and return type
    for (auto arg : f->arguments()) {
        LUISA_ASSERT(arg != nullptr, "Function argument is null.");
        analysis.used_types.emplace(arg->type());
    }
    analysis.used_types.emplace(f->type());

    // collect instruction usage info from function body
    if (auto def = f->definition()) {
        def->traverse_instructions([&](const xir::Instruction *inst) noexcept {
            switch (inst->derived_instruction_tag()) {
                case xir::DerivedInstructionTag::PRINT: {
                    this->_requires_printing = true;
                    auto print = static_cast<const xir::PrintInst *>(inst);
                    auto [iter, success] = _print_info.emplace(print, PrintInfo{nullptr, _print_formats.size()});
                    LUISA_ASSERT(success, "Print info already exists.");
                    luisa::vector<const Type *> arg_types;
                    arg_types.reserve(print->operand_count() + 2u);
                    arg_types.emplace_back(Type::of<uint>());// arg size
                    arg_types.emplace_back(Type::of<uint>());// fmt id
                    for (auto op_use : print->operand_uses()) {
                        LUISA_ASSERT(op_use->value() != nullptr, "Print operand use is null.");
                        arg_types.emplace_back(op_use->value()->type());
                    }
                    auto s = Type::structure(arg_types);
                    analysis.used_types.emplace(s);
                    iter->second.type = s;
                    _print_formats.emplace_back(print->format(), s);
                    break;
                }
                case xir::DerivedInstructionTag::RESOURCE_QUERY: {
                    auto is_ray_trace = false;
                    switch (static_cast<const xir::ResourceQueryInst *>(inst)->op()) {
                        case xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST: [[fallthrough]];
                        case xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR: {
                            this->_requires_optix = true;
                            analysis.requires_raytracing_closest = true;
                            is_ray_trace = true;
                            break;
                        }
                        case xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY: [[fallthrough]];
                        case xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR: {
                            this->_requires_optix = true;
                            analysis.requires_raytracing_any = true;
                            is_ray_trace = true;
                            break;
                        }
                        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL: [[fallthrough]];
                        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY: [[fallthrough]];
                        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR: [[fallthrough]];
                        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR: {
                            this->_requires_optix = true;
                            analysis.requires_raytracing_query = true;
                            is_ray_trace = true;
                            break;
                        }
                        default: break;
                    }
                    if (is_ray_trace) {
                        if (auto curve_basis_md = inst->find_metadata<xir::CurveBasisMD>()) {
                            analysis.required_curve_bases.propagate(curve_basis_md->curve_basis_set());
                        }
                    }
                    break;
                }
                case xir::DerivedInstructionTag::RAY_QUERY_PIPELINE: {
                    this->_requires_optix = true;
                    analysis.requires_raytracing_query = true;
                    auto pipeline = static_cast<const xir::RayQueryPipelineInst *>(inst);
                    if (auto curve_basis_md = pipeline->find_metadata<xir::CurveBasisMD>()) {
                        analysis.required_curve_bases.propagate(curve_basis_md->curve_basis_set());
                    }
                    analysis.ray_query_pipelines.emplace_back(pipeline);
                    break;
                }
                default: break;
            }
            // collect types from instruction operands and results
            analysis.used_types.emplace(inst->type());
            for (auto op_use : inst->operand_uses()) {
                if (auto value = op_use->value()) {
                    analysis.used_types.emplace(value->type());
                    switch (value->derived_value_tag()) {
                        case xir::DerivedValueTag::FUNCTION: {
                            _analyze_instruction_usage(static_cast<const xir::Function *>(value), analysis, visited);
                            break;
                        }
                        case xir::DerivedValueTag::CONSTANT: {
                            analysis.used_constants.emplace(static_cast<const xir::Constant *>(value));
                            break;
                        }
                        default: break;
                    }
                }
            }
        });
    }

    // collect this used function
    // note: we collect functions in the post-order traversal order so
    // that we can naturally emit them in a topologically sorted order
    analysis.used_functions_post_order.emplace_back(f);
}

bool CUDACodegenXIR::_should_emit_global_constant(const xir::Constant *c) noexcept {
    return !c->type()->is_basic();
}

bool CUDACodegenXIR::_is_ray_query_callback_function(const xir::CallableFunction *f) const noexcept {
    auto any_rq_user = false;
    auto all_rq_user = true;
    for (auto &&use : f->use_list()) {
        if (auto user = use->user(); user != nullptr && user->isa<xir::RayQueryPipelineInst>()) {
            any_rq_user = true;
        } else {
            all_rq_user = false;
        }
    }
    if (any_rq_user) {
        LUISA_ASSERT(all_rq_user, "Ray query callback function is not used by all ray query pipelines.");
        LUISA_ASSERT(f->type() == Type::of<void>(), "Ray query callback function must return void.");
        LUISA_ASSERT(!f->arguments().empty(), "Ray query callback function must have at least one argument.");
        auto rq_arg = f->arguments().front();
        LUISA_ASSERT(rq_arg != nullptr && rq_arg->type() != nullptr,
                     "Ray query callback function must have a valid ray query argument.");
        LUISA_ASSERT(rq_arg->type() == _ray_query_any_type || rq_arg->type() == _ray_query_all_type,
                     "Ray query callback function must have a ray query argument.");
    }
    return any_rq_user;
}

bool CUDACodegenXIR::_is_builtin_type(const Type *t) const noexcept {
    return t == _ray_type ||
           t == _triangle_hit_type ||
           t == _procedural_hit_type ||
           t == _committed_hit_type ||
           t == _ray_query_all_type ||
           t == _ray_query_any_type ||
           t == _indirect_buffer_type ||
           t == _motion_srt_type;
}

void CUDACodegenXIR::_emit_type_name(const Type *type) noexcept {
    if (type == nullptr) {
        _scratch << "void";
        return;
    }
    switch (type->tag()) {
        case Type::Tag::BOOL: _scratch << "lc_bool"; break;
        case Type::Tag::FLOAT16: _scratch << "lc_half"; break;
        case Type::Tag::FLOAT32: _scratch << "lc_float"; break;
        case Type::Tag::FLOAT64: _scratch << "lc_double"; break;
        case Type::Tag::INT8: _scratch << "lc_byte"; break;
        case Type::Tag::UINT8: _scratch << "lc_ubyte"; break;
        case Type::Tag::INT16: _scratch << "lc_short"; break;
        case Type::Tag::UINT16: _scratch << "lc_ushort"; break;
        case Type::Tag::INT32: _scratch << "lc_int"; break;
        case Type::Tag::UINT32: _scratch << "lc_uint"; break;
        case Type::Tag::INT64: _scratch << "lc_long"; break;
        case Type::Tag::UINT64: _scratch << "lc_ulong"; break;
        case Type::Tag::VECTOR: {
            _emit_type_name(type->element());
            _scratch << type->dimension();
            break;
        }
        case Type::Tag::MATRIX: {
            auto dim = type->dimension();
            _emit_type_name(type->element());
            _scratch << dim << "x" << dim;
            break;
        }
        case Type::Tag::ARRAY: {
            _scratch << "lc_array<";
            _emit_type_name(type->element());
            _scratch << ", ";
            _scratch << type->dimension() << ">";
            break;
        }
        case Type::Tag::STRUCTURE: {
            if (type == _ray_type) {
                _scratch << "LCRay";
            } else if (type == _triangle_hit_type) {
                _scratch << "LCTriangleHit";
            } else if (type == _procedural_hit_type) {
                _scratch << "LCProceduralHit";
            } else if (type == _committed_hit_type) {
                _scratch << "LCCommittedHit";
            } else if (type == _motion_srt_type) {
                _scratch << "LCMotionSRT";
            } else {
                _scratch << "S" << hash_to_string(type->hash());
            }
            break;
        }
        case Type::Tag::CUSTOM: {
            if (type == _ray_query_all_type) {
                _scratch << "LCRayQueryAll";
            } else if (type == _ray_query_any_type) {
                _scratch << "LCRayQueryAny";
            } else if (type == _indirect_buffer_type) {
                _scratch << "LCIndirectBuffer";
            } else {
                LUISA_ERROR_WITH_LOCATION(
                    "Unsupported custom type: {}.",
                    type->description());
            }
            break;
        }
        case Type::Tag::BUFFER: {
            if (type == _indirect_buffer_type) {
                _scratch << "LCIndirectBuffer";
            } else {
                _scratch << "LCBuffer<";
                if (auto elem = type->element()) {
                    _emit_type_name(elem);
                } else {
                    _scratch << "lc_ubyte";
                }
                _scratch << ">";
            }
            break;
        }
        case Type::Tag::TEXTURE: {
            auto elem = type->element();
            auto dim = type->dimension();
            _scratch << "LCTexture" << dim << "D<";
            _emit_type_name(elem);
            _scratch << ">";
            break;
        }
        case Type::Tag::BINDLESS_ARRAY: _scratch << "LCBindlessArray"; break;
        case Type::Tag::ACCEL: _scratch << "LCAccel"; break;
        default: LUISA_ERROR_WITH_LOCATION(
            "Invalid type {} in CUDA codegen.",
            type->description());
    }
}

void CUDACodegenXIR::_emit_type_definition(const Type *type, luisa::unordered_set<const Type *> &defined_types) noexcept {
    if (type == nullptr || !defined_types.emplace(type).second) { return; }
    switch (type->tag()) {
        case Type::Tag::STRUCTURE: {
            auto members = type->members();
            for (auto m : members) {
                _emit_type_definition(m, defined_types);
            }
            // generate the type definition if not a builtin type
            if (!_is_builtin_type(type)) {
                _scratch << "struct alignas(" << type->alignment() << ") ";
                _emit_type_name(type);
                _scratch << " {\n";
                for (auto i = 0u; i < members.size(); i++) {
                    _scratch << "  ";
                    _emit_type_name(members[i]);
                    _scratch << " m" << i << ";\n";
                }
                _scratch << "};\n\n";
            }
            break;
        }
        case Type::Tag::ARRAY: [[fallthrough]];
        case Type::Tag::BUFFER: {
            _emit_type_definition(type->element(), defined_types);
            break;
        }
        case Type::Tag::CUSTOM: {
            LUISA_ASSERT(type == _ray_query_all_type ||
                             type == _ray_query_any_type ||
                             type == _indirect_buffer_type,
                         "Unsupported custom type: {}.",
                         type->description());
            break;
        }
        default: break;
    }
}

void CUDACodegenXIR::_emit_type_definitions(luisa::unordered_set<const Type *> used_types) noexcept {
    luisa::vector<const Type *> types;
    types.reserve(used_types.size());
    for (auto t : used_types) {
        if (t != nullptr) { types.emplace_back(t); }
    }
    luisa::sort(types.begin(), types.end(), [](auto a, auto b) noexcept {
        return a->hash() < b->hash();
    });
    used_types.clear();
    for (auto t : types) {
        _emit_type_definition(t, used_types);
    }
}

void CUDACodegenXIR::_emit_kernel_params_struct(const xir::KernelFunction *kernel) noexcept {
    // declare the kernel argument struct
    _scratch << "struct alignas(16) Params {";
    {
        auto i = 0u;
        for (auto arg : kernel->arguments()) {
            LUISA_ASSERT(!arg->is_reference(), "Reference argument is not supported.");
            _scratch << "\n  alignas(16) ";
            _emit_type_name(arg->type());
            _scratch << " m" << i << ";";
            i++;
        }
    }
    if (_requires_printing) {
        _scratch << "\n  alignas(16) LCPrintBuffer print_buffer;";
    }
    _scratch << "\n  alignas(16) lc_uint4 ls_kid;";
    _scratch << "\n};\n\n";
    if (_requires_optix) {// optix requires __constant__ params
        _scratch << "extern \"C\" __constant__ Params params;\n\n";
    }
}

namespace {

template<typename T>
[[nodiscard]] auto cuda_codegen_xir_decode_literal(const void *data) noexcept {
    return *static_cast<const T *>(data);
}

void cuda_codegen_xir_emit_literal(StringScratch &s, const Type *type, const std::byte *data) noexcept {
    switch (type->tag()) {
        case Type::Tag::BOOL: s << (cuda_codegen_xir_decode_literal<bool>(data) ? "true" : "false"); break;
        case Type::Tag::INT8: s << luisa::format("lc_byte({})", static_cast<int>(cuda_codegen_xir_decode_literal<int8_t>(data))); break;
        case Type::Tag::UINT8: s << luisa::format("lc_ubyte({})", static_cast<uint>(cuda_codegen_xir_decode_literal<uint8_t>(data))); break;
        case Type::Tag::INT16: s << luisa::format("lc_short({})", cuda_codegen_xir_decode_literal<int16_t>(data)); break;
        case Type::Tag::UINT16: s << luisa::format("lc_ushort({})", cuda_codegen_xir_decode_literal<uint16_t>(data)); break;
        case Type::Tag::INT32: s << luisa::format("lc_int({})", cuda_codegen_xir_decode_literal<int32_t>(data)); break;
        case Type::Tag::UINT32: s << luisa::format("lc_uint({})", cuda_codegen_xir_decode_literal<uint32_t>(data)); break;
        case Type::Tag::INT64: s << luisa::format("lc_long({})", cuda_codegen_xir_decode_literal<int64_t>(data)); break;
        case Type::Tag::UINT64: s << luisa::format("lc_ulong({})", cuda_codegen_xir_decode_literal<uint64_t>(data)); break;
        case Type::Tag::FLOAT16: {
            auto v = cuda_codegen_xir_decode_literal<half>(data);
            LUISA_ASSERT(!luisa::isnan(v), "Encountered with NaN.");
            if (luisa::isinf(v)) {
                s << luisa::format("(lc_bit_cast<lc_half, lc_ushort>(0x{:04x}u))", luisa::bit_cast<ushort>(v));
            } else {
                s << luisa::format("lc_half({})", static_cast<float>(v));
            }
            break;
        }
        case Type::Tag::FLOAT32: {
            auto v = cuda_codegen_xir_decode_literal<float>(data);
            LUISA_ASSERT(!luisa::isnan(v), "Encountered with NaN.");
            if (luisa::isinf(v)) {
                s << luisa::format("(lc_bit_cast<lc_float, lc_uint>(0x{:08x}u))", luisa::bit_cast<uint>(v));
            } else {
                s << luisa::format("lc_float({})", v);
            }
            break;
        }
        case Type::Tag::FLOAT64: {
            auto v = cuda_codegen_xir_decode_literal<double>(data);
            LUISA_ASSERT(!luisa::isnan(v), "Encountered with NaN.");
            if (luisa::isinf(v)) {
                s << luisa::format("(lc_bit_cast<lc_double, lc_ulong>(0x{:016x}ull))", luisa::bit_cast<ulong>(v));
            } else {
                s << luisa::format("lc_double({})", v);
            }
            break;
        }
        case Type::Tag::VECTOR: {
            auto elem = type->element();
            auto elem_stride = elem->size();
            auto dim = type->dimension();
            s << "lc_make_" << elem->description() << dim << "(";
            for (auto i = 0u; i < dim; i++) {
                cuda_codegen_xir_emit_literal(s, elem, data + i * elem_stride);
                if (i != dim - 1u) { s << ", "; }
            }
            s << ")";
            break;
        }
        case Type::Tag::MATRIX: {
            auto elem = type->element();
            auto dim = type->dimension();
            auto col = Type::vector(elem, dim);
            auto col_stride = col->size();
            s << "lc_make_" << elem->description() << dim << "x" << dim << "(";
            for (auto i = 0u; i < dim; i++) {
                cuda_codegen_xir_emit_literal(s, col, data + i * col_stride);
                if (i != dim - 1u) { s << ", "; }
            }
            s << ")";
            break;
        }
        default: LUISA_NOT_IMPLEMENTED("Unsupported literal type {}.", type->description());
    }
}

}// namespace

void CUDACodegenXIR::_emit_value_name(const xir::Value *value, bool is_use) noexcept {
    auto get_index = [](const xir::Value *v, luisa::unordered_map<const xir::Value *, size_t> &indices) noexcept {
        return indices.try_emplace(v, indices.size()).first->second;
    };
    auto get_global_index = [&](const xir::Value *v) noexcept {
        return get_index(v, _global_value_indices);
    };
    auto get_local_index = [&](const xir::Value *v) noexcept {
        return get_index(v, _local_value_indices);
    };
    LUISA_DEBUG_ASSERT(value != nullptr, "Value is null.");
    switch (value->derived_value_tag()) {
        case xir::DerivedValueTag::UNDEFINED: {
            _scratch << "(lc_undef_value<";
            _emit_type_name(value->type());
            _scratch << ">())";
            break;
        }
        case xir::DerivedValueTag::FUNCTION: {
            switch (static_cast<const xir::Function *>(value)->derived_function_tag()) {
                case xir::DerivedFunctionTag::KERNEL: _scratch << "kernel_main"; break;
                case xir::DerivedFunctionTag::CALLABLE: _scratch << "callable_" << get_global_index(value); break;
                case xir::DerivedFunctionTag::EXTERNAL: LUISA_NOT_IMPLEMENTED(
                    "External function {} is not supported in XIR-based CUDA codegen.",
                    value->name().value_or("unknown"));
            }
            break;
        }
        case xir::DerivedValueTag::BASIC_BLOCK: {
            _scratch << "bb" << get_local_index(value);
            break;
        }
        case xir::DerivedValueTag::INSTRUCTION: {
            _scratch << (value->is_lvalue() ? "pv" : "v")
                     << get_local_index(value);
            break;
        }
        case xir::DerivedValueTag::CONSTANT: {
            if (auto c = static_cast<const xir::Constant *>(value); _should_emit_global_constant(c)) {
                if (is_use) {
                    _scratch << "(lc_decode_bytes<";
                    _emit_type_name(c->type());
                    _scratch << ">(";
                }
                _scratch << "constant_" << get_global_index(value);
                if (is_use) {
                    _scratch << "))";
                }
            } else {// literal
                cuda_codegen_xir_emit_literal(_scratch, c->type(), static_cast<const std::byte *>(c->data()));
            }
            break;
        }
        case xir::DerivedValueTag::ARGUMENT: {
            _scratch << (value->is_lvalue() ? "pa" : "a")
                     << get_local_index(value);
            break;
        }
        case xir::DerivedValueTag::SPECIAL_REGISTER: {
            switch (static_cast<const xir::SpecialRegister *>(value)->derived_special_register_tag()) {
                case xir::DerivedSpecialRegisterTag::THREAD_ID: _scratch << "sreg_tid"; break;
                case xir::DerivedSpecialRegisterTag::BLOCK_ID: _scratch << "sreg_bid"; break;
                case xir::DerivedSpecialRegisterTag::WARP_LANE_ID: _scratch << "sreg_lid"; break;
                case xir::DerivedSpecialRegisterTag::DISPATCH_ID: _scratch << "sreg_did"; break;
                case xir::DerivedSpecialRegisterTag::KERNEL_ID: _scratch << "sreg_kid"; break;
                case xir::DerivedSpecialRegisterTag::RASTER_BARYCENTRICS:
                case xir::DerivedSpecialRegisterTag::RASTER_OBJECT_ID: LUISA_NOT_IMPLEMENTED("Object ID is not supported in XIR-based CUDA codegen.");
                case xir::DerivedSpecialRegisterTag::BLOCK_SIZE: _scratch << "sreg_bs"; break;
                case xir::DerivedSpecialRegisterTag::WARP_SIZE: _scratch << "sreg_ws"; break;
                case xir::DerivedSpecialRegisterTag::DISPATCH_SIZE: _scratch << "sreg_ls"; break;
            }
            break;
        }
    }
}

void CUDACodegenXIR::_emit_global_constants(luisa::unordered_set<const xir::Constant *> used_constants) noexcept {
    luisa::vector<const xir::Constant *> constants;
    constants.reserve(used_constants.size());
    for (auto &&c : used_constants) {
        if (_should_emit_global_constant(c)) {
            constants.emplace_back(c);
        }
    }
    std::sort(constants.begin(), constants.end(), [](auto a, auto b) noexcept {
        return a->hash() < b->hash();
    });
    for (auto c : constants) {
        _emit_metadata(c->metadata_list(), 0);
        _scratch << "__constant__ alignas(16) LC_CONSTANT lc_ubyte ";
        _emit_value_name(c, false);
        auto n = c->type()->size();
        _scratch << "[" << n << "] = /* ";
        _emit_type_name(c->type());
        _scratch << " */ {";
        auto bytes = static_cast<const std::uint8_t *>(c->data());
        for (auto i = 0u; i < n; i++) {
            if (i % 16u == 0u) { _scratch << "\n   "; }
            _scratch << luisa::format(" 0x{:02x},", static_cast<uint32_t>(bytes[i]));
        }
        _scratch << "\n};\n\n";
    }
}

void CUDACodegenXIR::_emit_result_value_eq(const xir::Instruction *inst) noexcept {
    if (auto ret_type = inst->type()) {
        // we may define the result value locally if not breaking the lexical scope,
        // otherwise we are referencing the hoisted result value so do not define it again
        if (_lex_scope_info.lexical_scope_breakers.contains(inst)) {
            _scratch << "/* hoisted lexical scope breaker */ ";
        } else {
            _emit_type_name(ret_type);
            if (inst->is_lvalue()) {
                _scratch << " *";
            } else {
                _scratch << " ";
            }
        }
        _emit_value_name(inst);
        _scratch << " = ";
    }
}

void CUDACodegenXIR::_emit_ray_query_pipeline_inst(const xir::RayQueryPipelineInst *inst, int indent) noexcept {
    // retrieve info
    auto &&info = _ray_query_pipeline_info.at(inst);
    // prepare capture context if needed
    if (info.any_context_capture) {
        _scratch << "/* encode ray query context */\n";
        _emit_indent(indent);
        _scratch << "LCRayQueryCtx" << info.index << " ";
        _emit_value_name(inst);
        _scratch << "_ctx = {};\n";
        for (auto i = 0u; i < info.args.size(); i++) {
            if (info.args[i].tag == RayQueryPipelineArgument::Tag::CONTEXT_CAPTURE) {
                _emit_indent(indent + 1);
                _emit_value_name(inst);
                _scratch << "_ctx.m" << info.args[i].mapped_index << " = ";
                if (info.args[i].is_pointer) { _scratch << "*"; }
                _scratch << "(";
                _emit_value_name(inst->captured_argument(i));
                _scratch << ");\n";
            }
        }
        _emit_indent(indent);
    }
    // invoke the ray query pipeline
    _scratch << "lc_ray_query_trace(*(";
    _emit_value_name(inst->query_object());
    _scratch << "), " << info.index << ", ";
    if (info.any_context_capture) {
        _scratch << "&";
        _emit_value_name(inst);
        _scratch << "_ctx";
    } else {
        _scratch << "nullptr";
    }
    _scratch << ");";
    // copy out captured pointer values
    for (auto i = 0u; i < info.args.size(); i++) {
        if (info.args[i].is_pointer) {
            _scratch << "\n";
            _emit_indent(indent + 1);
            _scratch << "*(";
            _emit_value_name(inst->captured_argument(i));
            _scratch << ") = ";
            _emit_value_name(inst);
            _scratch << "_ctx.m" << info.args[i].mapped_index << "; /* copy out ray query captured pointer */";
        }
    }
}

[[nodiscard]] const xir::Instruction *CUDACodegenXIR::_find_innermost_loop() const noexcept {
    for (auto it = _control_flow_stack.rbegin(); it != _control_flow_stack.rend(); ++it) {
        if ((*it)->isa<xir::LoopInst>() || (*it)->isa<xir::SimpleLoopInst>()) {
            return *it;
        }
    }
    return nullptr;
}

void CUDACodegenXIR::_emit_instructions(const xir::InstructionList &inst_list, int indent) noexcept {
    for (auto inst : inst_list) {
        _emit_metadata(inst->metadata_list(), indent);
        _emit_indent(indent);
        auto emit_result_value_eq = [&] { _emit_result_value_eq(inst); };
        switch (inst->derived_instruction_tag()) {
            case xir::DerivedInstructionTag::IF: {
                _with_control_flow(inst, [&] {
                    _emit_if_inst(static_cast<const xir::IfInst *>(inst), indent);
                });
                break;
            }
            case xir::DerivedInstructionTag::SWITCH: {
                _with_control_flow(inst, [&] {
                    _emit_switch_inst(static_cast<const xir::SwitchInst *>(inst), indent);
                });
                break;
            }
            case xir::DerivedInstructionTag::INDEXED_BRANCH:
                LUISA_ERROR_WITH_LOCATION(
                    "CUDA XIR codegen requires structured control flow; "
                    "run restructure_cfg before codegen.");
            case xir::DerivedInstructionTag::LOOP: {
                _with_control_flow(inst, [&] {
                    _emit_loop_inst(static_cast<const xir::LoopInst *>(inst), indent);
                });
                break;
            }
            case xir::DerivedInstructionTag::SIMPLE_LOOP: {
                _with_control_flow(inst, [&] {
                    _emit_simple_loop_inst(static_cast<const xir::SimpleLoopInst *>(inst), indent);
                });
                break;
            }
            case xir::DerivedInstructionTag::BRANCH: _emit_branch_inst(static_cast<const xir::BranchInst *>(inst)); break;
            case xir::DerivedInstructionTag::CONDITIONAL_BRANCH: _emit_conditional_branch_inst(static_cast<const xir::ConditionalBranchInst *>(inst)); break;
            case xir::DerivedInstructionTag::UNREACHABLE: {
                if (auto &&msg = static_cast<const xir::UnreachableInst *>(inst)->message(); !msg.empty()) {
                    _scratch << "lc_unreachable_with_message(__FILE__, __LINE__, "
                             << luisa::format("{:?}", msg)
                             << ");";
                } else {
                    _scratch << "lc_unreachable(__FILE__, __LINE__);";
                }
                break;
            }
            case xir::DerivedInstructionTag::BREAK: {
                auto loop = _find_innermost_loop();
                LUISA_ASSERT(loop != nullptr, "Break instruction is not in a loop.");
                if (loop->isa<xir::LoopInst>()) {
                    _scratch << "/* break inside generic loop */ { loop_break_";
                    _emit_value_name(loop);
                    _scratch << " = true; break; }";
                } else {
                    _scratch << "break;";
                }
                break;
            }
            case xir::DerivedInstructionTag::CONTINUE: {
                auto loop = _find_innermost_loop();
                LUISA_ASSERT(loop != nullptr, "Continue instruction is not in a loop.");
                if (loop->isa<xir::LoopInst>()) {
                    _scratch << "/* continue inside generic loop */ break;";
                } else {
                    _scratch << "continue;";
                }
                break;
            }
            case xir::DerivedInstructionTag::RETURN: {
                if (auto ret = static_cast<const xir::ReturnInst *>(inst)->return_value()) {
                    _scratch << "return ";
                    _emit_value_name(ret);
                    _scratch << ";";
                } else {
                    _scratch << "return;";
                }
                break;
            }
            case xir::DerivedInstructionTag::RASTER_DISCARD: LUISA_NOT_IMPLEMENTED("Raster discard is not supported in XIR-based CUDA codegen.");
            case xir::DerivedInstructionTag::PHI: LUISA_ERROR_WITH_LOCATION("Phi instructions should be eliminated before codegen.");
            case xir::DerivedInstructionTag::ALLOCA: {
                // emit the alloca variable
                if (static_cast<const xir::AllocaInst *>(inst)->is_shared()) {
                    _scratch << "__shared__ ";
                }
                _emit_type_name(inst->type());
                _scratch << " ";
                _emit_value_name(inst);
                _scratch << "_alloca;";
                // emit the pointer to the alloca variable
                _scratch << "\n";
                _emit_indent(indent);
                emit_result_value_eq();
                _scratch << "&";
                _emit_value_name(inst);
                _scratch << "_alloca;";
                break;
            }
            case xir::DerivedInstructionTag::LOAD: {
                emit_result_value_eq();
                _scratch << "*(";
                _emit_value_name(static_cast<const xir::LoadInst *>(inst)->variable());
                _scratch << ");";
                break;
            }
            case xir::DerivedInstructionTag::STORE: {
                _scratch << "*(";
                _emit_value_name(static_cast<const xir::StoreInst *>(inst)->variable());
                _scratch << ") = ";
                _emit_value_name(static_cast<const xir::StoreInst *>(inst)->value());
                _scratch << ";";
                break;
            }
            case xir::DerivedInstructionTag::GEP: {
                emit_result_value_eq();
                auto gep = static_cast<const xir::GEPInst *>(inst);
                _scratch << "&((*(";
                _emit_value_name(gep->base());
                _scratch << "))";
                _emit_access_chain(gep->base()->type(), gep->index_uses());
                _scratch << ");";
                break;
            }
            case xir::DerivedInstructionTag::ATOMIC: _emit_atomic_inst(static_cast<const xir::AtomicInst *>(inst)); break;
            case xir::DerivedInstructionTag::ARITHMETIC: _emit_arithmetic_inst(static_cast<const xir::ArithmeticInst *>(inst), indent); break;
            case xir::DerivedInstructionTag::THREAD_GROUP: _emit_thread_group_inst(static_cast<const xir::ThreadGroupInst *>(inst)); break;
            case xir::DerivedInstructionTag::RESOURCE_QUERY: _emit_resource_query_inst(static_cast<const xir::ResourceQueryInst *>(inst)); break;
            case xir::DerivedInstructionTag::RESOURCE_READ: _emit_resource_read_inst(static_cast<const xir::ResourceReadInst *>(inst)); break;
            case xir::DerivedInstructionTag::RESOURCE_WRITE: _emit_resource_write_inst(static_cast<const xir::ResourceWriteInst *>(inst)); break;
            case xir::DerivedInstructionTag::RAY_QUERY_LOOP: LUISA_ERROR_WITH_LOCATION("Ray query loop instructions should be eliminated before codegen.");
            case xir::DerivedInstructionTag::RAY_QUERY_DISPATCH: LUISA_ERROR_WITH_LOCATION("Ray query dispatch instructions should be eliminated before codegen.");
            case xir::DerivedInstructionTag::RAY_QUERY_OBJECT_READ: _emit_ray_query_object_read_inst(static_cast<const xir::RayQueryObjectReadInst *>(inst)); break;
            case xir::DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE: _emit_ray_query_object_write_inst(static_cast<const xir::RayQueryObjectWriteInst *>(inst)); break;
            case xir::DerivedInstructionTag::RAY_QUERY_PIPELINE: _emit_ray_query_pipeline_inst(static_cast<const xir::RayQueryPipelineInst *>(inst), indent); break;
            case xir::DerivedInstructionTag::AUTODIFF_SCOPE: LUISA_ERROR_WITH_LOCATION("Autodiff scope instructions should be eliminated before codegen.");
            case xir::DerivedInstructionTag::AUTODIFF_INTRINSIC: LUISA_ERROR_WITH_LOCATION("Autodiff intrinsic instructions should be eliminated before codegen.");
            case xir::DerivedInstructionTag::CALL: {
                // Runtime tensor operators (plan.md §1.4) have no XIR
                // representation: the AST->XIR translator rejects every
                // `TENSOR_*` CallOp loudly (see ast2xir.cpp), so they never
                // reach this codegen. They are implemented directly by the
                // AST codegen path (cuda_codegen_ast.cpp) as `lc_tensor_*`
                // device builtins. If an external/unknown function call ever
                // surfaces here, `_emit_value_name` fails loudly on
                // `DerivedFunctionTag::EXTERNAL` instead of miscompiling.
                emit_result_value_eq();
                auto call = static_cast<const xir::CallInst *>(inst);
                _emit_value_name(call->callee());
                _scratch << "(";
                auto any_arg = false;
                for (auto arg_use : call->argument_uses()) {
                    any_arg = true;
                    _emit_value_name(arg_use->value());
                    _scratch << ", ";
                }
                if (!_requires_optix && _requires_printing) {
                    any_arg = true;
                    _scratch << "print_buffer, ";
                }
                if (any_arg) {
                    _scratch.pop_back();
                    _scratch.pop_back();
                }
                _scratch << ");";
                break;
            }
            case xir::DerivedInstructionTag::CAST: {
                emit_result_value_eq();
                auto cast = static_cast<const xir::CastInst *>(inst);
                switch (cast->op()) {
                    case xir::CastOp::STATIC_CAST: _scratch << "static_cast<"; break;
                    case xir::CastOp::BITWISE_CAST: _scratch << "lc_bit_cast<"; break;
                }
                _emit_type_name(cast->type());
                _scratch << ">(";
                _emit_value_name(cast->value());
                _scratch << ");";
                break;
            }
            case xir::DerivedInstructionTag::PRINT: {
                auto p = static_cast<const xir::PrintInst *>(inst);
                auto info = _print_info.at(p);
                _scratch << "lc_print_impl(LC_PRINT_BUFFER, ";
                _emit_type_name(info.type);
                _scratch << "{" << info.type->size() << ", "
                         << info.index;
                for (auto op_use : p->operand_uses()) {
                    _scratch << ", ";
                    _emit_value_name(op_use->value());
                }
                _scratch << "});";
                break;
            }
            case xir::DerivedInstructionTag::DEBUG_BREAK: {
                _scratch << "lc_debug_break();";
                break;
            }
            case xir::DerivedInstructionTag::CLOCK: {
                emit_result_value_eq();
                _scratch << "clock64();";
                break;
            }
            case xir::DerivedInstructionTag::ASSERT: {
                if (auto a = static_cast<const xir::AssertInst *>(inst); a->message().empty()) {
                    _scratch << "lc_assert(";
                    _emit_value_name(a->condition());
                    _scratch << ");";
                } else {
                    _scratch << "lc_assert_with_message(";
                    _emit_value_name(a->condition());
                    _scratch << ", "
                             << luisa::format("{:?}", a->message())
                             << ");";
                }
                break;
            }
            case xir::DerivedInstructionTag::ASSUME: {
                _scratch << "lc_assume(";
                _emit_value_name(static_cast<const xir::AssumeInst *>(inst)->condition());
                _scratch << ");";
                break;
            }
            case xir::DerivedInstructionTag::OUTLINE: LUISA_ERROR_WITH_LOCATION("Outline instructions should be eliminated before codegen.");
        }
        _scratch << "\n";
        if (auto merge = inst->control_flow_merge(); merge != nullptr && merge->merge_block() != nullptr) {
            _emit_instructions(merge->merge_block()->instructions(), indent);
        }
    }
}

void CUDACodegenXIR::_emit_metadata(const xir::MetadataList &md_list, int indent) const noexcept {
    for (auto md : md_list) {
        _emit_indent(indent);
        _scratch << "// ";
        switch (md->derived_metadata_tag()) {
            case xir::DerivedMetadataTag::NAME: {
                auto name_md = static_cast<const xir::NameMD *>(md);
                _scratch << luisa::format("name: {:?}",
                                          name_md->name());
                break;
            }
            case xir::DerivedMetadataTag::LOCATION: {
                auto loc_md = static_cast<const xir::LocationMD *>(md);
                _scratch << luisa::format("location: {:?}, line {}",
                                          loc_md->file().string(), loc_md->line());
                break;
            }
            case xir::DerivedMetadataTag::COMMENT: {
                auto comment_md = static_cast<const xir::CommentMD *>(md);
                for (auto c : comment_md->comment()) {
                    if (c == '\n') {
                        _scratch << "\n";
                        _emit_indent(indent);
                        _scratch << "// ";
                    } else if (isprint(c)) {
                        char s[] = {c, '\0'};
                        _scratch << s;
                    }
                }
                break;
            }
            case xir::DerivedMetadataTag::CURVE_BASIS: {
                auto bases = static_cast<const xir::CurveBasisMD *>(md)->curve_basis_set();
                _scratch << "curve basis:";
                if (bases.none()) {
                    _scratch << " none";
                } else {
                    if (bases.test(CurveBasis::PIECEWISE_LINEAR)) { _scratch << " piecewise_linear"; }
                    if (bases.test(CurveBasis::CUBIC_BSPLINE)) { _scratch << " cubic_bspline"; }
                    if (bases.test(CurveBasis::CATMULL_ROM)) { _scratch << " catmull_rom"; }
                    if (bases.test(CurveBasis::BEZIER)) { _scratch << " bezier"; }
                }
                break;
            }
            case xir::DerivedMetadataTag::SIGNATURE_CONSTRAINT:
                _scratch << "signature constraint";
                break;
            case xir::DerivedMetadataTag::REG2MEM_SPILL: {
                auto spill = static_cast<const xir::Reg2MemSpillMD *>(md);
                _scratch << "reg2mem spill: " << xir::to_string(spill->kind());
                break;
            }
        }
        _scratch << "\n";
    }
}

void CUDACodegenXIR::_emit_indent(int indent) const noexcept {
    _scratch.string().append(indent * 2, ' ');
}

void CUDACodegenXIR::_emit_access_chain(const Type *base_type, luisa::span<const xir::Use *const> chain) noexcept {
    for (auto t = base_type; !chain.empty(); chain = chain.subspan(1)) {
        auto index = chain.front()->value();
        switch (t->tag()) {
            case Type::Tag::VECTOR: [[fallthrough]];
            case Type::Tag::ARRAY: {
                _scratch << "[";
                _emit_value_name(index);
                _scratch << "]";
                t = t->element();
                break;
            }
            case Type::Tag::MATRIX: {
                _scratch << "[";
                _emit_value_name(index);
                _scratch << "]";
                t = Type::vector(t->element(), t->dimension());
                break;
            }
            case Type::Tag::STRUCTURE: {
                LUISA_ASSERT(index->isa<xir::Constant>(), "Structure index must be a constant.");
                auto i = [c = static_cast<const xir::Constant *>(index)]() noexcept -> size_t {
                    switch (c->type()->tag()) {
                        case Type::Tag::INT8: return c->as<int8_t>();
                        case Type::Tag::UINT8: return c->as<uint8_t>();
                        case Type::Tag::INT16: return c->as<int16_t>();
                        case Type::Tag::UINT16: return c->as<uint16_t>();
                        case Type::Tag::INT32: return c->as<int32_t>();
                        case Type::Tag::UINT32: return c->as<uint32_t>();
                        case Type::Tag::INT64: return c->as<int64_t>();
                        case Type::Tag::UINT64: return c->as<uint64_t>();
                        default: break;
                    }
                    LUISA_ERROR_WITH_LOCATION(
                        "Invalid structure index type {}.",
                        c->type()->description());
                }();
                LUISA_ASSERT(i < t->members().size(), "Structure index out of range.");
                _scratch << ".m" << i;
                t = t->members()[i];
                break;
            }
            default: LUISA_ERROR_WITH_LOCATION("Invalid access chain type {}.", t->description());
        }
    }
}

void CUDACodegenXIR::_emit_if_inst(const xir::IfInst *inst, int indent) noexcept {
    _scratch << "if (";
    _emit_value_name(inst->condition());
    _scratch << ") {";
    if (auto true_block = inst->true_block(); true_block != nullptr && !true_block->instructions().empty()) {
        _scratch << "\n";
        _emit_instructions(true_block->instructions(), indent + 1);
        _emit_indent(indent);
    }
    _scratch << "} else {";
    if (auto false_block = inst->false_block(); false_block != nullptr && !false_block->instructions().empty()) {
        _scratch << "\n";
        _emit_instructions(false_block->instructions(), indent + 1);
        _emit_indent(indent);
    }
    _scratch << "}";
}

void CUDACodegenXIR::_emit_switch_inst(const xir::SwitchInst *inst, int indent) noexcept {
    _scratch << "switch (";
    _emit_value_name(inst->value());
    _scratch << ") {";
    auto value_type = inst->value()->type();
    for (auto i = 0u; i < inst->case_count(); i++) {
        _scratch << "\n";
        _emit_indent(indent + 1);
        _scratch << "case static_cast<";
        _emit_type_name(value_type);
        _scratch << ">(";
        auto value = inst->case_value(i);
        switch (value_type->tag()) {
            case Type::Tag::INT8:
                _scratch << luisa::format(
                    "{}", static_cast<int64_t>(luisa::bit_cast<int8_t>(
                              static_cast<uint8_t>(value))));
                break;
            case Type::Tag::INT16:
                _scratch << luisa::format(
                    "{}", static_cast<int64_t>(luisa::bit_cast<int16_t>(
                              static_cast<uint16_t>(value))));
                break;
            case Type::Tag::INT32:
                _scratch << luisa::format(
                    "{}", static_cast<int64_t>(luisa::bit_cast<int32_t>(
                              static_cast<uint32_t>(value))));
                break;
            case Type::Tag::INT64:
                if (value == uint64_t{0x8000000000000000ull}) {
                    _scratch << "(-9223372036854775807ll - 1ll)";
                } else {
                    _scratch << luisa::format(
                        "{}ll", luisa::bit_cast<int64_t>(value));
                }
                break;
            default: _scratch << value << "ull"; break;
        }
        _scratch << "): {\n";
        if (auto block = inst->case_block(i); block != nullptr) {
            _emit_instructions(block->instructions(), indent + 2);
        }
        _emit_indent(indent + 1);
        _scratch << "}";
    }
    if (auto default_block = inst->default_block(); default_block != nullptr) {
        _scratch << "\n";
        _emit_indent(indent + 1);
        _scratch << "default: {\n";
        _emit_instructions(default_block->instructions(), indent + 2);
        _emit_indent(indent + 1);
        _scratch << "}\n";
    }
    _scratch << "}";
}

void CUDACodegenXIR::_emit_loop_inst(const xir::LoopInst *inst, int indent) noexcept {
    // template:
    // for (;;) {
    //   /* prepare */
    //   if (br(merge)) { break; }
    //   /* body */
    //   bool loop_break = false;
    //   do {
    //     continue -> { break; }
    //     break -> { loop_break = true; break; }
    //   } while (false);
    //   if (loop_break) { break; }
    //   /* update */
    // }
    _scratch << "for (;;) { /* generic loop */\n";
    _emit_indent(indent + 1);
    _scratch << "/* generic loop prepare */\n";
    // prepare
    if (auto prepare_block = inst->prepare_block(); prepare_block != nullptr && !prepare_block->instructions().empty()) {
        _emit_instructions(prepare_block->instructions(), indent + 1);
    }
    _emit_indent(indent + 1);
    // body
    _scratch << "/* generic loop body */\n";
    _emit_indent(indent + 1);
    _scratch << "bool loop_break_";
    _emit_value_name(inst);
    _scratch << " = false;\n";
    _emit_indent(indent + 1);
    _scratch << "do {\n";
    if (auto body_block = inst->body_block(); body_block != nullptr && !body_block->instructions().empty()) {
        _emit_instructions(body_block->instructions(), indent + 2);
    }
    _emit_indent(indent + 1);
    _scratch << "} while (false);\n";
    _emit_indent(indent + 1);
    _scratch << "if (loop_break_";
    _emit_value_name(inst);
    _scratch << ") { break; }\n";
    // update
    if (auto update_block = inst->update_block(); update_block != nullptr && !update_block->instructions().empty()) {
        _emit_indent(indent + 1);
        _scratch << "/* generic loop update */\n";
        _emit_instructions(update_block->instructions(), indent + 1);
    }
    _emit_indent(indent);
    _scratch << "}";
}

void CUDACodegenXIR::_emit_simple_loop_inst(const xir::SimpleLoopInst *inst, int indent) noexcept {
    // do-while loop
    _scratch << "for (;;) { /* simple loop */";
    if (auto body_block = inst->body_block(); body_block != nullptr && !body_block->instructions().empty()) {
        _scratch << "\n";
        _emit_instructions(body_block->instructions(), indent + 1);
        _emit_indent(indent);
    }
    _scratch << "}";
}

void CUDACodegenXIR::_emit_operand_list(luisa::span<const xir::Use *const> operands) noexcept {
    auto any_arg = false;
    for (auto &&arg_use : operands) {
        any_arg = true;
        _emit_value_name(arg_use->value());
        _scratch << ", ";
    }
    if (any_arg) {
        _scratch.pop_back();
        _scratch.pop_back();
    }
}

void CUDACodegenXIR::_emit_intrinsic_call(luisa::string_view name, const xir::Instruction *inst) noexcept {
    _emit_result_value_eq(inst);
    _scratch << name << "(";
    _emit_operand_list(inst->operand_uses());
    _scratch << ");";
}

int CUDACodegenXIR::_find_ray_query_captured_kernel_param_index(const xir::Value *capture) const noexcept {
    // try to find out if the captured argument is a kernel parameter
    // if so, return the index of the kernel parameter
    // otherwise, return -1

    if (!capture->isa<xir::Argument>() || capture->isa<xir::ReferenceArgument>()) {
        // if captured value is not an argument or is a reference argument, skip
        return -1;
    }
    // we can safely cast to argument and find its index in the parent function
    auto argument = static_cast<const xir::Argument *>(capture);
    auto parent = argument->parent_function();
    auto arg_index = [argument, parent] {
        auto i = 0u;
        for (auto parent_arg : parent->arguments()) {
            if (parent_arg == argument) { return i; }
            i++;
        }
        LUISA_ERROR_WITH_LOCATION("Cannot find argument index for captured value.");
    }();
    // if the argument is a kernel parameter, we can return its index
    if (parent->isa<xir::KernelFunction>()) { return arg_index; }
    // must be a callable function otherwise
    LUISA_ASSERT(parent->isa<xir::CallableFunction>(), "Parent function is not a kernel or callable function.");
    // find all uses of the callable and check if they all point to the same kernel parameter
    auto kernel_param_index = -1;
    for (auto &&use : parent->use_list()) {
        if (auto user = use->user(); user != nullptr && user->isa<xir::CallInst>()) {
            if (auto in_arg = _find_ray_query_captured_kernel_param_index(
                    static_cast<const xir::CallInst *>(user)->argument(arg_index));
                in_arg != -1) {
                if (kernel_param_index == -1) { kernel_param_index = in_arg; }
                if (kernel_param_index != in_arg) { return -1; }
            } else {
                return -1;
            }
        } else {
            return -1;
        }
    }
    return kernel_param_index;
}

void CUDACodegenXIR::_preprocess_ray_query_pipelines(luisa::span<const xir::RayQueryPipelineInst *const> pipelines) noexcept {
    for (auto p : pipelines) {
        auto [iter, success] = _ray_query_pipeline_info.emplace(
            p, RayQueryPipelineInfo{
                   .index = static_cast<uint>(_ray_query_pipeline_info.size()),
                   .any_context_capture = false});
        LUISA_ASSERT(success, "Ray query pipeline {} already exists.", p->name().value_or("unknown"));
        // sort the captured arguments
        struct ContextCapture {
            const Type *type;
            int arg_index;
            int type_align;
        };
        // collect argument info
        luisa::vector<ContextCapture> context;
        {
            auto captures = p->captured_argument_uses();
            iter->second.args.reserve(captures.size());
            context.reserve(captures.size());
            for (auto i = 0; i < captures.size(); i++) {
                auto capture = captures[i]->value();
                LUISA_ASSERT(capture != nullptr, "Captured argument is null.");
                if (auto k = _find_ray_query_captured_kernel_param_index(capture); k != -1) {
                    iter->second.args.emplace_back(RayQueryPipelineArgument{
                        .tag = RayQueryPipelineArgument::Tag::KERNEL_PARAM,
                        .is_pointer = false,
                        .mapped_index = k,
                    });
                } else {
                    iter->second.any_context_capture = true;
                    iter->second.args.emplace_back(RayQueryPipelineArgument{
                        .tag = RayQueryPipelineArgument::Tag::CONTEXT_CAPTURE,
                        .is_pointer = capture->is_lvalue(),
                        .mapped_index = 0,// to fill
                    });
                    // collect the non-kernel argument indices
                    auto t = capture->type();
                    LUISA_ASSERT(t != nullptr, "Captured argument type is null.");
                    auto type_align = t->is_resource() || t->is_custom() ? 16u : t->alignment();
                    context.emplace_back(ContextCapture{
                        .type = t,
                        .arg_index = i,
                        .type_align = static_cast<short>(type_align),
                    });
                }
            }
            // sort the non-kernel argument indices according to their size
            std::stable_sort(context.begin(), context.end(), [](auto lhs, auto rhs) noexcept {
                return rhs.type_align < lhs.type_align;
            });
            for (auto i = 0u; i < context.size(); i++) {
                auto arg = context[i].arg_index;
                iter->second.args[arg].mapped_index = static_cast<int>(i);
            }
        }
        // emit context struct
        _scratch << "struct LCRayQueryCtx" << iter->second.index << "{";
        if (!context.empty()) {
            for (auto i = 0u; i < context.size(); i++) {
                auto t = context[i].type;
                _scratch << "\n  ";
                _emit_type_name(t);
                _scratch << " m" << i << ";";
            }
            _scratch << "\n";
        }
        _scratch << "};\n\n";
    }
}

void CUDACodegenXIR::_postprocess_ray_query_pipelines(luisa::span<const xir::RayQueryPipelineInst *const> pipelines,
                                                      luisa::span<const Function::Binding> bindings) noexcept {
    for (auto p : pipelines) {
        // retrieve the pipeline info
        auto &&info = _ray_query_pipeline_info.at(p);
        using namespace std::string_view_literals;
        for (auto [f, signature] : {
                 std::make_pair(p->on_surface_function(), "LUISA_DECL_RAY_QUERY_TRIANGLE_IMPL"sv),
                 std::make_pair(p->on_procedural_function(), "LUISA_DECL_RAY_QUERY_PROCEDURAL_IMPL"sv),
             }) {
            _scratch << signature << "(" << info.index << ") {\n"
                     << "  LCIntersectionResult result{};\n";
            if (f != nullptr) {
                if (info.any_context_capture) {
                    _scratch << "  /* decode ray query context */\n"
                             << "  auto ctx = static_cast<LCRayQueryCtx" << info.index << " *>(ctx_in);\n";
                }
                // generate compiler hints
                for (auto i = 0u; i < bindings.size(); i++) {
                    if (auto binding = luisa::get_if<Function::TextureBinding>(&bindings[i]);
                        binding != nullptr && info.args[i].tag == RayQueryPipelineArgument::Tag::KERNEL_PARAM) {
                        auto surface = reinterpret_cast<CUDATexture *>(binding->handle)->binding(binding->level);
                        // generate hints for the underlying storage
                        _scratch << "  lc_assume(params.m" << i << ".surface.storage == " << surface.storage << ");\n";
                    }
                }
                // invoke the ray query callback
                _scratch << "  /* invoke ray query pipeline */\n"
                         << "  ";
                _emit_value_name(f);
                _scratch << "(result";
                for (auto arg : info.args) {
                    _scratch << ", ";
                    if (arg.is_pointer) { _scratch << "&"; }
                    switch (arg.tag) {
                        case RayQueryPipelineArgument::Tag::CONTEXT_CAPTURE: _scratch << "(ctx->m"; break;
                        case RayQueryPipelineArgument::Tag::KERNEL_PARAM: _scratch << "(params.m"; break;
                    }
                    _scratch << arg.mapped_index << ")";
                }
                _scratch << ");\n";
            }
            _scratch << "  return result;\n"
                     << "}\n\n";
        }
    }
}

void CUDACodegenXIR::_emit_atomic_inst(const xir::AtomicInst *inst) noexcept {
    LUISA_ASSERT(inst->operand_count() >= 1u /* base */ + inst->value_count(),
                 "Atomic instruction {} has {} operands, but at least {} is expected.",
                 xir::to_string(inst->op()), inst->operand_count(), 1u + inst->value_count());
    _emit_result_value_eq(inst);
    switch (inst->op()) {
        case xir::AtomicOp::EXCHANGE: _scratch << "lc_atomic_exchange"; break;
        case xir::AtomicOp::COMPARE_EXCHANGE: _scratch << "lc_atomic_compare_exchange"; break;
        case xir::AtomicOp::FETCH_ADD: _scratch << "lc_atomic_fetch_add"; break;
        case xir::AtomicOp::FETCH_SUB: _scratch << "lc_atomic_fetch_sub"; break;
        case xir::AtomicOp::FETCH_AND: _scratch << "lc_atomic_fetch_and"; break;
        case xir::AtomicOp::FETCH_OR: _scratch << "lc_atomic_fetch_or"; break;
        case xir::AtomicOp::FETCH_XOR: _scratch << "lc_atomic_fetch_xor"; break;
        case xir::AtomicOp::FETCH_MIN: _scratch << "lc_atomic_fetch_min"; break;
        case xir::AtomicOp::FETCH_MAX: _scratch << "lc_atomic_fetch_max"; break;
    }
    _scratch << "(";
    auto base = inst->base();
    auto base_type = base->type();
    auto indices = inst->index_uses();
    if (base_type->is_buffer()) {// decode the buffer type
        LUISA_ASSERT(!indices.empty(), "Atomic instruction {} has no indices.", xir::to_string(inst->op()));
        _scratch << "((";
        _emit_value_name(base);
        _scratch << ").ptr[";
        _emit_value_name(indices[0]->value());
        _scratch << "])";
        base_type = base_type->element();
        indices = indices.subspan(1);
    } else {
        _scratch << "(*(";
        _emit_value_name(base);
        _scratch << "))";
    }
    _emit_access_chain(base_type, indices);
    auto values = inst->value_uses();
    for (auto value_use : values) {
        _scratch << ", ";
        _emit_value_name(value_use->value());
    }
    _scratch << ");";
}

void CUDACodegenXIR::_emit_arithmetic_inst(const xir::ArithmeticInst *inst, int indent) noexcept {
    auto u = [&](auto op) noexcept { _emit_with_template(inst, op, "(", 0, ")"); };
    auto b = [&](auto op) noexcept { _emit_with_template(inst, "(", 0, ") ", op, " (", 1, ")"); };
    auto f = [&](auto s) noexcept { _emit_intrinsic_call(s, inst); };
    switch (inst->op()) {
        case xir::ArithmeticOp::UNARY_MINUS: u("-"); break;
        case xir::ArithmeticOp::UNARY_BIT_NOT: {
            if (auto t = inst->type(); t->is_bool() || t->is_bool_vector()) {
                u("!");
            } else {
                u("~");
            }
            break;
        }
        case xir::ArithmeticOp::BINARY_ADD: b("+"); break;
        case xir::ArithmeticOp::BINARY_SUB: b("-"); break;
        case xir::ArithmeticOp::BINARY_MUL: b("*"); break;
        case xir::ArithmeticOp::BINARY_DIV: b("/"); break;
        case xir::ArithmeticOp::BINARY_MOD:
            if (inst->type()->is_float_or_float_vector()) {
                f("lc_fmod");
            } else {
                b("%");
            }
            break;
        case xir::ArithmeticOp::BINARY_BIT_AND: {
            if (auto t = inst->type(); t->is_bool() || t->is_bool_vector()) {
                b("&&");
            } else {
                b("&");
            }
            break;
        }
        case xir::ArithmeticOp::BINARY_BIT_OR: {
            if (auto t = inst->type(); t->is_bool() || t->is_bool_vector()) {
                b("||");
            } else {
                b("|");
            }
            break;
        }
        case xir::ArithmeticOp::BINARY_BIT_XOR: b("^"); break;
        case xir::ArithmeticOp::BINARY_SHIFT_LEFT: b("<<"); break;
        case xir::ArithmeticOp::BINARY_SHIFT_RIGHT: b(">>"); break;
        case xir::ArithmeticOp::BINARY_ROTATE_LEFT: LUISA_NOT_IMPLEMENTED("lc_rotl"); break;
        case xir::ArithmeticOp::BINARY_ROTATE_RIGHT: LUISA_NOT_IMPLEMENTED("lc_rotr"); break;
        case xir::ArithmeticOp::BINARY_LESS: b("<"); break;
        case xir::ArithmeticOp::BINARY_GREATER: b(">"); break;
        case xir::ArithmeticOp::BINARY_LESS_EQUAL: b("<="); break;
        case xir::ArithmeticOp::BINARY_GREATER_EQUAL: b(">="); break;
        case xir::ArithmeticOp::BINARY_EQUAL: b("=="); break;
        case xir::ArithmeticOp::BINARY_NOT_EQUAL: b("!="); break;
        case xir::ArithmeticOp::ALL: f("lc_all"); break;
        case xir::ArithmeticOp::ANY: f("lc_any"); break;
        case xir::ArithmeticOp::SELECT: f("lc_select"); break;
        case xir::ArithmeticOp::CLAMP: f("lc_clamp"); break;
        case xir::ArithmeticOp::SATURATE: f("lc_saturate"); break;
        case xir::ArithmeticOp::LERP: f("lc_lerp"); break;
        case xir::ArithmeticOp::STEP: f("lc_step"); break;
        case xir::ArithmeticOp::SMOOTHSTEP: f("lc_smoothstep"); break;
        case xir::ArithmeticOp::ABS: f("lc_abs"); break;
        case xir::ArithmeticOp::MIN: f("lc_min"); break;
        case xir::ArithmeticOp::MAX: f("lc_max"); break;
        case xir::ArithmeticOp::CLZ: f("lc_clz"); break;
        case xir::ArithmeticOp::CTZ: f("lc_ctz"); break;
        case xir::ArithmeticOp::POPCOUNT: f("lc_popcount"); break;
        case xir::ArithmeticOp::REVERSE: f("lc_reverse"); break;
        case xir::ArithmeticOp::ISINF: f("lc_isinf"); break;
        case xir::ArithmeticOp::ISNAN: f("lc_isnan"); break;
        case xir::ArithmeticOp::ACOS: f("lc_acos"); break;
        case xir::ArithmeticOp::ACOSH: f("lc_acosh"); break;
        case xir::ArithmeticOp::ASIN: f("lc_asin"); break;
        case xir::ArithmeticOp::ASINH: f("lc_asinh"); break;
        case xir::ArithmeticOp::ATAN: f("lc_atan"); break;
        case xir::ArithmeticOp::ATAN2: f("lc_atan2"); break;
        case xir::ArithmeticOp::ATANH: f("lc_atanh"); break;
        case xir::ArithmeticOp::COS: f("lc_cos"); break;
        case xir::ArithmeticOp::COSH: f("lc_cosh"); break;
        case xir::ArithmeticOp::SIN: f("lc_sin"); break;
        case xir::ArithmeticOp::SINH: f("lc_sinh"); break;
        case xir::ArithmeticOp::TAN: f("lc_tan"); break;
        case xir::ArithmeticOp::TANH: f("lc_tanh"); break;
        case xir::ArithmeticOp::EXP: f("lc_exp"); break;
        case xir::ArithmeticOp::EXP2: f("lc_exp2"); break;
        case xir::ArithmeticOp::EXP10: f("lc_exp10"); break;
        case xir::ArithmeticOp::LOG: f("lc_log"); break;
        case xir::ArithmeticOp::LOG2: f("lc_log2"); break;
        case xir::ArithmeticOp::LOG10: f("lc_log10"); break;
        case xir::ArithmeticOp::POW: f("lc_pow"); break;
        case xir::ArithmeticOp::POW_INT: f("lc_powi"); break;
        case xir::ArithmeticOp::SQRT: f("lc_sqrt"); break;
        case xir::ArithmeticOp::RSQRT: f("lc_rsqrt"); break;
        case xir::ArithmeticOp::CEIL: f("lc_ceil"); break;
        case xir::ArithmeticOp::FLOOR: f("lc_floor"); break;
        case xir::ArithmeticOp::FRACT: f("lc_fract"); break;
        case xir::ArithmeticOp::TRUNC: f("lc_trunc"); break;
        case xir::ArithmeticOp::ROUND: f("lc_round"); break;
        case xir::ArithmeticOp::RINT: f("lc_rint"); break;
        case xir::ArithmeticOp::FMA: f("lc_fma"); break;
        case xir::ArithmeticOp::COPYSIGN: f("lc_copysign"); break;
        case xir::ArithmeticOp::CROSS: f("lc_cross"); break;
        case xir::ArithmeticOp::DOT: f("lc_dot"); break;
        case xir::ArithmeticOp::LENGTH: f("lc_length"); break;
        case xir::ArithmeticOp::LENGTH_SQUARED: f("lc_length_squared"); break;
        case xir::ArithmeticOp::NORMALIZE: f("lc_normalize"); break;
        case xir::ArithmeticOp::FACEFORWARD: f("lc_faceforward"); break;
        case xir::ArithmeticOp::REFLECT: f("lc_reflect"); break;
        case xir::ArithmeticOp::REDUCE_SUM: f("lc_reduce_sum"); break;
        case xir::ArithmeticOp::REDUCE_PRODUCT: f("lc_reduce_prod"); break;
        case xir::ArithmeticOp::REDUCE_MIN: f("lc_reduce_min"); break;
        case xir::ArithmeticOp::REDUCE_MAX: f("lc_reduce_max"); break;
        case xir::ArithmeticOp::OUTER_PRODUCT: f("lc_outer_product"); break;
        case xir::ArithmeticOp::MATRIX_COMP_NEG: u("-"); break;
        case xir::ArithmeticOp::MATRIX_COMP_ADD: b("+"); break;
        case xir::ArithmeticOp::MATRIX_COMP_SUB: b("-"); break;
        case xir::ArithmeticOp::MATRIX_COMP_MUL: {
            if (inst->operand(0)->type()->is_scalar() || inst->operand(1)->type()->is_scalar()) {
                b("*");
            } else {
                f("lc_mat_comp_mul");
            }
            break;
        }
        case xir::ArithmeticOp::MATRIX_COMP_DIV: {
            if (inst->operand(0)->type()->is_scalar() || inst->operand(1)->type()->is_scalar()) {
                b("/");
            } else {
                _emit_with_template(inst, "lc_mat_comp_mul(", 0, ", 1.f / ", 1, ")");
            }
            break;
        }
        case xir::ArithmeticOp::MATRIX_LINALG_MUL: b("*"); break;
        case xir::ArithmeticOp::MATRIX_DETERMINANT: f("lc_determinant"); break;
        case xir::ArithmeticOp::MATRIX_TRANSPOSE: f("lc_transpose"); break;
        case xir::ArithmeticOp::MATRIX_INVERSE: f("lc_inverse"); break;
        case xir::ArithmeticOp::AGGREGATE: {
            _emit_result_value_eq(inst);
            switch (auto t = inst->type(); t->tag()) {
                case Type::Tag::VECTOR: {
                    _scratch << "lc_make_" << t->element()->description()
                             << t->dimension() << "(";
                    _emit_operand_list(inst->operand_uses());
                    _scratch << "); /* aggregate */";
                    break;
                }
                case Type::Tag::MATRIX: {
                    _scratch << "lc_make_" << t->element()->description()
                             << t->dimension() << "x" << t->dimension() << "(";
                    _emit_operand_list(inst->operand_uses());
                    _scratch << "); /* aggregate */";
                    break;
                }
                case Type::Tag::ARRAY: [[fallthrough]];
                case Type::Tag::STRUCTURE: {
                    _emit_type_name(t);
                    _scratch << "{";
                    _emit_operand_list(inst->operand_uses());
                    _scratch << "}; /* aggregate */";
                    break;
                }
                default: LUISA_ERROR_WITH_LOCATION("Invalid aggregate type {}.", t->description());
            }
            break;
        }
        case xir::ArithmeticOp::SHUFFLE: {
            switch (auto t = inst->type(); t->tag()) {
                case Type::Tag::VECTOR: [[fallthrough]];
                case Type::Tag::MATRIX: [[fallthrough]];
                case Type::Tag::ARRAY: {
                    LUISA_ASSERT(inst->operand_count() == 1u /* the source aggregate */ + t->dimension(),
                                 "Invalid shuffle instruction.");
                    _emit_result_value_eq(inst);
                    _emit_type_name(t);
                    _scratch << "{";
                    auto v = inst->operand(0);
                    for (auto op : inst->operand_uses().subspan(1)) {
                        _scratch << "(";
                        _emit_value_name(v);
                        _scratch << ")[";
                        _emit_value_name(op->value());
                        _scratch << "], ";
                    }
                    _scratch.pop_back();
                    _scratch.pop_back();
                    _scratch << "}; /* shuffle */";
                    break;
                }
                default: LUISA_ERROR_WITH_LOCATION("Invalid shuffle type {}.", t->description());
            }
            break;
        }
        case xir::ArithmeticOp::INSERT: {
            // T result = v;
            // result.access_chain = e;
            LUISA_DEBUG_ASSERT(inst->operand_count() > 2u, "Insert instruction should have at least 3 operands.");
            _emit_result_value_eq(inst);
            auto v = inst->operand(0);
            _emit_value_name(v);
            _scratch << ";\n";
            _emit_indent(indent);
            _emit_value_name(inst);
            _emit_access_chain(inst->type(), inst->operand_uses().subspan(2));
            _scratch << " = ";
            auto e = inst->operand(1);
            _emit_value_name(e);
            _scratch << "; /* insert */";
            break;
        }
        case xir::ArithmeticOp::EXTRACT: {
            _emit_result_value_eq(inst);
            _scratch << "(";
            auto v = inst->operand(0);
            _emit_value_name(v);
            _scratch << ")";
            _emit_access_chain(v->type(), inst->operand_uses().subspan(1));
            _scratch << "; /* extract */";
            break;
        }
    }
}

void CUDACodegenXIR::_emit_thread_group_inst(const xir::ThreadGroupInst *inst) noexcept {
    auto f = [&](auto s) noexcept { _emit_intrinsic_call(s, inst); };
    switch (inst->op()) {
        case xir::ThreadGroupOp::SHADER_EXECUTION_REORDER: f("lc_shader_execution_reorder"); break;
        case xir::ThreadGroupOp::RASTER_QUAD_DDX: LUISA_NOT_IMPLEMENTED("lc_raster_quad_ddx"); break;
        case xir::ThreadGroupOp::RASTER_QUAD_DDY: LUISA_NOT_IMPLEMENTED("lc_raster_quad_ddy"); break;
        case xir::ThreadGroupOp::WARP_IS_FIRST_ACTIVE_LANE: f("lc_warp_is_first_active_lane"); break;
        case xir::ThreadGroupOp::WARP_FIRST_ACTIVE_LANE: f("lc_warp_first_active_lane"); break;
        case xir::ThreadGroupOp::WARP_ACTIVE_ALL_EQUAL: f("lc_warp_active_all_equal"); break;
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_AND: f("lc_warp_active_bit_and"); break;
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_OR: f("lc_warp_active_bit_or"); break;
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_XOR: f("lc_warp_active_bit_xor"); break;
        case xir::ThreadGroupOp::WARP_ACTIVE_COUNT_BITS: f("lc_warp_active_count_bits"); break;
        case xir::ThreadGroupOp::WARP_ACTIVE_MAX: f("lc_warp_active_max"); break;
        case xir::ThreadGroupOp::WARP_ACTIVE_MIN: f("lc_warp_active_min"); break;
        case xir::ThreadGroupOp::WARP_ACTIVE_PRODUCT: f("lc_warp_active_product"); break;
        case xir::ThreadGroupOp::WARP_ACTIVE_SUM: f("lc_warp_active_sum"); break;
        case xir::ThreadGroupOp::WARP_ACTIVE_ALL: f("lc_warp_active_all"); break;
        case xir::ThreadGroupOp::WARP_ACTIVE_ANY: f("lc_warp_active_any"); break;
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_MASK: f("lc_warp_active_bit_mask"); break;
        case xir::ThreadGroupOp::WARP_PREFIX_COUNT_BITS: f("lc_warp_prefix_count_bits"); break;
        case xir::ThreadGroupOp::WARP_PREFIX_SUM: f("lc_warp_prefix_sum"); break;
        case xir::ThreadGroupOp::WARP_PREFIX_PRODUCT: f("lc_warp_prefix_product"); break;
        case xir::ThreadGroupOp::WARP_READ_LANE: f("lc_warp_read_lane"); break;
        case xir::ThreadGroupOp::WARP_READ_FIRST_ACTIVE_LANE: f("lc_warp_read_first_active_lane"); break;
        case xir::ThreadGroupOp::SYNCHRONIZE_BLOCK: f("lc_synchronize_block"); break;
    }
}

void CUDACodegenXIR::_emit_resource_query_inst(const xir::ResourceQueryInst *inst) noexcept {
    _emit_result_value_eq(inst);
    switch (inst->op()) {
        case xir::ResourceQueryOp::BUFFER_SIZE: _scratch << "lc_buffer_size"; break;
        case xir::ResourceQueryOp::BYTE_BUFFER_SIZE: _scratch << "lc_byte_buffer_size"; break;
        case xir::ResourceQueryOp::TEXTURE2D_SIZE: _scratch << "lc_texture_size"; break;
        case xir::ResourceQueryOp::TEXTURE3D_SIZE: _scratch << "lc_texture_size"; break;
        case xir::ResourceQueryOp::BINDLESS_BUFFER_SIZE: {
            LUISA_ASSERT(inst->operand_count() == 3u, "Bindless buffer size instruction should have 3 operands.");
            _scratch << "([](const LCBindlessArray b, size_t i, size_t s) noexcept { return lc_bindless_buffer_size<lc_ubyte>(b, i) / s; })";
            break;
        }
        case xir::ResourceQueryOp::BINDLESS_BYTE_BUFFER_SIZE: _scratch << "lc_bindless_buffer_size"; break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE: _scratch << "lc_bindless_texture_size2d"; break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE: _scratch << "lc_bindless_texture_size3d"; break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE_LEVEL: _scratch << "lc_bindless_texture_size2d_level"; break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE_LEVEL: _scratch << "lc_bindless_texture_size3d_level"; break;
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE: LUISA_NOT_IMPLEMENTED("TEXTURE2D_SAMPLE");
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE_LEVEL: LUISA_NOT_IMPLEMENTED("TEXTURE2D_SAMPLE_LEVEL");
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD: LUISA_NOT_IMPLEMENTED("TEXTURE2D_SAMPLE_GRAD");
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD_LEVEL: LUISA_NOT_IMPLEMENTED("TEXTURE2D_SAMPLE_GRAD_LEVEL");
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE: LUISA_NOT_IMPLEMENTED("TEXTURE3D_SAMPLE");
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE_LEVEL: LUISA_NOT_IMPLEMENTED("TEXTURE3D_SAMPLE_LEVEL");
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD: LUISA_NOT_IMPLEMENTED("TEXTURE3D_SAMPLE_GRAD");
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD_LEVEL: LUISA_NOT_IMPLEMENTED("TEXTURE3D_SAMPLE_GRAD_LEVEL");
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE: _scratch << "lc_bindless_texture_sample2d"; break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL: _scratch << "lc_bindless_texture_sample2d_level"; break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD: _scratch << "lc_bindless_texture_sample2d_grad"; break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL: LUISA_NOT_IMPLEMENTED("BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL");
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE: _scratch << "lc_bindless_texture_sample3d"; break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL: _scratch << "lc_bindless_texture_sample3d_level"; break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD: _scratch << "lc_bindless_texture_sample3d_grad"; break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL: LUISA_NOT_IMPLEMENTED("BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL");
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER: LUISA_NOT_IMPLEMENTED("BINDLESS_TEXTURE2D_SAMPLE_SAMPLER"); break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER: LUISA_NOT_IMPLEMENTED("BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER"); break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER: LUISA_NOT_IMPLEMENTED("BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER"); break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER: LUISA_NOT_IMPLEMENTED("BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER"); break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER: LUISA_NOT_IMPLEMENTED("BINDLESS_TEXTURE3D_SAMPLE_SAMPLER"); break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER: LUISA_NOT_IMPLEMENTED("BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER"); break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER: LUISA_NOT_IMPLEMENTED("BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER"); break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER: LUISA_NOT_IMPLEMENTED("BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER"); break;
        case xir::ResourceQueryOp::BUFFER_DEVICE_ADDRESS: _scratch << "lc_buffer_address"; break;
        case xir::ResourceQueryOp::BINDLESS_BUFFER_DEVICE_ADDRESS: _scratch << "lc_bindless_buffer_address"; break;
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_TRANSFORM: _scratch << "lc_accel_instance_transform"; break;
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_USER_ID: _scratch << "lc_accel_instance_user_id"; break;
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_VISIBILITY_MASK: _scratch << "lc_accel_instance_visibility_mask"; break;
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST: _scratch << "lc_accel_trace_closest"; break;
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY: _scratch << "lc_accel_trace_any"; break;
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL: _scratch << "lc_accel_query_all"; break;
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY: _scratch << "lc_accel_query_any"; break;
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_MATRIX: _scratch << "lc_accel_instance_motion_matrix"; break;
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_SRT: _scratch << "lc_accel_instance_motion_srt"; break;
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR: _scratch << "lc_accel_trace_closest_motion_blur"; break;
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR: _scratch << "lc_accel_trace_any_motion_blur"; break;
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR: _scratch << "lc_accel_query_all_motion_blur"; break;
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR: _scratch << "lc_accel_query_any_motion_blur"; break;
    }
    _scratch << "(";
    _emit_operand_list(inst->operand_uses());
    _scratch << ");";
}

void CUDACodegenXIR::_emit_resource_read_inst(const xir::ResourceReadInst *inst) noexcept {
    _emit_result_value_eq(inst);
    switch (inst->op()) {
        case xir::ResourceReadOp::BUFFER_READ: _scratch << "lc_buffer_read"; break;
        case xir::ResourceReadOp::BUFFER_VOLATILE_READ: _scratch << "lc_buffer_volatile_read"; break;
        case xir::ResourceReadOp::BYTE_BUFFER_READ: {
            _scratch << "lc_byte_buffer_read<";
            _emit_type_name(inst->type());
            _scratch << ">";
            break;
        }
        case xir::ResourceReadOp::BYTE_BUFFER_VOLATILE_READ: {
            _scratch << "lc_byte_buffer_read<";
            _emit_type_name(inst->type());
            _scratch << ">";
            break;
        }
        case xir::ResourceReadOp::TEXTURE2D_READ: _scratch << "lc_texture_read"; break;
        case xir::ResourceReadOp::TEXTURE3D_READ: _scratch << "lc_texture_read"; break;
        case xir::ResourceReadOp::BINDLESS_BUFFER_READ: {
            _scratch << "lc_bindless_buffer_read<";
            _emit_type_name(inst->type());
            _scratch << ">";
            break;
        }
        case xir::ResourceReadOp::BINDLESS_BYTE_BUFFER_READ: {
            _scratch << "lc_bindless_byte_buffer_read<";
            _emit_type_name(inst->type());
            _scratch << ">";
            break;
        }
        case xir::ResourceReadOp::BINDLESS_TEXTURE2D_READ: _scratch << "lc_bindless_texture_read2d"; break;
        case xir::ResourceReadOp::BINDLESS_TEXTURE3D_READ: _scratch << "lc_bindless_texture_read3d"; break;
        case xir::ResourceReadOp::BINDLESS_TEXTURE2D_READ_LEVEL: _scratch << "lc_bindless_texture_read2d_level"; break;
        case xir::ResourceReadOp::BINDLESS_TEXTURE3D_READ_LEVEL: _scratch << "lc_bindless_texture_read3d_level"; break;
        case xir::ResourceReadOp::DEVICE_ADDRESS_READ: {
            _scratch << "*reinterpret_cast<const ";
            _emit_type_name(inst->type());
            _scratch << " *>";
            break;
        }
        case xir::ResourceReadOp::COOPERATIVE_MUL_ADD:
        case xir::ResourceReadOp::BINDLESS_COOPERATIVE_MUL_ADD:
        case xir::ResourceReadOp::COOPERATIVE_MUL:
        case xir::ResourceReadOp::BINDLESS_COOPERATIVE_MUL:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_LOAD:
        case xir::ResourceReadOp::BINDLESS_COOPERATIVE_VECTOR_LOAD:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_SPLAT:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_CAST:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_WORKGROUP_LOAD:
        // Future cooperative-vector element-wise operations — TODO: implement
        // in the CUDA XIR backend.
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_DOT:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_ABS:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_SIGN:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_FLOOR:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_CEIL:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_FRACT:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_TRUNC:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_ROUND:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_RINT:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_SQRT:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_RSQRT:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_EXP2:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_EXP10:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_LOG2:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_LOG10:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_SATURATE:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_ISINF:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_ISNAN:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_SIN:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_COS:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_TAN:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_ASIN:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_ACOS:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_SINH:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_COSH:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_ASINH:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_ACOSH:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_ATANH:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_MIX:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_LERP:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_POW:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_STEP:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_SMOOTHSTEP:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_ADD:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_SUB:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_MUL:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_DIV:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_LESS:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_LESS_EQUAL:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_GREATER:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_GREATER_EQUAL:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_EQUAL:
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_NOT_EQUAL:
            LUISA_NOT_IMPLEMENTED("Cooperative-vector element-wise operations are not implemented in the CUDA XIR backend yet.");
    }
    _scratch << "(";
    _emit_operand_list(inst->operand_uses());
    _scratch << ");";
}

void CUDACodegenXIR::_emit_resource_write_inst(const xir::ResourceWriteInst *inst) noexcept {
    _emit_result_value_eq(inst);
    switch (inst->op()) {
        case xir::ResourceWriteOp::BUFFER_WRITE: _scratch << "lc_buffer_write"; break;
        case xir::ResourceWriteOp::BUFFER_VOLATILE_WRITE: _scratch << "lc_buffer_volatile_write"; break;
        case xir::ResourceWriteOp::BYTE_BUFFER_WRITE: _scratch << "lc_byte_buffer_write"; break;
        case xir::ResourceWriteOp::BYTE_BUFFER_VOLATILE_WRITE: _scratch << "lc_byte_buffer_volatile_write"; break;
        case xir::ResourceWriteOp::TEXTURE2D_WRITE: _scratch << "lc_texture_write"; break;
        case xir::ResourceWriteOp::TEXTURE3D_WRITE: _scratch << "lc_texture_write"; break;
        case xir::ResourceWriteOp::BINDLESS_BUFFER_WRITE: _scratch << "lc_bindless_buffer_write"; break;
        case xir::ResourceWriteOp::BINDLESS_BYTE_BUFFER_WRITE: _scratch << "lc_bindless_byte_buffer_write"; break;
        case xir::ResourceWriteOp::DEVICE_ADDRESS_WRITE: _scratch << "([](lc_ulong p, auto v) noexcept { *reinterpret_cast<decltype(v) *>(p) = v; })"; break;
        case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_TRANSFORM: _scratch << "lc_accel_set_instance_transform"; break;
        case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_VISIBILITY_MASK: _scratch << "lc_accel_set_instance_visibility"; break;
        case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_OPACITY: _scratch << "lc_accel_set_instance_opacity"; break;
        case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_USER_ID: _scratch << "lc_accel_set_instance_user_id"; break;
        case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_MOTION_MATRIX: _scratch << "lc_accel_set_instance_motion_matrix"; break;
        case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_MOTION_SRT: _scratch << "lc_accel_set_instance_motion_srt"; break;
        case xir::ResourceWriteOp::INDIRECT_DISPATCH_SET_KERNEL: _scratch << "lc_indirect_set_dispatch_kernel"; break;
        case xir::ResourceWriteOp::INDIRECT_DISPATCH_SET_COUNT: _scratch << "lc_indirect_set_dispatch_count"; break;
        case xir::ResourceWriteOp::COOPERATIVE_OUTER_PRODUCT_ACCUMULATE:
        case xir::ResourceWriteOp::COOPERATIVE_VECTOR_ACCUMULATE:
        case xir::ResourceWriteOp::COOPERATIVE_VECTOR_STORE:
        case xir::ResourceWriteOp::BINDLESS_COOPERATIVE_VECTOR_STORE:
        case xir::ResourceWriteOp::COOPERATIVE_VECTOR_WORKGROUP_STORE:
            LUISA_NOT_IMPLEMENTED();
    }
    _scratch << "(";
    _emit_operand_list(inst->operand_uses());
    _scratch << ");";
}

void CUDACodegenXIR::_emit_ray_query_object_read_inst(const xir::RayQueryObjectReadInst *inst) noexcept {
    switch (inst->op()) {
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_WORLD_SPACE_RAY:
            _emit_intrinsic_call("LC_RAY_QUERY_WORLD_RAY", inst);
            break;
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_CANDIDATE_OBJECT_SPACE_RAY:
            _emit_intrinsic_call("LC_RAY_QUERY_OBJECT_RAY", inst);
            break;
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_PROCEDURAL_CANDIDATE_HIT:
            _emit_intrinsic_call("LC_RAY_QUERY_PROCEDURAL_CANDIDATE_HIT", inst);
            break;
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_TRIANGLE_CANDIDATE_HIT:
            _emit_intrinsic_call("LC_RAY_QUERY_TRIANGLE_CANDIDATE_HIT", inst);
            break;
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_COMMITTED_HIT:
            _emit_with_template(inst, "lc_ray_query_committed_hit(*(", 0, "))");
            break;
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TRIANGLE_CANDIDATE:
            LUISA_NOT_IMPLEMENTED("LC_RAY_QUERY_IS_TRIANGLE_CANDIDATE");
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_PROCEDURAL_CANDIDATE:
            LUISA_NOT_IMPLEMENTED("LC_RAY_QUERY_IS_PROCEDURAL_CANDIDATE");
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED:
            LUISA_NOT_IMPLEMENTED("LC_RAY_QUERY_IS_TERMINATED");
    }
}

void CUDACodegenXIR::_emit_ray_query_object_write_inst(const xir::RayQueryObjectWriteInst *inst) noexcept {
    switch (inst->op()) {
        case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_TRIANGLE:
            _emit_intrinsic_call("LC_RAY_QUERY_COMMIT_TRIANGLE", inst);
            break;
        case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_PROCEDURAL:
            _emit_intrinsic_call("LC_RAY_QUERY_COMMIT_PROCEDURAL", inst);
            break;
        case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_TERMINATE:
            _emit_intrinsic_call("LC_RAY_QUERY_TERMINATE", inst);
            break;
        case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED:
            LUISA_NOT_IMPLEMENTED("LC_RAY_QUERY_PROCEED");
    }
}

void CUDACodegenXIR::_emit_branch_inst(const xir::BranchInst *inst) const noexcept {
    LUISA_DEBUG_ASSERT(!_control_flow_stack.empty(), "Control flow stack is empty.");
    switch (auto control_flow = _control_flow_stack.back(); control_flow->derived_instruction_tag()) {
        case xir::DerivedInstructionTag::IF: {
            [[maybe_unused]] auto if_inst = static_cast<const xir::IfInst *>(control_flow);
            LUISA_ASSERT(inst->target_block() == if_inst->merge_block(),
                         "Branch target block is not the merge block of the if instruction.");
            _scratch << "/* (eliminated) if unconditional branch */";
            break;
        }
        case xir::DerivedInstructionTag::SWITCH: {
            [[maybe_unused]] auto switch_inst = static_cast<const xir::SwitchInst *>(control_flow);
            LUISA_ASSERT(inst->target_block() == switch_inst->merge_block(),
                         "Branch target block is not the merge block of the switch instruction.");
            _scratch << "break; /* switch unconditional branch */";
            break;
        }
        case xir::DerivedInstructionTag::SIMPLE_LOOP: {
            auto loop = static_cast<const xir::SimpleLoopInst *>(control_flow);
            if (inst->target_block() == loop->merge_block()) {
                _scratch << "break; /* simple loop unconditional branch */";
            } else {
                _scratch << "/* (eliminated) simple loop unconditional branch */";
            }
            break;
        }
        case xir::DerivedInstructionTag::LOOP: {
            auto loop = static_cast<const xir::LoopInst *>(control_flow);
            if (inst->target_block() == loop->merge_block()) {
                _scratch << "break; /* generic loop unconditional branch */";
            } else {
                _scratch << "/* (eliminated) generic loop unconditional branch */";
            }
            break;
        }
        default: LUISA_ERROR_WITH_LOCATION("Control flow stack is not a loop instruction.");
    }
}

void CUDACodegenXIR::_emit_conditional_branch_inst(const xir::ConditionalBranchInst *inst) noexcept {
    LUISA_DEBUG_ASSERT(!_control_flow_stack.empty(), "Control flow stack is empty.");
    switch (auto control_flow = _control_flow_stack.back(); control_flow->derived_instruction_tag()) {
        case xir::DerivedInstructionTag::LOOP: [[fallthrough]];
        case xir::DerivedInstructionTag::SIMPLE_LOOP: {
            if (auto merge_block = control_flow->control_flow_merge()->merge_block()) {
                if (inst->true_block() == merge_block) {// break if true
                    _scratch << "if (";
                    _emit_value_name(inst->condition());
                    _scratch << ") /* loop conditional branch */ { break; }";
                } else if (inst->false_block() == merge_block) {// break if not true
                    _scratch << "if (!(";
                    _emit_value_name(inst->condition());
                    _scratch << ")) /* loop conditional branch */ { break; }";
                } else {
                    _scratch << "/* (eliminated) loop conditional branch */";
                }
            }
            break;
        }
        default: LUISA_ERROR_WITH_LOCATION("Control flow stack is not a loop instruction.");
    }
}

void CUDACodegenXIR::_emit_function_definition(const xir::FunctionDefinition *def,
                                               luisa::span<const Function::Binding> bindings) noexcept {
    _lex_scope_info = xir::lex_scope_analysis_pass_run_on_function(def, {.loop_body_is_nested = true});
    _local_value_indices.clear();
    switch (def->derived_function_tag()) {
        case xir::DerivedFunctionTag::KERNEL: {
            auto kernel = static_cast<const xir::KernelFunction *>(def);
            _emit_metadata(kernel->metadata_list(), 0);
            _emit_kernel_definition(kernel, bindings);
            break;
        }
        case xir::DerivedFunctionTag::CALLABLE: {
            auto callable = static_cast<const xir::CallableFunction *>(def);
            _emit_metadata(callable->metadata_list(), 0);
            if (_is_ray_query_callback_function(callable)) {
                _emit_ray_query_callback_definition(callable);
            } else {
                _emit_callable_definition(callable);
            }
            break;
        }
        default: LUISA_NOT_IMPLEMENTED(
            "Unsupported function definition {} in XIR-based CUDA codegen.",
            def->name().value_or("unknown"));
    }
    _lex_scope_info = {};
    _local_value_indices.clear();
}

void CUDACodegenXIR::_emit_kernel_definition(const xir::KernelFunction *kernel,
                                             luisa::span<const Function::Binding> bindings) noexcept {
    if (_requires_optix) {
        _scratch << "extern \"C\" __global__ void __raygen__main() {";
    } else {
        _scratch << "extern \"C\" __global__ void kernel_main(const Params params) {";
    }
    // decode the kernel arguments
    _scratch << "\n\n  /* kernel arguments */";
    {
        auto i = 0u;
        for (auto arg : kernel->arguments()) {
            _scratch << "\n  auto const ";
            _emit_value_name(arg);
            _scratch << " = params.m" << i << ";";
            i++;
        }
    }
    if (!_requires_optix && _requires_printing) {
        _scratch << "\n  auto const print_buffer = params.print_buffer;";
    }
    // compiler hints from bindings
    {
        auto i = 0u;
        for (auto arg : kernel->arguments()) {
            if (i >= bindings.size()) { break; }
            if (auto binding = luisa::get_if<Function::TextureBinding>(&bindings[i])) {
                auto surface = reinterpret_cast<CUDATexture *>(binding->handle)->binding(binding->level);
                // generate hints for the underlying storage
                _scratch << "\n  lc_assume(";
                _emit_value_name(arg);
                _scratch << ".surface.storage == " << surface.storage << ");";
            }
            i++;
        }
    }
    // emit built-in variables
    _scratch
        << "\n\n  /* built-in variables */"
        // block size
        << "\n  constexpr auto sreg_bs = lc_block_size();"
        // launch size
        << "\n  const auto sreg_ls = lc_dispatch_size();"
        // dispatch id
        << "\n  const auto sreg_did = lc_dispatch_id();"
        // thread id
        << "\n  const auto sreg_tid = lc_thread_id();"
        // block id
        << "\n  const auto sreg_bid = lc_block_id();"
        // kernel id
        << "\n  const auto sreg_kid = lc_kernel_id();"
        // warp size
        << "\n  const auto sreg_ws = lc_warp_size();"
        // warp lane id
        << "\n  const auto sreg_lid = lc_warp_lane_id();";
    // emit launch size check if not using OptiX (Optix handles this internally)
    if (!_requires_optix) {
        _scratch << "\n  if (lc_any(sreg_did >= sreg_ls)) { return; }";
    }
    _scratch << "\n";
    // emit lexical scope breakers due to the mismatch of SSA and C++ scopes
    _emit_hoisted_lexical_scope_breakers();
    // emit function body
    _scratch << "\n  /* function body */\n";
    _emit_instructions(kernel->body_block()->instructions(), 1);
    _scratch << "}\n\n";
}

void CUDACodegenXIR::_emit_hoisted_lexical_scope_breakers() noexcept {
    if (!_lex_scope_info.lexical_scope_breaks_ordered.empty()) {
        _scratch << "\n  /* hoisted lexical scope breakers */\n";
        for (auto breaker : _lex_scope_info.lexical_scope_breaks_ordered) {
            _emit_indent(1);
            _emit_type_name(breaker->type());
            _scratch << " ";
            _emit_value_name(breaker);
            _scratch << ";\n";
        }
    }
}

void CUDACodegenXIR::_emit_callable_definition(const xir::CallableFunction *callable) noexcept {
    // emit function signature
    _scratch << "extern \"C\" ";
    if (callable->arguments().count_size() >= 8u) {
        _scratch << "__forceinline__ ";// nvcc can be stupid with inlining
    }
    _scratch << "__device__ ";
    _emit_type_name(callable->type());
    _scratch << " ";
    _emit_value_name(callable);
    _scratch << "(";
    auto any_arg = false;
    for (auto arg : callable->arguments()) {
        any_arg = true;
        _scratch << "\n    ";
        _emit_type_name(arg->type());
        if (arg->is_reference()) {
            _scratch << " *const ";
        } else {
            _scratch << " const ";
        }
        _emit_value_name(arg);
        _scratch << ",";
    }
    if (!_requires_optix && _requires_printing) {
        any_arg = true;
        _scratch << "\n    LCPrintBuffer const print_buffer,";
    }
    if (any_arg) { _scratch.pop_back(); }
    _scratch << ") noexcept {\n";
    // emit lexical scope breakers due to the mismatch of SSA and C++ scopes
    _emit_hoisted_lexical_scope_breakers();
    // emit function body
    _scratch << "\n  /* function body */\n";
    _emit_instructions(callable->body_block()->instructions(), 1);
    _scratch << "}\n\n";
}

void CUDACodegenXIR::_emit_ray_query_callback_definition(const xir::CallableFunction *callable) noexcept {
    // emit function signature
    // note: the function signature should already have been validated by
    // `_is_ray_query_callback_function` so we do not need to check it again
    _scratch << "extern \"C\" __forceinline__ __device__ void ";
    _emit_value_name(callable);
    // the first argument should be replaced by the ray query result
    _scratch << "(LCIntersectionResult &result";
    // emit the rest of the arguments
    for (auto it = ++callable->arguments().begin(); it != callable->arguments().end(); ++it) {
        auto capture = *it;
        _scratch << ", ";
        _emit_type_name(capture->type());
        if (capture->is_reference()) {
            _scratch << " *const ";
        } else {
            _scratch << " const ";
        }
        _emit_value_name(capture);
    }
    _scratch << ") noexcept {\n";
    // emit lexical scope breakers due to the mismatch of SSA and C++ scopes
    _emit_hoisted_lexical_scope_breakers();
    // emit function body
    _scratch << "\n  /* function body */\n";
    _emit_instructions(callable->body_block()->instructions(), 1);
    _scratch << "}\n\n";
}

void CUDACodegenXIR::emit(const xir::Module *module,
                          luisa::span<const Function::Binding> bindings,
                          luisa::string_view device_lib,
                          luisa::string_view native_include) noexcept {

    // find the kernel function
    auto kernel = [module] {
        const xir::KernelFunction *kernel = nullptr;
        for (auto f : module->function_list()) {
            if (f->isa<xir::KernelFunction>()) {
                LUISA_ASSERT(kernel == nullptr,
                             "CUDA codegen: expected exactly one kernel function, "
                             "found {:?}.",
                             f->name().value_or("unknown"));
                kernel = static_cast<const xir::KernelFunction *>(f);
            }
        }
        LUISA_ASSERT(kernel != nullptr,
                     "CUDA codegen: kernel function not found in module.");
        return kernel;
    }();

    // analyze instruction usage to prepare for code generation
    auto analysis = [this, kernel] {
        InstructionUsageAnalysis analysis;
        analysis.used_types.reserve(Type::count());
        analysis.used_constants.reserve(64u);
        analysis.ray_query_pipelines.reserve(16u);
        analysis.used_functions_post_order.reserve(64u);
        luisa::unordered_set<const xir::Function *> visited;
        visited.reserve(64u);
        _analyze_instruction_usage(kernel, analysis, visited);
        LUISA_ASSERT(!analysis.used_functions_post_order.empty() &&
                         analysis.used_functions_post_order.back() == kernel,
                     "CUDA codegen: kernel function not found in post order traversal.");
        return analysis;
    }();

    // generate macro definitions and header
    if (_requires_optix) {
        _scratch << "#define LUISA_ENABLE_OPTIX\n";
        if (analysis.required_curve_bases.any()) {
            _scratch << "#define LUISA_ENABLE_OPTIX_CURVE\n";
        }
        if (analysis.requires_raytracing_closest) {
            _scratch << "#define LUISA_ENABLE_OPTIX_TRACE_CLOSEST\n";
        }
        if (analysis.requires_raytracing_any) {
            _scratch << "#define LUISA_ENABLE_OPTIX_TRACE_ANY\n";
        }
        if (analysis.requires_raytracing_query) {
            _scratch << "#define LUISA_ENABLE_OPTIX_RAY_QUERY\n";
        }
        if (auto n = analysis.ray_query_pipelines.size(); n != 0u) {
            _scratch << "#define LUISA_RAY_QUERY_IMPL_COUNT " << n << "\n";
        }
    }
    _scratch << "#define LC_BLOCK_SIZE lc_make_uint3("
             << kernel->block_size().x << ", "
             << kernel->block_size().y << ", "
             << kernel->block_size().z << ")\n"
             << "\n/* built-in device library begin */\n"
             << device_lib
             << "\n/* built-in device library end */\n\n";

    // emit type declarations
    _emit_type_definitions(std::move(analysis.used_types));

    // emit the kernel parameter struct
    _emit_kernel_params_struct(kernel);

    // emit global constants
    _emit_global_constants(std::move(analysis.used_constants));

    // emit native include if any
    if (!native_include.empty()) {
        _scratch << "\n/* native include begin */\n\n"
                 << native_include
                 << "\n/* native include end */\n\n";
    }

    // prepare for ray query pipelines
    _preprocess_ray_query_pipelines(analysis.ray_query_pipelines);

    // emit function definitions
    for (auto f : analysis.used_functions_post_order) {
        if (auto def = f->definition()) {
            _emit_function_definition(def, bindings);
        } else {
            LUISA_NOT_IMPLEMENTED("External function");
        }
    }

    // finalize the ray query pipelines
    _postprocess_ray_query_pipelines(analysis.ray_query_pipelines, bindings);

    // emit indirect dispatch kernel if allowed
    if (_allow_indirect_dispatch && !_requires_optix) {
        _scratch << "extern \"C\" __global__ void kernel_launcher(Params params, const LCIndirectBuffer indirect) {\n"
                 << "  auto i = blockIdx.x * blockDim.x + threadIdx.x;\n"
                 << "  auto n = min(indirect.header()->size, indirect.capacity - indirect.offset);\n"
                 << "  if (i < n) {\n"
                 << "    auto args = params;\n"
                 << "    auto d = indirect.dispatches()[i + indirect.offset];\n"
                 << "    args.ls_kid = d.dispatch_size_and_kernel_id;\n"
                 << "    auto block_size = lc_block_size();\n"
                 << "#ifdef LUISA_DEBUG\n"
                 << "    lc_assert(lc_all(block_size == d.block_size));\n"
                 << "#endif\n"
                 << "    auto dispatch_size = lc_make_uint3(d.dispatch_size_and_kernel_id);\n"
                 << "    if (lc_all(dispatch_size > 0u)) {\n"
                 << "      auto block_count = (dispatch_size + block_size - 1u) / block_size;\n"
                 << "      auto nb = dim3(block_count.x, block_count.y, block_count.z);\n"
                 << "      auto bs = dim3(block_size.x, block_size.y, block_size.z);\n"
                 << "      kernel_main<<<nb, bs>>>(args);\n"
                 << "    }\n"
                 << "  }\n"
                 << "}\n\n";
    }
}

}// namespace luisa::compute::cuda

#endif
