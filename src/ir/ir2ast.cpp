#include "core/logging.h"
#include <ir/ir2ast.h>

namespace luisa::compute {

    void IR2AST::_convert_block(const ir::BasicBlock *block) noexcept {
        auto node_ref = block->first;
        while (node_ref != ir::INVALID_REF) {
            auto node = ir::luisa_compute_ir_node_get(node_ref);
            switch (node->instruction->tag) {
                case ir::Instruction::Tag::Local: _convert_instr_local(node); break;
                case ir::Instruction::Tag::UserData: _convert_instr_user_data(node); break;
                case ir::Instruction::Tag::Invalid: _convert_instr_invalid(node); break;
                case ir::Instruction::Tag::Const: _convert_instr_const(node); break;
                case ir::Instruction::Tag::Update: _convert_instr_update(node); break;
                case ir::Instruction::Tag::Call: {
                    auto expr = _convert_node(node);
                    break;
                }
                case ir::Instruction::Tag::Phi: _convert_instr_phi(node); break;
                case ir::Instruction::Tag::Return: _convert_instr_return(node); break;
                case ir::Instruction::Tag::Loop: _convert_instr_loop(node); break;
                case ir::Instruction::Tag::GenericLoop: _convert_instr_generic_loop(node); break;
                case ir::Instruction::Tag::Break: _convert_instr_break(node); break;
                case ir::Instruction::Tag::Continue: _convert_instr_continue(node); break;
                case ir::Instruction::Tag::If: _convert_instr_if(node); break;
                case ir::Instruction::Tag::Switch: _convert_instr_switch(node); break;
                case ir::Instruction::Tag::AdScope: _convert_instr_ad_scope(node); break;
                case ir::Instruction::Tag::AdDetach: _convert_instr_ad_detach(node); break;
                case ir::Instruction::Tag::Comment: _convert_instr_comment(node); break;
                case ir::Instruction::Tag::Debug: _convert_instr_debug(node); break;
                default: LUISA_ERROR_WITH_LOCATION("Invalid instruction in body.");
            }
            node_ref = node->next;
        }
        
        if (auto iter = _ctx->block_to_phis.find(block);
            iter != _ctx->block_to_phis.end()) {
            for (auto phi : iter->second) {
                _ctx->function_builder->assign(_convert_node(phi.dst), _convert_node(phi.src));
            }
        }
    }
    
    const Expression *IR2AST::_convert_node(ir::NodeRef node_ref) noexcept {
        return _convert_node(ir::luisa_compute_ir_node_get(node_ref));
    }

    const Expression *IR2AST::_convert_node(const ir::Node *node) noexcept {
        if (auto iter = _ctx->node_to_exprs.find(node); iter != _ctx->node_to_exprs.end()) {
            return iter->second;
        }
        auto type = _convert_type(node->type_.get());

        auto expr = [&, index = _ctx->node_to_exprs.size()]() -> const Expression * {
            switch (node->instruction->tag) {
                case ir::Instruction::Tag::Buffer: return _ctx->function_builder->buffer(type->element());
                case ir::Instruction::Tag::Bindless: return _ctx->function_builder->bindless_array();
                case ir::Instruction::Tag::Texture2D: [[fallthrough]];
                case ir::Instruction::Tag::Texture3D: {
                    // for Texture{2|3}D, type is vector<primitive,4>
                    // where primitive could be int, float or uint
                    auto dimension = node->instruction->tag == ir::Instruction::Tag::Texture2D ? 2u : 3u;
                    auto texture_type = Type::texture(type->element(), dimension);
                    return _ctx->function_builder->texture(texture_type);
                }
                case ir::Instruction::Tag::Accel: return _ctx->function_builder->accel();
                case ir::Instruction::Tag::Shared: return _ctx->function_builder->shared(type);
                case ir::Instruction::Tag::Local: return _ctx->function_builder->local(type);
                case ir::Instruction::Tag::UserData: LUISA_ERROR_WITH_LOCATION("Instruction 'UserData' is not implemented.");
                case ir::Instruction::Tag::Const: return _convert_constant(node->instruction->const_._0);
                case ir::Instruction::Tag::Call: {
                    LUISA_ASSERT(node->type_->tag != ir::Type::Tag::Void, "Cannot assign Void to variables.");
                    return _convert_instr_call(node);
                }
                default: break;
            }
            LUISA_ERROR_WITH_LOCATION("Invalid node type: {}.", (int)node->instruction->tag);
        }();
        if (!_ctx->zero_init) {
            _ctx->node_to_exprs.emplace(node, expr);
        }
        return expr;
    }

    void IR2AST::_convert_instr_local(const ir::Node *node) noexcept {
        auto type = _convert_type(node->type_.get());
        auto init = _convert_node(node->instruction->local.init);

        auto variable = _ctx->function_builder->local(type);
        // construct a local variable with certain type
        if (_ctx->zero_init) {
            _ctx->zero_init = false;
        } else {
            _ctx->function_builder->assign(variable, init);
        }
        // assign the init value to the variable

        // Remark: About zero_init
        // AST variables are zero initialized by default, which is not the case for IR variables.
        // So a local defninition in AST will be translated into a ZeroInitializer call + a local definition in IR
        // 
        // When we translate IR back to AST, we hope that we can remove useless ZeroInitializer calls
        // As ZeroInitializer calls all come right before the following local definition, we simply record them with a bool
        // _ctx->zero_init is true  => we no longer insert the assign (use default zero initialized instead)
        // _ctx->zero_init is false => we need to assign the value to the new defined variable.
    }

    void IR2AST::_convert_instr_user_data(const ir::Node *user_data) noexcept {
        LUISA_ERROR_WITH_LOCATION("Instruction 'UserData' is not implemented.");
    }

    void IR2AST::_convert_instr_invalid(const ir::Node *invalid) noexcept {
        // do nothing
    }

    void IR2AST::_convert_instr_const(const ir::Node *const_) noexcept {
        // do nothing
    }

    void IR2AST::_convert_instr_update(const ir::Node *node) noexcept {
        auto lhs = _convert_node(node->instruction->update.var);
        auto rhs = _convert_node(node->instruction->update.value);
        _ctx->function_builder->assign(lhs, rhs);
    }

    const Expression *IR2AST::_convert_instr_call(const ir::Node *node) noexcept {
        auto type = _convert_type(node->type_.get());
        auto &&[func, arg_slice] = node->instruction->call;
        auto args = luisa::span{arg_slice.ptr, arg_slice.len};
        auto builtin_func = [&](size_t arg_num, luisa::string_view name, CallOp call_op) -> const Expression * {
            auto argument_information = arg_num == 0 ?
                "no arguments" : arg_num == 1 ?
                "1 argument" : luisa::format("{} arguments", arg_num);
            LUISA_ASSERT(args.size() == arg_num, "`{}` takes {}.", name, argument_information);
            auto converted_args = luisa::vector<const Expression *>{};
            for (const auto &arg : args) {
                converted_args.push_back(_convert_node(arg));
            }
            if (type == nullptr) {
                _ctx->function_builder->call(call_op, luisa::span{converted_args});
                return nullptr;
            } else {
                return _ctx->function_builder->call(type, call_op, luisa::span{converted_args});
            }
        };
        auto unary_op = [&](luisa::string_view name, UnaryOp un_op) -> const Expression * {
            LUISA_ASSERT(args.size() == 1u, "`{}` takes 1 argument.", name);
            return _ctx->function_builder->unary(type, un_op, _convert_node(args[0]));
        };
        auto binary_op = [&](luisa::string_view name, BinaryOp bin_op) -> const Expression * {
            LUISA_ASSERT(args.size() == 2u, "`{}` takes 2 arguments.", name);
            return _ctx->function_builder->binary(type, bin_op, _convert_node(args[0]), _convert_node(args[1]));
        };
        auto make_vector = [&](size_t length) -> const Expression * {
            LUISA_ASSERT(args.size() == length, "`MakeVec` takes {} argument(s).", length);
            auto inner_type = ir::luisa_compute_ir_node_get(args[0])->type_.get();
            LUISA_ASSERT(inner_type->tag == ir::Type::Tag::Primitive, "`MakeVec` supports primitive type only.");
            LUISA_ASSERT(type->is_vector(), "`MakeVec` must return a vector.");

            auto converted_args = luisa::vector<const Expression *>{};
            for (const auto &arg : args) {
                converted_args.push_back(_convert_node(arg));
            }
            auto vector_op = _decide_make_vector_op(_convert_primitive_type(inner_type->primitive._0), type->dimension());
            return _ctx->function_builder->call(type, vector_op, luisa::span{converted_args});
        };
        auto rotate = [&](BinaryOp this_op) -> const Expression * {
            LUISA_ASSERT(this_op == BinaryOp::SHL || this_op == BinaryOp::SHR, "rotate is only valid with SHL and SHR.");
            LUISA_ASSERT(args.size() == 2u, "`RotLeft` and `RotRight` takes 2 arguments.");
            auto lhs = _convert_node(args[0]);
            auto rhs = _convert_node(args[1]);
            auto lhs_bit_length = (uint)lhs->type()->size() * 8;
            // this_op == SHL -> complement_op == SHR, vice versa
            auto complement_op = this_op == BinaryOp::SHL ? BinaryOp::SHR : BinaryOp::SHL;

            // rot_left(a, b) = (a << b) | (a >> (bit_length(a) - b))
            // rot_right(a, b) = (a >> b) | (a << (bit_length(a) - b))
            auto part1 = _ctx->function_builder->binary(
                lhs->type(),
                this_op,
                lhs,
                rhs
            ); 
            // (a << b) for rot_left, (a >> b) for rot_right
            auto shl_length = _ctx->function_builder->binary(
                rhs->type(),
                BinaryOp::SUB,
                _ctx->function_builder->literal(rhs->type(), lhs_bit_length),
                rhs
            ); // bit_length(a) - b
            auto part2 = _ctx->function_builder->binary(
                lhs->type(),
                complement_op,
                lhs,
                shl_length
            ); // (a >> (bit_length(a) - b)) for rot_left, (a << (bit_length(a) - b)) for rot_right
            return _ctx->function_builder->binary(lhs->type(), BinaryOp::BIT_OR, part1, part2);
        };
        auto make_matrix = [&](size_t dimension) -> const Expression * {
            LUISA_ASSERT(args.size() == dimension, "`Mat` takes {} argument(s).", dimension);
            LUISA_ASSERT(type->is_matrix(), "`Mat` must return a matrix.");
            auto matrix_dimension = type->dimension();
            auto converted_args = luisa::vector<const Expression *>{};
            for (const auto &arg : args) {
                converted_args.push_back(_convert_node(arg));
            }
            auto matrix_op = _decide_make_matrix_op(type->dimension());
            return _ctx->function_builder->call(type, matrix_op, luisa::span{converted_args});
        };

        auto constant_index = [](ir::NodeRef index) noexcept -> uint64_t  {
            auto node = ir::luisa_compute_ir_node_get(index);
            LUISA_ASSERT(node->instruction->tag == ir::Instruction::Tag::Const,
                "Index must be constant uint32.");
            auto &&c = node->instruction->const_._0;
            switch (c.tag) {
                case ir::Const::Tag::Zero: return 0;
                case ir::Const::Tag::One: return 1;
                case ir::Const::Tag::Int32: return c.int32._0;
                case ir::Const::Tag::Uint32: return c.uint32._0;
                case ir::Const::Tag::Int64: return c.int64._0;
                case ir::Const::Tag::Uint64: return c.uint64._0;
                case ir::Const::Tag::Generic: {
                    auto t = node->type_.get();
                    LUISA_ASSERT(t->tag == ir::Type::Tag::Primitive, "Invalid index type.");
                    auto do_cast = [&c]<typename T>() noexcept {
                        T x{};
                        std::memcpy(&x, c.generic._0.ptr, sizeof(T));
                        return static_cast<uint64_t>(x);
                    };
                    switch (t->primitive._0) {
                        case ir::Primitive::Int32: return do_cast.operator()<int32_t>();
                        case ir::Primitive::Uint32: return do_cast.operator()<uint32_t>();
                        case ir::Primitive::Int64: return do_cast.operator()<int64_t>();
                        case ir::Primitive::Uint64: return do_cast.operator()<uint64_t>();
                        default: break;
                    }
                }
                default: break;
            }
            LUISA_ERROR_WITH_LOCATION("Invalid index.");
        };
        switch (func.tag) {
            case ir::Func::Tag::ZeroInitializer: {
                LUISA_ASSERT(args.empty(), "`ZeroInitializer` takes no arguments.");
                _ctx->zero_init = true;
                return nullptr;
            }
            case ir::Func::Tag::Assume: return builtin_func(1, "Assume", CallOp::ASSUME);
            case ir::Func::Tag::Unreachable: return builtin_func(0, "Unreachable", CallOp::UNREACHABLE);
            case ir::Func::Tag::Assert: {
                LUISA_ASSERT(args.size() == 1u, "`Assert` takes 1 argument.");
                LUISA_WARNING_WITH_LOCATION("`Assert` not implemented currently.");
                return nullptr;
            }
            case ir::Func::Tag::ThreadId: {
                LUISA_ASSERT(args.empty(), "`ThreadId` takes no arguments.");
                return _ctx->function_builder->thread_id();
            }
            case ir::Func::Tag::BlockId: {
                LUISA_ASSERT(args.empty(), "`BlockId` takes no arguments.");
                return _ctx->function_builder->block_id();
            }
            case ir::Func::Tag::DispatchId: {
                LUISA_ASSERT(args.empty(), "`DispatchId` takes no arguments.");
                return _ctx->function_builder->dispatch_id();
            }
            case ir::Func::Tag::DispatchSize:{
                LUISA_ASSERT(args.empty(), "`DispatchSize` takes no arguments.");
                return _ctx->function_builder->dispatch_size();
            }
            case ir::Func::Tag::RequiresGradient: return builtin_func(1, "RequiresGradient", CallOp::REQUIRES_GRADIENT);
            case ir::Func::Tag::Gradient: return builtin_func(1, "Gradient", CallOp::GRADIENT);
            case ir::Func::Tag::GradientMarker: return builtin_func(2, "GradientMarker", CallOp::GRADIENT_MARKER);
            case ir::Func::Tag::AccGrad: return builtin_func(2, "AccGrad", CallOp::ACCUMULATE_GRADIENT);
            case ir::Func::Tag::Detach: return builtin_func(1, "Detach", CallOp::DETACH);
            case ir::Func::Tag::RayTracingInstanceTransform: return builtin_func(2, "RayTracingInstanceTransform", CallOp::RAY_TRACING_INSTANCE_TRANSFORM);
            case ir::Func::Tag::RayTracingSetInstanceTransform: return builtin_func(3, "RayTracingSetInstanceTransform", CallOp::RAY_TRACING_SET_INSTANCE_TRANSFORM);
            case ir::Func::Tag::RayTracingSetInstanceVisibility: return builtin_func(3, "RayTracingSetInstanceVisibility", CallOp::RAY_TRACING_SET_INSTANCE_VISIBILITY);
            case ir::Func::Tag::RayTracingSetInstanceOpacity: return builtin_func(3, "RayTracingSetInstanceOpacity", CallOp::RAY_TRACING_SET_INSTANCE_OPACITY);
            case ir::Func::Tag::RayTracingTraceClosest: return builtin_func(3, "RayTracingTraceClosest", CallOp::RAY_TRACING_TRACE_CLOSEST);
            case ir::Func::Tag::RayTracingTraceAny: return builtin_func(3, "RayTracingTraceAny", CallOp::RAY_TRACING_TRACE_ANY);
            case ir::Func::Tag::RayQueryProceed: return builtin_func(3, "RayQueryProceed", CallOp::RAY_QUERY_PROCEED);
            case ir::Func::Tag::RayQueryIsCandidateTriangle: return builtin_func(1, "RayQueryIsCandidateTriangle", CallOp::RAY_QUERY_IS_CANDIDATE_TRIANGLE);
            case ir::Func::Tag::RayQueryProceduralCandidateHit: return builtin_func(1, "RayQueryProceduralCandidateHit", CallOp::RAY_QUERY_PROCEDURAL_CANDIDATE_HIT);
            case ir::Func::Tag::RayQueryTriangleCandidateHit: return builtin_func(1, "RayQueryTriangleCandidateHit", CallOp::RAY_QUERY_TRIANGLE_CANDIDATE_HIT);
            case ir::Func::Tag::RayQueryCommittedHit: return builtin_func(1, "RayQueryCommittedHit", CallOp::RAY_QUERY_COMMITTED_HIT);
            case ir::Func::Tag::RayQueryCommitTriangle: return builtin_func(1, "RayQueryCommitTriangle", CallOp::RAY_QUERY_COMMIT_TRIANGLE);
            case ir::Func::Tag::RayQueryCommitProcedural: return builtin_func(1, "RayQueryCommitProcedural", CallOp::RAY_QUERY_COMMIT_PROCEDURAL);
            case ir::Func::Tag::Load: {
                LUISA_ASSERT(args.size() == 1u, "`Load` takes 1 argument.");
                return _convert_node(args[0]);
            }
            case ir::Func::Tag::Cast: {
                LUISA_ASSERT(args.size() == 1u, "Cast takes 1 argument.");
                switch (type->tag()) {
                    case Type::Tag::VECTOR: {
                        auto primitive = type->element();
                        auto length = type->dimension();
                        auto vector_op = _decide_make_vector_op(primitive, length);
                        return _ctx->function_builder->call(type, vector_op, {_convert_node(args[0])});
                    }
                    default: return _ctx->function_builder->cast(type, CastOp::STATIC, _convert_node(args[0]));
                }
            }
            case ir::Func::Tag::Bitcast: {
                LUISA_ASSERT(args.size() == 1u, "BitCast takes 1 argument.");
                return _ctx->function_builder->cast(type, CastOp::BITWISE, _convert_node(args[0]));
            }
            case ir::Func::Tag::Add: return binary_op("Add", BinaryOp::ADD);
            case ir::Func::Tag::Sub: return binary_op("Sub", BinaryOp::SUB);
            case ir::Func::Tag::Mul: return binary_op("Mul", BinaryOp::MUL);
            case ir::Func::Tag::Div: return binary_op("Div", BinaryOp::DIV);
            case ir::Func::Tag::Rem: return binary_op("Rem", BinaryOp::MOD);
            case ir::Func::Tag::BitAnd: return binary_op("BitAnd", BinaryOp::BIT_AND);
            case ir::Func::Tag::BitOr: return binary_op("BitOr", BinaryOp::BIT_OR);
            case ir::Func::Tag::BitXor: return binary_op("BitXor", BinaryOp::BIT_XOR);
            case ir::Func::Tag::Shl: return binary_op("Shl", BinaryOp::SHL);
            case ir::Func::Tag::Shr: return binary_op("Shr", BinaryOp::SHR);
            case ir::Func::Tag::RotRight: return rotate(BinaryOp::SHR);
            case ir::Func::Tag::RotLeft: return rotate(BinaryOp::SHL);
            case ir::Func::Tag::Eq: return binary_op("Eq", BinaryOp::EQUAL);
            case ir::Func::Tag::Ne: return binary_op("Ne", BinaryOp::NOT_EQUAL);
            case ir::Func::Tag::Lt: return binary_op("Lt", BinaryOp::LESS);
            case ir::Func::Tag::Le: return binary_op("Le", BinaryOp::LESS_EQUAL);
            case ir::Func::Tag::Gt: return binary_op("Gt", BinaryOp::GREATER);
            case ir::Func::Tag::Ge: return binary_op("Ge", BinaryOp::GREATER_EQUAL);
            case ir::Func::Tag::MatCompMul: return builtin_func(2, "MatCompMul", CallOp::MATRIX_COMPONENT_WISE_MULTIPLICATION);
            case ir::Func::Tag::Neg: return unary_op("Neg", UnaryOp::MINUS);
            case ir::Func::Tag::Not: return unary_op("Not", UnaryOp::NOT);
            case ir::Func::Tag::BitNot: return unary_op("BitNot", UnaryOp::BIT_NOT);
            case ir::Func::Tag::All: return builtin_func(1, "All", CallOp::ALL);
            case ir::Func::Tag::Any: return builtin_func(1, "Any", CallOp::ANY);
            case ir::Func::Tag::Select: return builtin_func(3, "Select", CallOp::SELECT);
            case ir::Func::Tag::Clamp: return builtin_func(3, "Clamp", CallOp::CLAMP);
            case ir::Func::Tag::Lerp: return builtin_func(3, "Lerp", CallOp::LERP);
            case ir::Func::Tag::Step: return builtin_func(2, "Step", CallOp::STEP);
            case ir::Func::Tag::Abs: return builtin_func(1, "Abs", CallOp::ABS);
            case ir::Func::Tag::Min: return builtin_func(2, "Min", CallOp::MIN);
            case ir::Func::Tag::Max: return builtin_func(2, "Max", CallOp::MAX);
            case ir::Func::Tag::ReduceSum: return builtin_func(1, "ReduceSum", CallOp::REDUCE_SUM);
            case ir::Func::Tag::ReduceProd: return builtin_func(1, "ReduceProd", CallOp::REDUCE_PRODUCT);
            case ir::Func::Tag::ReduceMin: return builtin_func(1, "ReduceMin", CallOp::REDUCE_MIN);
            case ir::Func::Tag::ReduceMax: return builtin_func(1, "ReduceMax", CallOp::REDUCE_MAX);
            case ir::Func::Tag::Clz: return builtin_func(1, "Clz", CallOp::CLZ);
            case ir::Func::Tag::Ctz: return builtin_func(1, "Ctz", CallOp::CTZ);
            case ir::Func::Tag::PopCount: return builtin_func(1, "PopCount", CallOp::POPCOUNT);
            case ir::Func::Tag::Reverse: return builtin_func(1, "Reverse", CallOp::REVERSE);
            case ir::Func::Tag::IsInf: return builtin_func(1, "IsInf", CallOp::ISINF);
            case ir::Func::Tag::IsNan: return builtin_func(1, "IsNan", CallOp::ISNAN);
            case ir::Func::Tag::Acos: return builtin_func(1, "Acos", CallOp::ACOS);
            case ir::Func::Tag::Acosh: return builtin_func(1, "Acosh", CallOp::ACOSH);
            case ir::Func::Tag::Asin: return builtin_func(1, "Asin", CallOp::ASIN);
            case ir::Func::Tag::Asinh: return builtin_func(1, "Asinh", CallOp::ASINH);
            case ir::Func::Tag::Atan: return builtin_func(1, "Atan", CallOp::ATAN);
            case ir::Func::Tag::Atan2: return builtin_func(2, "Atan2", CallOp::ATAN2);
            case ir::Func::Tag::Atanh: return builtin_func(1, "Atanh", CallOp::ATANH);
            case ir::Func::Tag::Cos: return builtin_func(1, "Cos", CallOp::COS);
            case ir::Func::Tag::Cosh: return builtin_func(1, "Cosh", CallOp::COSH);
            case ir::Func::Tag::Sin: return builtin_func(1, "Sin", CallOp::SIN);
            case ir::Func::Tag::Sinh: return builtin_func(1, "Sinh", CallOp::SINH);
            case ir::Func::Tag::Tan: return builtin_func(1, "Tan", CallOp::TAN);
            case ir::Func::Tag::Tanh: return builtin_func(1, "Tanh", CallOp::TANH);
            case ir::Func::Tag::Exp: return builtin_func(1, "Exp", CallOp::EXP);
            case ir::Func::Tag::Exp2: return builtin_func(1, "Exp2", CallOp::EXP2);
            case ir::Func::Tag::Exp10: return builtin_func(1, "Exp10", CallOp::EXP10);
            case ir::Func::Tag::Log: return builtin_func(1, "Log", CallOp::LOG);
            case ir::Func::Tag::Log2: return builtin_func(1, "Log2", CallOp::LOG2);
            case ir::Func::Tag::Log10: return builtin_func(1, "Log10", CallOp::LOG10);
            case ir::Func::Tag::Powi: return builtin_func(2, "Powi", CallOp::POW);
            case ir::Func::Tag::Powf: return builtin_func(2, "Powf", CallOp::POW);
            case ir::Func::Tag::Sqrt: return builtin_func(1, "Sqrt", CallOp::SQRT);
            case ir::Func::Tag::Rsqrt: return builtin_func(1, "Rsqrt", CallOp::RSQRT);
            case ir::Func::Tag::Ceil: return builtin_func(1, "Ceil", CallOp::CEIL);
            case ir::Func::Tag::Floor: return builtin_func(1, "Floor", CallOp::FLOOR);
            case ir::Func::Tag::Fract: return builtin_func(1, "Fract", CallOp::FRACT);
            case ir::Func::Tag::Trunc: return builtin_func(1, "Trunc", CallOp::TRUNC);
            case ir::Func::Tag::Round: return builtin_func(1, "Round", CallOp::ROUND);
            case ir::Func::Tag::Fma: return builtin_func(3, "Fma", CallOp::FMA);
            case ir::Func::Tag::Copysign: return builtin_func(2, "Copysign", CallOp::COPYSIGN);
            case ir::Func::Tag::Cross: return builtin_func(2, "Cross", CallOp::CROSS);
            case ir::Func::Tag::Dot: return builtin_func(2, "Dot", CallOp::DOT);
            case ir::Func::Tag::OuterProduct: return builtin_func(2, "OuterProduct", CallOp::OUTER_PRODUCT);
            case ir::Func::Tag::Length: return builtin_func(1, "Length", CallOp::LENGTH);
            case ir::Func::Tag::LengthSquared: return builtin_func(1, "LengthSquared", CallOp::LENGTH_SQUARED);
            case ir::Func::Tag::Normalize: return builtin_func(1, "Normalize", CallOp::NORMALIZE);
            case ir::Func::Tag::Faceforward: return builtin_func(3, "Faceforward", CallOp::FACEFORWARD);
            case ir::Func::Tag::Determinant: return builtin_func(1, "Determinant", CallOp::DETERMINANT);
            case ir::Func::Tag::Transpose: return builtin_func(1, "Transpose", CallOp::TRANSPOSE);
            case ir::Func::Tag::Inverse: return builtin_func(1, "Inverse", CallOp::INVERSE);
            case ir::Func::Tag::SynchronizeBlock: return builtin_func(0, "SynchronizeBlock", CallOp::SYNCHRONIZE_BLOCK);
            case ir::Func::Tag::AtomicExchange: return builtin_func(3, "AtomicExchange", CallOp::ATOMIC_EXCHANGE);
            case ir::Func::Tag::AtomicCompareExchange: return builtin_func(4, "AtomicCompareExchange", CallOp::ATOMIC_COMPARE_EXCHANGE);
            case ir::Func::Tag::AtomicFetchAdd: return builtin_func(3, "AtomicFetchAdd", CallOp::ATOMIC_FETCH_ADD);
            case ir::Func::Tag::AtomicFetchSub: return builtin_func(3, "AtomicFetchSub", CallOp::ATOMIC_FETCH_SUB);
            case ir::Func::Tag::AtomicFetchAnd: return builtin_func(3, "AtomicFetchAnd", CallOp::ATOMIC_FETCH_AND);
            case ir::Func::Tag::AtomicFetchOr: return builtin_func(3, "AtomicFetchOr", CallOp::ATOMIC_FETCH_OR);
            case ir::Func::Tag::AtomicFetchXor: return builtin_func(3, "AtomicFetchXor", CallOp::ATOMIC_FETCH_XOR);
            case ir::Func::Tag::AtomicFetchMin: return builtin_func(3, "AtomicFetchMin", CallOp::ATOMIC_FETCH_MIN);
            case ir::Func::Tag::AtomicFetchMax: return builtin_func(3, "AtomicFetchMax", CallOp::ATOMIC_FETCH_MAX);
            case ir::Func::Tag::BufferRead: return builtin_func(2, "BufferRead", CallOp::BUFFER_READ);
            case ir::Func::Tag::BufferWrite: return builtin_func(3, "BufferWrite", CallOp::BUFFER_WRITE);
            case ir::Func::Tag::BufferSize: return builtin_func(1, "BufferSize", CallOp::BUFFER_SIZE);
            case ir::Func::Tag::Texture2dRead: return builtin_func(2, "Texture2DRead", CallOp::TEXTURE_READ);
            case ir::Func::Tag::Texture3dRead: return builtin_func(2, "Texture2DRead", CallOp::TEXTURE_READ);
            case ir::Func::Tag::Texture2dWrite: return builtin_func(3, "TextureWrite", CallOp::TEXTURE_WRITE);
            case ir::Func::Tag::Texture3dWrite: return builtin_func(3, "TextureWrite", CallOp::TEXTURE_WRITE);
            case ir::Func::Tag::BindlessTexture2dSample: return builtin_func(3, "BindlessTexture2dSample", CallOp::BINDLESS_TEXTURE2D_SAMPLE);
            case ir::Func::Tag::BindlessTexture2dSampleLevel: return builtin_func(4, "BindlessTexture2dSampleLevel", CallOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL);
            case ir::Func::Tag::BindlessTexture2dSampleGrad: return builtin_func(5, "BindlessTexture2dSampleGrad", CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD);
            case ir::Func::Tag::BindlessTexture3dSample: return builtin_func(3, "BindlessTexture3dSample", CallOp::BINDLESS_TEXTURE3D_SAMPLE);
            case ir::Func::Tag::BindlessTexture3dSampleLevel: return builtin_func(4, "BindlessTexture3dSampleLevel", CallOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL);
            case ir::Func::Tag::BindlessTexture3dSampleGrad: return builtin_func(5, "BindlessTexture3dSampleGrad", CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD);
            case ir::Func::Tag::BindlessTexture2dRead: return builtin_func(2, "BindlessTexture2dRead", CallOp::BINDLESS_TEXTURE2D_READ);
            case ir::Func::Tag::BindlessTexture3dRead: return builtin_func(2, "BindlessTexture3dRead", CallOp::BINDLESS_TEXTURE3D_READ);
            case ir::Func::Tag::BindlessTexture2dReadLevel: return builtin_func(3, "BindlessTexture2dReadLevel", CallOp::BINDLESS_TEXTURE2D_READ_LEVEL);
            case ir::Func::Tag::BindlessTexture3dReadLevel: return builtin_func(3, "BindlessTexture3dReadLevel", CallOp::BINDLESS_TEXTURE3D_READ_LEVEL);
            case ir::Func::Tag::BindlessTexture2dSize: return builtin_func(1, "BindlessTexture2dSize", CallOp::BINDLESS_TEXTURE2D_SIZE);
            case ir::Func::Tag::BindlessTexture3dSize: return builtin_func(1, "BindlessTexture3dSize", CallOp::BINDLESS_TEXTURE3D_SIZE);
            case ir::Func::Tag::BindlessTexture2dSizeLevel: return builtin_func(2, "BindlessTexture2dSizeLevel", CallOp::BINDLESS_TEXTURE2D_SIZE_LEVEL);
            case ir::Func::Tag::BindlessTexture3dSizeLevel: return builtin_func(2, "BindlessTexture3dSizeLevel", CallOp::BINDLESS_TEXTURE3D_SIZE_LEVEL);
            case ir::Func::Tag::BindlessBufferRead: return builtin_func(2, "BindlessBufferRead", CallOp::BINDLESS_BUFFER_READ);
            case ir::Func::Tag::BindlessBufferSize: return builtin_func(1, "BindlessBufferSize", CallOp::BINDLESS_BUFFER_SIZE);
            case ir::Func::Tag::BindlessBufferType: return builtin_func(1, "BindlessBufferType", CallOp::BINDLESS_BUFFER_TYPE);
            case ir::Func::Tag::Vec: return make_vector(1);
            case ir::Func::Tag::Vec2: return make_vector(2);
            case ir::Func::Tag::Vec3: return make_vector(3);
            case ir::Func::Tag::Vec4: return make_vector(4);
            case ir::Func::Tag::Permute: {
                auto src = _convert_node(args[0]);
                LUISA_ASSERT(type->is_vector() || type->is_matrix(), "Invalid vector type");
                LUISA_ASSERT(args.size() == type->dimension() + 1, "Vector type inconsistent with arguments.");
                auto inner_type = type->element();

                auto reconstruct_args = luisa::vector<const Expression *>{};
                for (const auto &arg : args) {
                    auto reconstruct_arg = _ctx->function_builder->access(inner_type, src, _convert_node(arg));
                    reconstruct_args.push_back(reconstruct_arg);
                }
                auto op = type->is_vector() ? 
                    _decide_make_vector_op(inner_type, type->dimension()) :
                    _decide_make_matrix_op(type->dimension());
                return _ctx->function_builder->call(type, op, luisa::span{reconstruct_args});
            }
            case ir::Func::Tag::InsertElement: {
                // for Sturct, a.c = b
                // for Vector/Matrix, a[c] = b;
                LUISA_ASSERT(args.size() == 3u, "InsertElement takes 3 arguments.");
                auto self = ir::luisa_compute_ir_node_get(args[0]);
                auto self_type = _convert_type(self->type_.get());
                auto new_value = _convert_node(args[1]);
                if (self->type_->tag == ir::Type::Tag::Struct) {
                    auto member_index = constant_index(args[2]);
                    auto member_type = self_type->members()[member_index];
                    auto ref = _ctx->function_builder->member(member_type, _convert_node(self), member_index);
                    _ctx->function_builder->assign(ref, new_value);
                } else {
                    auto index = _convert_node(args[2]);
                    auto inner_type = self_type->element();
                    auto ref = _ctx->function_builder->access(inner_type, _convert_node(self), index);
                    _ctx->function_builder->assign(ref, new_value);
                }
                return nullptr;
            }
            case ir::Func::Tag::ExtractElement: [[fallthrough]];
            case ir::Func::Tag::GetElementPtr: {
                auto op = func.tag == ir::Func::Tag::ExtractElement ?
                    "ExtractElement" :
                    "GetElementPtr";
                LUISA_ASSERT(args.size() == 2u, "{} takes 2 arguments.", op);
                auto self = ir::luisa_compute_ir_node_get(args[0]);
                auto self_type = _convert_type(self->type_.get());
                if (self->type_->tag == ir::Type::Tag::Struct) {
                    auto member_index = constant_index(args[1]);
                    auto member_type = self_type->members()[member_index];
                    return _ctx->function_builder->member(member_type, _convert_node(self), member_index);
                } else {
                    auto index = _convert_node(args[1]);
                    auto inner_type = self_type->element();
                    return _ctx->function_builder->access(inner_type, _convert_node(self), index);
                }
            }
            case ir::Func::Tag::Struct: {
                auto alignment = node->type_->struct_._0.alignment;
                auto fields = luisa::vector<const Type *>{};
                auto converted_args = luisa::vector<const Expression *>{};
                for (const auto &arg_ref : args) {
                    auto arg = ir::luisa_compute_ir_node_get(arg_ref);
                    fields.push_back(_convert_type(arg->type_.get()));
                }
                auto struct_type = Type::structure(alignment, fields);

                auto struct_instance = _ctx->function_builder->local(struct_type);
                for (auto member_index = 0u; member_index < args.size(); member_index++) {
                    auto member_type = struct_type->members()[member_index];
                    auto access = _ctx->function_builder->member(member_type, struct_instance, member_index);
                    _ctx->function_builder->assign(access, _convert_node(args[member_index]));
                }
                return struct_instance;
            }
            case ir::Func::Tag::Mat: return make_matrix(1);
            case ir::Func::Tag::Mat2: return make_matrix(2);
            case ir::Func::Tag::Mat3: return make_matrix(3);
            case ir::Func::Tag::Mat4: return make_matrix(4);
            case ir::Func::Tag::Callable: {
                auto p_callable = func.callable._0._0;
                LUISA_ASSERT(!p_callable.is_null(), "Invalid callable.");
                auto converted_args = luisa::vector<const Expression *>{};
                for (const auto &arg : args) {
                    converted_args.push_back(_convert_node(arg));
                }
                auto callable = convert_callable(p_callable.get())->function();
            }
            case ir::Func::Tag::CpuCustomOp: LUISA_ERROR_WITH_LOCATION("CpuCustomOp is not implemented.");
            // FIXME: these tags are deprecated and will be removed.
            case ir::Func::Tag::RayTracingInstanceAabb: [[fallthrough]];
            case ir::Func::Tag::RayTracingInstanceVisibility: [[fallthrough]];
            case ir::Func::Tag::RayTracingInstanceOpacity: [[fallthrough]];
            case ir::Func::Tag::RayTracingSetInstanceAabb: [[fallthrough]];
            default: LUISA_ERROR_WITH_LOCATION("Invalid function tag.");
        }
        return nullptr;
    }

    void IR2AST::_convert_instr_phi(const ir::Node *phi) noexcept {
        // do nothing
    }

    void IR2AST::_convert_instr_return(const ir::Node *node) noexcept {
        if (auto ret = node->instruction->return_._0; ret != ir::INVALID_REF) {
            _ctx->function_builder->return_(_convert_node(ret));
        } else {
            _ctx->function_builder->return_(nullptr);
        }
    }

    void IR2AST::_convert_instr_loop(const ir::Node *node) noexcept {
        // loop {
        //     body();
        //     if (!cond) {
        //         break;
        //     }
        // }
        auto cond = _convert_node(node->instruction->loop.cond);
        auto loop_scope = _ctx->function_builder->loop_();
        _ctx->function_builder->push_scope(loop_scope->body());
        _convert_block(node->instruction->loop.body.get());
        auto if_scope = _ctx->function_builder->if_(cond);
        _ctx->function_builder->push_scope(if_scope->false_branch());
        _ctx->function_builder->break_();
        _ctx->function_builder->pop_scope(if_scope->false_branch());
        _ctx->function_builder->pop_scope(loop_scope->body());
    }

    void IR2AST::_convert_instr_generic_loop(const ir::Node *node) noexcept {
        // bool first_entrance = true;
        // loop {
        //     if (!first_entrance) {
        //         update();
        //     } else {
        //         first_entrance = false;
        //     }
        //     prepare();
        //     if (!cond()) break;
        //     body();
        // }
        auto first_entrance = _ctx->function_builder->local(Type::from("bool"));
        _ctx->function_builder->assign(first_entrance, _ctx->function_builder->literal(Type::from("bool"), true));
        auto loop_scope = _ctx->function_builder->loop_();
        _ctx->function_builder->push_scope(loop_scope->body());
        auto update_if_scope = _ctx->function_builder->if_(first_entrance);
        _ctx->function_builder->push_scope(update_if_scope->true_branch());
        _ctx->function_builder->assign(first_entrance, _ctx->function_builder->literal(Type::from("bool"), false));
        _ctx->function_builder->pop_scope(update_if_scope->true_branch());
        _ctx->function_builder->push_scope(update_if_scope->false_branch());
        _convert_block(node->instruction->generic_loop.update.get());
        _ctx->function_builder->pop_scope(update_if_scope->false_branch());
        _convert_block(node->instruction->generic_loop.prepare.get());
        auto loop_cond = _convert_node(node->instruction->generic_loop.cond);
        auto cond_if_scope = _ctx->function_builder->if_(loop_cond);
        _ctx->function_builder->push_scope(cond_if_scope->true_branch());
        _ctx->function_builder->break_();
        _ctx->function_builder->pop_scope(cond_if_scope->true_branch());
        _convert_block(node->instruction->generic_loop.body.get());
        _ctx->function_builder->pop_scope(loop_scope->body());
    }

    void IR2AST::_convert_instr_break(const ir::Node *node) noexcept {
        _ctx->function_builder->break_();
    }

    void IR2AST::_convert_instr_continue(const ir::Node *node) noexcept {
        _ctx->function_builder->continue_();
    }

    void IR2AST::_convert_instr_if(const ir::Node *node) noexcept {
        auto cond = _convert_node(node->instruction->if_.cond);
        auto if_scope = _ctx->function_builder->if_(cond);
        _ctx->function_builder->push_scope(if_scope->true_branch());
        _convert_block(node->instruction->if_.true_branch.get());
        _ctx->function_builder->pop_scope(if_scope->true_branch());
        _ctx->function_builder->push_scope(if_scope->false_branch());
        _convert_block(node->instruction->if_.false_branch.get());
        _ctx->function_builder->pop_scope(if_scope->false_branch());
    }

    void IR2AST::_convert_instr_switch(const ir::Node *node) noexcept {
        auto value = _convert_node(node->instruction->switch_.value);
        auto switch_scope = _ctx->function_builder->switch_(value);
        _ctx->function_builder->push_scope(switch_scope->body());
        auto data = node->instruction->switch_.cases.ptr;
        auto len = node->instruction->switch_.cases.len / sizeof(ir::SwitchCase);
        for (auto i = 0; i < len; i++) {
            auto value = _ctx->function_builder->literal(Type::from("int"), data[i].value);
            auto case_scope = _ctx->function_builder->case_(value);
            _ctx->function_builder->push_scope(case_scope->body());
            _convert_block(data[i].block.get());
            _ctx->function_builder->pop_scope(case_scope->body());
        }
        auto default_scope = _ctx->function_builder->default_();
        _ctx->function_builder->push_scope(default_scope->body());
        _convert_block(node->instruction->switch_.default_.get());
        _ctx->function_builder->pop_scope(default_scope->body());
        _ctx->function_builder->pop_scope(switch_scope->body());
    }

    void IR2AST::_convert_instr_ad_scope(const ir::Node *node) noexcept {
        _ctx->function_builder->comment_("ADScope Forward Begin");
        _convert_block(node->instruction->ad_scope.forward.get());
        _ctx->function_builder->comment_("ADScope Forward End");
        _ctx->function_builder->comment_("ADScope Backward Begin");
        _convert_block(node->instruction->ad_scope.backward.get());
        _ctx->function_builder->comment_("ADScope Backward End");
        _ctx->function_builder->comment_("ADScope Epilogue Begin");
        _convert_block(node->instruction->ad_scope.epilogue.get());
        _ctx->function_builder->comment_("ADScope Epilogue End");
    }

    void IR2AST::_convert_instr_ad_detach(const ir::Node *node) noexcept {
        _ctx->function_builder->comment_("AD Detach Begin");
        _convert_block(node->instruction->ad_detach._0.get());
        _ctx->function_builder->comment_("AD Detach End");
    }

    void IR2AST::_convert_instr_comment(const ir::Node *node) noexcept {
        auto comment_body = node->instruction->comment._0;
        auto comment_content = luisa::string{reinterpret_cast<const char *>(comment_body.ptr), comment_body.len};
        _ctx->function_builder->comment_(comment_content);
    }

    void IR2AST::_convert_instr_debug(const ir::Node *node) noexcept {
        LUISA_WARNING_WITH_LOCATION("Instruction 'Debug' is not implemented.");
        auto debug_body = node->instruction->debug._0;
        auto debug_content = luisa::string_view{reinterpret_cast<const char *>(debug_body.ptr), debug_body.len};
        _ctx->function_builder->comment_(luisa::format("Debug: {}", debug_content));
    }

    const Expression *IR2AST::_convert_constant(const ir::Const &const_) noexcept {
        auto decode = []<typename T>(const uint8_t *data) noexcept {
            T x{};
            std::memcpy(&x, data, sizeof(T));
            return x;
        };
        // Primitive -> inline as literal
        // Generic
        //     Primitive -> inline as literal
        //     Array or Vector or Matrix -> Constant
        //     Other -> Error
        switch (const_.tag) {
            case ir::Const::Tag::Zero:
                return _ctx->function_builder->literal(_convert_type(const_.zero._0.get()), 0);
            case ir::Const::Tag::One:
                return _ctx->function_builder->literal(_convert_type(const_.one._0.get()), 1);
            case ir::Const::Tag::Bool:
                return _ctx->function_builder->literal(Type::from("bool"), const_.bool_._0);
            case ir::Const::Tag::Float32:
                return _ctx->function_builder->literal(Type::from("float"), const_.float32._0);
            case ir::Const::Tag::Int32:
                return _ctx->function_builder->literal(Type::from("int"), const_.int32._0);
            case ir::Const::Tag::Uint32:
                return _ctx->function_builder->literal(Type::from("uint"), const_.uint32._0);
            case ir::Const::Tag::Generic: {
                // LUISA_VERBOSE_WITH_LOCATION("converting Generic const: {}", (int)const_.generic._1->tag);
                switch (auto type = const_.generic._1.get(); type->tag) {
                    case ir::Type::Tag::Primitive: {
                        auto &&data = const_.generic._0;
                        switch (type->primitive._0) {
                            case ir::Primitive::Bool: return _ctx->function_builder->literal(Type::from("bool"), decode.operator()<bool>(data.ptr));
                            case ir::Primitive::Float32: return _ctx->function_builder->literal(Type::from("float"), decode.operator()<float>(data.ptr));
                            case ir::Primitive::Int32: return _ctx->function_builder->literal(Type::from("int"), decode.operator()<int32_t>(data.ptr));
                            case ir::Primitive::Uint32: return _ctx->function_builder->literal(Type::from("uint"), decode.operator()<uint32_t>(data.ptr));
                            case ir::Primitive::Float64: [[fallthrough]];
                            case ir::Primitive::Int64: [[fallthrough]];
                            case ir::Primitive::Uint64: [[fallthrough]];
                            default: LUISA_ERROR_WITH_LOCATION("64-bit primitive types are not yet supported.");
                        }
                    }
                    case ir::Type::Tag::Matrix: {
                        auto &&[elem, dimension] = type->matrix._0;
                        auto &&data = const_.generic._0;
                        auto matrix_type = _convert_type(type);
                        auto &&primitive_type = elem.scalar._0;
                        
                        switch (elem.tag) {
                            case ir::VectorElementType::Tag::Scalar: {
                                switch (dimension) {
                                    case 2: return _ctx->function_builder->literal(matrix_type, decode.operator()<float2x2>(data.ptr));                   
                                    case 3: return _ctx->function_builder->literal(matrix_type, decode.operator()<float3x3>(data.ptr));
                                    case 4: return _ctx->function_builder->literal(matrix_type, decode.operator()<float4x4>(data.ptr));
                                    default: LUISA_ERROR_WITH_LOCATION("Matrices with dimension other than 2, 3 and 4 are not supported.");
                                }
                            }
                            case ir::VectorElementType::Tag::Vector: LUISA_ERROR_WITH_LOCATION("Vector of vector is not supported.");
                        }   
                    }
                    case ir::Type::Tag::Vector: {
                        auto &&[elem, length] = type->vector._0;
                        auto &&data = const_.generic._0;
                        auto vector_type = _convert_type(type);
                        auto &&primitive_type = elem.scalar._0;
                        switch (elem.tag) {
                            case ir::VectorElementType::Tag::Scalar: {
                                switch (primitive_type) {
                                    case ir::Primitive::Bool: {
                                        switch (length) {
                                            case 2: return _ctx->function_builder->literal(vector_type, decode.operator()<bool2>(data.ptr));                   
                                            case 3: return _ctx->function_builder->literal(vector_type, decode.operator()<bool3>(data.ptr));
                                            case 4: return _ctx->function_builder->literal(vector_type, decode.operator()<bool4>(data.ptr));
                                            default: LUISA_ERROR_WITH_LOCATION("Vectors with length other than 2, 3 and 4 are not supported.");
                                        }
                                    }
                                    case ir::Primitive::Float32: {
                                        switch (length) {
                                            case 2: return _ctx->function_builder->literal(vector_type, decode.operator()<float2>(data.ptr));                   
                                            case 3: return _ctx->function_builder->literal(vector_type, decode.operator()<float3>(data.ptr));
                                            case 4: return _ctx->function_builder->literal(vector_type, decode.operator()<float4>(data.ptr));
                                            default: LUISA_ERROR_WITH_LOCATION("Vectors with length other than 2, 3 and 4 are not supported.");
                                        }
                                    }
                                    case ir::Primitive::Int32: {
                                        switch (length) {
                                            case 2: return _ctx->function_builder->literal(vector_type, decode.operator()<int2>(data.ptr));                   
                                            case 3: return _ctx->function_builder->literal(vector_type, decode.operator()<int3>(data.ptr));
                                            case 4: return _ctx->function_builder->literal(vector_type, decode.operator()<int4>(data.ptr));
                                            default: LUISA_ERROR_WITH_LOCATION("Vectors with length other than 2, 3 and 4 are not supported.");
                                        }
                                    }
                                    case ir::Primitive::Uint32: {
                                        switch (length) {
                                            case 2: return _ctx->function_builder->literal(vector_type, decode.operator()<uint2>(data.ptr));                   
                                            case 3: return _ctx->function_builder->literal(vector_type, decode.operator()<uint3>(data.ptr));
                                            case 4: return _ctx->function_builder->literal(vector_type, decode.operator()<uint4>(data.ptr));
                                            default: LUISA_ERROR_WITH_LOCATION("Vectors with length other than 2, 3 and 4 are not supported.");
                                        }
                                    }
                                    case ir::Primitive::Float64: [[fallthrough]];
                                    case ir::Primitive::Int64: [[fallthrough]];
                                    case ir::Primitive::Uint64: [[fallthrough]];
                                    default: LUISA_ERROR_WITH_LOCATION("64-bit primitive types are not yet supported.");
                                }
                            }
                            case ir::VectorElementType::Tag::Vector: LUISA_ERROR_WITH_LOCATION("Vector of vector is not supported.");
                        }
                    }
                    case ir::Type::Tag::Array: {
                        auto elem = type->array._0.element.get();
                        auto elem_type = _convert_type(elem);
                        auto array_type = _convert_type(type);
                        auto &&data = const_.generic._0;
                        auto get_payload = [data]<typename T>() noexcept {
                            return luisa::span(reinterpret_cast<const T *>(data.ptr), data.len / sizeof(T));
                        };
                        switch (elem->tag) {
                            case ir::Type::Tag::Primitive: {
                                switch (elem->primitive._0) {
                                    case ir::Primitive::Bool: return _ctx->function_builder->constant(array_type, ConstantData::create(get_payload.operator()<bool>()));
                                    case ir::Primitive::Float32: return _ctx->function_builder->constant(array_type, ConstantData::create(get_payload.operator()<float>()));
                                    case ir::Primitive::Int32: return _ctx->function_builder->constant(array_type, ConstantData::create(get_payload.operator()<int32_t>()));
                                    case ir::Primitive::Uint32:  return _ctx->function_builder->constant(array_type, ConstantData::create(get_payload.operator()<uint32_t>()));
                                    case ir::Primitive::Float64: [[fallthrough]];
                                    case ir::Primitive::Int64: [[fallthrough]];
                                    case ir::Primitive::Uint64: [[fallthrough]];
                                    default: LUISA_ERROR_WITH_LOCATION("64-bit primitive types are not yet supported.");
                                }
                            }
                            case ir::Type::Tag::Matrix: {
                                switch (elem->matrix._0.dimension) {
                                    case 2: return _ctx->function_builder->constant(array_type, ConstantData::create(get_payload.operator()<float2x2>()));
                                    case 3: return _ctx->function_builder->constant(array_type, ConstantData::create(get_payload.operator()<float3x3>()));
                                    case 4: return _ctx->function_builder->constant(array_type, ConstantData::create(get_payload.operator()<float4x4>()));
                                    default: LUISA_ERROR_WITH_LOCATION("Matrices with dimension other than 2, 3 and 4 are not supported.");
                                }
                            }
                            case ir::Type::Tag::Vector: {
                                switch (elem->vector._0.element.tag) {
                                    case ir::VectorElementType::Tag::Scalar:{
                                        switch (elem->vector._0.element.scalar._0) {
                                            case ir::Primitive::Bool: {
                                                switch (elem->vector._0.length) {
                                                    case 2: return _ctx->function_builder->constant(array_type, ConstantData::create(get_payload.operator()<bool2>()));
                                                    case 3: return _ctx->function_builder->constant(array_type, ConstantData::create(get_payload.operator()<bool3>()));
                                                    case 4: return _ctx->function_builder->constant(array_type, ConstantData::create(get_payload.operator()<bool4>()));
                                                    default: LUISA_ERROR_WITH_LOCATION("Vectors with length other than 2, 3 and 4 are not supported.");
                                                }
                                            }
                                            case ir::Primitive::Float32: {
                                                switch (elem->vector._0.length) {
                                                    case 2: return _ctx->function_builder->constant(array_type, ConstantData::create(get_payload.operator()<float2>()));
                                                    case 3: return _ctx->function_builder->constant(array_type, ConstantData::create(get_payload.operator()<float3>()));
                                                    case 4: return _ctx->function_builder->constant(array_type, ConstantData::create(get_payload.operator()<float4>()));
                                                    default: LUISA_ERROR_WITH_LOCATION("Vectors with length other than 2, 3 and 4 are not supported.");
                                                }
                                            }
                                            case ir::Primitive::Int32: {
                                                switch (elem->vector._0.length) {
                                                    case 2: return _ctx->function_builder->constant(array_type, ConstantData::create(get_payload.operator()<int2>()));
                                                    case 3: return _ctx->function_builder->constant(array_type, ConstantData::create(get_payload.operator()<int3>()));
                                                    case 4: return _ctx->function_builder->constant(array_type, ConstantData::create(get_payload.operator()<int4>()));
                                                    default: LUISA_ERROR_WITH_LOCATION("Vectors with length other than 2, 3 and 4 are not supported.");
                                                }
                                            }
                                            case ir::Primitive::Uint32: {
                                                switch (elem->vector._0.length) {
                                                    case 2: return _ctx->function_builder->constant(array_type, ConstantData::create(get_payload.operator()<uint2>()));
                                                    case 3: return _ctx->function_builder->constant(array_type, ConstantData::create(get_payload.operator()<uint3>()));
                                                    case 4: return _ctx->function_builder->constant(array_type, ConstantData::create(get_payload.operator()<uint4>()));
                                                    default: LUISA_ERROR_WITH_LOCATION("Vectors with length other than 2, 3 and 4 are not supported.");
                                                }
                                            }   
                                            case ir::Primitive::Float64: [[fallthrough]];
                                            case ir::Primitive::Int64: [[fallthrough]];
                                            case ir::Primitive::Uint64: [[fallthrough]];
                                            default: LUISA_ERROR_WITH_LOCATION("64-bit primitive types are not yet supported.");
                                        }
                                    }
                                    case ir::VectorElementType::Tag::Vector: LUISA_ERROR_WITH_LOCATION("Vector of vector is not supported.");
                                }
                            }
                            case ir::Type::Tag::Array: LUISA_ERROR_WITH_LOCATION("Array of arrays is not supported.");
                            case ir::Type::Tag::Struct: LUISA_ERROR_WITH_LOCATION("Array of structs is not supported.");
                            case ir::Type::Tag::Void: LUISA_ERROR_WITH_LOCATION("Array of void is invalid.");
                            default: LUISA_ERROR_WITH_LOCATION("Invalid array type.");
                        }
                    }
                    default: LUISA_ERROR_WITH_LOCATION("Invalid array type.");
                }
                break;
            }
            case ir::Const::Tag::Float64: [[fallthrough]];
            case ir::Const::Tag::Int64: [[fallthrough]];
            case ir::Const::Tag::Uint64: [[fallthrough]];
            default: LUISA_ERROR_WITH_LOCATION("64-bit primitive types are not yet supported.");
        }
        // 其余情况生成 literal
    }

    const Type *IR2AST::_convert_primitive_type(const ir::Primitive &type) noexcept {
        switch (type) {
            case ir::Primitive::Bool: return Type::from("bool");
            case ir::Primitive::Float32: return Type::from("float");
            case ir::Primitive::Int32: return Type::from("int");
            case ir::Primitive::Uint32: return Type::from("uint");
            case ir::Primitive::Float64: [[fallthrough]];
            case ir::Primitive::Int64: [[fallthrough]];
            case ir::Primitive::Uint64: LUISA_ERROR_WITH_LOCATION("64-bit primitive types are not yet supported.");
            default: LUISA_ERROR_WITH_LOCATION("Invalid primitive type.");
        };
    }

    const Type *IR2AST::_convert_type(const ir::Type *type) noexcept {
        switch (type->tag) {
            case ir::Type::Tag::Void: return nullptr;
            case ir::Type::Tag::Primitive: return _convert_primitive_type(type->primitive._0);
            case ir::Type::Tag::Vector: {
                switch (type->vector._0.element.tag) {
                    case ir::VectorElementType::Tag::Scalar: {
                        auto element_type = _convert_primitive_type(type->vector._0.element.scalar._0);
                        return Type::vector(element_type, type->vector._0.length);
                    }
                    case ir::VectorElementType::Tag::Vector: LUISA_ERROR_WITH_LOCATION("Vector of vectors is not supported.");
                }
            }
            case ir::Type::Tag::Matrix: return Type::matrix(type->matrix._0.dimension);
            case ir::Type::Tag::Array: {
                auto element_type = _convert_type(type->array._0.element.get());
                return Type::array(element_type, type->array._0.length);
            }
            case ir::Type::Tag::Struct: {
                auto struct_type = type->struct_._0;
                luisa::vector<const Type *> fields;
                fields.reserve(struct_type.fields.len);
                for (auto i = 0; i < struct_type.fields.len; i++) {
                    fields.push_back(_convert_type(struct_type.fields.ptr[i].get()));
                }
                return Type::structure(struct_type.alignment, fields);
            }
            default: break;
        }
        LUISA_ERROR_WITH_LOCATION("Invalid type.");
    }

    void IR2AST::_collect_phis(const ir::BasicBlock *bb) noexcept {
        _iterate(bb, [this](const ir::Node *node) noexcept {
            auto instr = node->instruction.get();
            switch (instr->tag) {
                case ir::Instruction::Tag::Phi: {
                    auto &&incomings = instr->phi._0;
                    for (auto i = 0u; i < incomings.len; i++) {
                        const auto &incoming = incomings.ptr[i];
                        auto src_block = incoming.block.get();
                        auto src_value = ir::luisa_compute_ir_node_get(incoming.value);
                        _ctx->block_to_phis[src_block].push_back(PhiAssignment{.dst=node, .src=src_value});
                    }
                    break;
                }
                case ir::Instruction::Tag::Loop: {
                    _collect_phis(instr->loop.body.get());
                    break;
                }
                case ir::Instruction::Tag::GenericLoop: {
                    _collect_phis(instr->generic_loop.prepare.get());
                    _collect_phis(instr->generic_loop.body.get());
                    _collect_phis(instr->generic_loop.update.get());
                    break;
                }
                case ir::Instruction::Tag::If: {
                    _collect_phis(instr->if_.true_branch.get());
                    _collect_phis(instr->if_.false_branch.get());
                    break;
                }
                case ir::Instruction::Tag::Switch: {
                    const auto &cases = instr->switch_.cases;
                    for (auto i = 0u; i < cases.len; i++) {
                        _collect_phis(cases.ptr[i].block.get());
                    }
                    _collect_phis(instr->switch_.default_.get());
                    break;
                }
                case ir::Instruction::Tag::AdScope: {
                    _collect_phis(instr->ad_scope.forward.get());
                    _collect_phis(instr->ad_scope.backward.get());
                    _collect_phis(instr->ad_scope.epilogue.get());
                    break;
                }
                case ir::Instruction::Tag::AdDetach: {
                    _collect_phis(instr->ad_detach._0.get());
                    break;
                }
                default: break;
            }
        });
    }

    [[nodiscard]] CallOp IR2AST::_decide_make_vector_op(const Type *primitive, size_t length) noexcept {
        LUISA_ASSERT(primitive->is_scalar(), "Only scalar types are allowed here.");
        switch (primitive->tag()) {
            case Type::Tag::BOOL:
                switch (length) {
                    case 2: return CallOp::MAKE_BOOL2;
                    case 3: return CallOp::MAKE_BOOL3;
                    case 4: return CallOp::MAKE_BOOL4;
                    default: LUISA_ERROR_WITH_LOCATION("Vectors with length other than 2, 3 and 4 are not supported.");
                }
            case Type::Tag::FLOAT32:
                switch (length) {
                    case 2: return CallOp::MAKE_FLOAT2;
                    case 3: return CallOp::MAKE_FLOAT3;
                    case 4: return CallOp::MAKE_FLOAT4;
                    default: LUISA_ERROR_WITH_LOCATION("Vectors with length other than 2, 3 and 4 are not supported.");
                }
            case Type::Tag::INT32:
                switch (length) {
                    case 2: return CallOp::MAKE_INT2;
                    case 3: return CallOp::MAKE_INT3;
                    case 4: return CallOp::MAKE_INT4;
                    default: LUISA_ERROR_WITH_LOCATION("Vectors with length other than 2, 3 and 4 are not supported.");
                }
            case Type::Tag::UINT32:
                switch (length) {
                    case 2: return CallOp::MAKE_UINT2;
                    case 3: return CallOp::MAKE_UINT3;
                    case 4: return CallOp::MAKE_UINT4;
                    default: LUISA_ERROR_WITH_LOCATION("Vectors with length other than 2, 3 and 4 are not supported.");
                }
            default: LUISA_ERROR_WITH_LOCATION("64-bit primitive types are not yet supported.");
        }
    }

    [[nodiscard]] CallOp IR2AST::_decide_make_matrix_op(size_t dimension) noexcept {
        switch (dimension) {
            case 2: return CallOp::MAKE_FLOAT2X2;
            case 3: return CallOp::MAKE_FLOAT3X3;
            case 4: return CallOp::MAKE_FLOAT4X4;
            default: LUISA_ERROR_WITH_LOCATION("Matrices with dimension other than 2, 3 and 4 are not supported.");
        }
    }

    [[nodiscard]] const RefExpr *IR2AST::_convert_argument(const ir::Node *node) noexcept {
        auto type = _convert_type(node->type_.get());
        switch (node->instruction->tag) {
            case ir::Instruction::Tag::Uniform: [[fallthrough]];
            case ir::Instruction::Tag::Argument: return _ctx->function_builder->argument(type);
            // Uniform is the argument of a kernel
            // while Argument is the argument of a callable
            case ir::Instruction::Tag::Texture2D: [[fallthrough]];
            case ir::Instruction::Tag::Texture3D: {
                auto dimension = node->instruction->tag == ir::Instruction::Tag::Texture2D ? 2u : 3u;
                auto texture_type = Type::texture(type, dimension);             
                return _ctx->function_builder->texture(texture_type);
            }
            case ir::Instruction::Tag::Buffer: {
                auto buffer_type = Type::buffer(type);
                return _ctx->function_builder->buffer(buffer_type);
            }
            case ir::Instruction::Tag::Bindless: return _ctx->function_builder->bindless_array();
            case ir::Instruction::Tag::Accel: return _ctx->function_builder->accel();
            default: LUISA_ERROR_WITH_LOCATION("Invalid argument type: {}.", (int)node->instruction->tag);
        }
    }

    [[nodiscard]] const RefExpr *IR2AST::_convert_captured(const ir::Capture &captured) noexcept {
        auto node = ir::luisa_compute_ir_node_get(captured.node);
        auto type = _convert_type(node->type_.get());
        switch (captured.binding.tag) {
            case ir::Binding::Tag::Buffer: {
                auto buffer_type = Type::buffer(type);
                auto &&[handle, offset_bytes, size_bytes] = captured.binding.buffer._0;
                return _ctx->function_builder->buffer_binding(buffer_type, handle, offset_bytes, size_bytes);
            }
            case ir::Binding::Tag::Accel: {
                auto handle = captured.binding.accel._0.handle;
                return _ctx->function_builder->accel_binding(handle);
            }
            case ir::Binding::Tag::BindlessArray: {
                auto handle = captured.binding.bindless_array._0.handle;
                return _ctx->function_builder->bindless_array_binding(handle);
            }
            case ir::Binding::Tag::Texture: {
                auto dimension = [&]() {
                    switch (node->instruction->tag) {
                        case ir::Instruction::Tag::Texture2D: return 2u;
                        case ir::Instruction::Tag::Texture3D: return 3u;
                        default: LUISA_ERROR_WITH_LOCATION("Binding tag inconsistent with instruction tag.");
                    }
                }();
                auto texture_type = Type::texture(type, dimension);
                auto &&[handle, level] = captured.binding.texture._0;
                return _ctx->function_builder->texture_binding(texture_type, handle, level);
            }
            default: LUISA_ERROR_WITH_LOCATION("Invalid binding tag.");
        }
    }

    [[nodiscard]] luisa::shared_ptr<detail::FunctionBuilder> IR2AST::convert_kernel(const ir::KernelModule *kernel) noexcept {
        IR2ASTContext ctx {
            .module = kernel->module,
            .function_builder = luisa::make_shared<detail::FunctionBuilder>(Function::Tag::KERNEL)
        };
        auto old_ctx = _ctx;
        _ctx = &ctx;

        detail::FunctionBuilder::push(_ctx->function_builder.get());
        _ctx->function_builder->push_scope(_ctx->function_builder->body());
        auto entry = kernel->module.entry.get();    
        _collect_phis(entry);
        
        auto captures = kernel->captures;
        auto args = kernel->args;
        for (auto i = 0; i < captures.len; i++) {
            auto captured = captures.ptr[i];
            auto node = ir::luisa_compute_ir_node_get(captured.node);    
            _ctx->node_to_exprs.emplace(node, _convert_captured(captured));
        }
        for (auto i = 0; i < args.len; i++) {
            auto arg = ir::luisa_compute_ir_node_get(args.ptr[i]);
            _ctx->node_to_exprs.emplace(arg, _convert_argument(arg));
        }
        auto arg_dispatch_size = _ctx->function_builder->argument(Type::from("vector<uint,3>"));
        // kernel always has a parameter dispatch_size: uint3

        auto shared = kernel->shared;
        for (auto i = 0; i < shared.len; i++) {
            auto shared_var = ir::luisa_compute_ir_node_get(shared.ptr[i]);
            auto type = _convert_type(shared_var->type_.get());
            auto shared_var_expr = _ctx->function_builder->shared(type);
            _ctx->node_to_exprs.emplace(shared_var, shared_var_expr);
        }

        auto dispatch_id = _ctx->function_builder->dispatch_id();
        auto dispatch_size = _ctx->function_builder->dispatch_size();
        auto comp = _ctx->function_builder->binary(
            Type::from("vector<bool,3>"),
            BinaryOp::GREATER_EQUAL,
            dispatch_id,
            dispatch_size
        );
        auto cond = _ctx->function_builder->call(
            Type::from("bool"),
            CallOp::ANY,
            {comp}
        );
        auto if_scope = _ctx->function_builder->if_(cond);
        _ctx->function_builder->push_scope(if_scope->true_branch());
        _ctx->function_builder->return_(nullptr);
        _ctx->function_builder->pop_scope(if_scope->true_branch());
        // if (lc_any(lc_dispatch_id() >= lc_dispatch_size())) return;
        
        _convert_block(entry);
        _ctx->function_builder->pop_scope(_ctx->function_builder->body());

        _ctx = old_ctx;
        return ctx.function_builder;
    }

    [[nodiscard]] luisa::shared_ptr<detail::FunctionBuilder> IR2AST::convert_callable(const ir::CallableModule *callable) noexcept {
        IR2ASTContext ctx {
            .module = callable->module,
            .function_builder = luisa::make_shared<detail::FunctionBuilder>(Function::Tag::CALLABLE),
            .zero_init = false
        };
        auto old_ctx = _ctx;
        _ctx = &ctx;

        for (auto i = 0; i < callable->args.len; i++) {
            auto arg = ir::luisa_compute_ir_node_get(callable->args.ptr[i]);
            _ctx->node_to_exprs.emplace(arg, _convert_argument(arg));
        }

        auto entry = callable->module.entry.get();
        _collect_phis(entry);
        _convert_block(entry);
        
        _ctx = old_ctx;
        return ctx.function_builder;
    }

    [[nodiscard]] luisa::shared_ptr<detail::FunctionBuilder> IR2AST::build(const ir::KernelModule *kernel) noexcept {
        IR2AST builder;
        return builder.convert_kernel(kernel);
    }

}