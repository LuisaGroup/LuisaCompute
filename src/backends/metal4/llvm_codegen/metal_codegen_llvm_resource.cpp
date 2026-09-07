#include "metal_codegen_llvm_impl.h"

namespace luisa::compute::metal::detail {

llvm::Value *MetalCodegenLLVMImpl::_translate_resource_query(IB &builder, FunctionContext &function, const xir::ResourceQueryInst *inst) noexcept {
    switch (inst->op()) {
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL: [[fallthrough]];
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY: [[fallthrough]];
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR: [[fallthrough]];
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR: {
            auto motion =
                inst->op() == xir::ResourceQueryOp::
                                  RAY_TRACING_QUERY_ALL_MOTION_BLUR ||
                inst->op() == xir::ResourceQueryOp::
                                  RAY_TRACING_QUERY_ANY_MOTION_BLUR;
            auto config = _air_ray_tracing_config(inst, motion);
            const xir::StoreInst *initialization{nullptr};
            for (auto use : inst->use_list()) {
                auto user = use->user();
                LUISA_ASSERT(
                    initialization == nullptr &&
                        user != nullptr && user->isa<xir::StoreInst>(),
                    "Metal AIR ray-query construction must have one initialization store.");
                initialization = static_cast<const xir::StoreInst *>(user);
                LUISA_ASSERT(initialization->value() == inst,
                             "Metal AIR ray-query result must be the value of its initialization store.");
            }
            LUISA_ASSERT(initialization != nullptr &&
                             initialization->variable()->isa<xir::AllocaInst>() &&
                             is_ray_query_type(initialization->variable()->type()),
                         "Metal AIR ray-query construction has no local query object.");
            auto query_object = initialization->variable();
            auto query = _value(builder, function, query_object);
            if (_pipeline_query_objects.contains(query_object)) {
                // The pipeline instruction consumes the constructor operands
                // directly and records the returned intersection value. Keep
                // the initialization store as an identity operation so the
                // XIR query object's address remains stable.
                return query;
            }
            LUISA_ASSERT(
                !motion,
                "Metal AIR stateful intersection_query cannot lower a motion "
                "query; it must be outlined to a ray-query pipeline first.");
            auto accel = _value(builder, function, inst->operand(0u));
            auto ray = _value(builder, function, inst->operand(1u));
            auto mask = _value(
                builder, function, inst->operand(motion ? 3u : 2u));
            auto handle_wrapper = builder.CreateExtractValue(accel, 0u);
            auto handle = builder.CreateExtractValue(handle_wrapper, 0u);
            auto vectorize_float3 = [&builder](llvm::Value *value) noexcept {
                if (value->getType()->isVectorTy()) { return value; }
                LUISA_ASSERT(
                    value->getType()->isArrayTy() &&
                        llvm::cast<llvm::ArrayType>(value->getType())
                                ->getNumElements() == 3u,
                    "Invalid Metal AIR ray vector storage type.");
                auto result = static_cast<llvm::Value *>(
                    llvm::PoisonValue::get(
                        llvm::FixedVectorType::get(
                            builder.getFloatTy(), 3u)));
                for (auto i = 0u; i < 3u; i++) {
                    result = builder.CreateInsertElement(
                        result, builder.CreateExtractValue(value, i), i);
                }
                return result;
            };
            auto origin = vectorize_float3(
                builder.CreateExtractValue(ray, 0u));
            auto t_min = builder.CreateExtractValue(ray, 1u);
            auto direction = vectorize_float3(
                builder.CreateExtractValue(ray, 2u));
            auto t_max = builder.CreateExtractValue(ray, 3u);
            mask = builder.CreateZExtOrTrunc(mask, builder.getInt32Ty());
            auto has_procedural_branch = false;
            for (auto use : query_object->use_list()) {
                auto user = use->user();
                if (user == nullptr) { continue; }
                if (user->isa<xir::RayQueryObjectReadInst>()) {
                    auto read = static_cast<const xir::RayQueryObjectReadInst *>(user);
                    has_procedural_branch |=
                        read->op() == xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_PROCEDURAL_CANDIDATE_HIT ||
                        read->op() == xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_PROCEDURAL_CANDIDATE;
                } else if (user->isa<xir::RayQueryObjectWriteInst>()) {
                    auto write = static_cast<const xir::RayQueryObjectWriteInst *>(user);
                    has_procedural_branch |=
                        write->op() == xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_PROCEDURAL;
                }
            }
            auto i32_zero = builder.getInt32(0u);
            llvm::SmallVector<llvm::Value *, 18u> arguments{
                query, origin, direction, t_min, t_max, handle, mask,
                i32_zero, i32_zero, i32_zero, i32_zero, i32_zero,
                builder.getInt32(
                    config.geometry_type(has_procedural_branch)),
                builder.getInt32(config.curve_basis), i32_zero,
                builder.getInt32(config.curve_control_point_count),
                builder.getInt1(false),
                builder.getInt1(
                    inst->op() ==
                            xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY ||
                    inst->op() == xir::ResourceQueryOp::
                                      RAY_TRACING_QUERY_ANY_MOTION_BLUR)};
            std::array extra_pointer_types{
                std::pair<unsigned, llvm::Type *>{
                    5u, _air_accel_handle()}};
            static_cast<void>(_air_ray_query_call(
                builder, "reset_intersection_query",
                builder.getVoidTy(), arguments, config, false,
                extra_pointer_types));
            return query;
        }
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_TRANSFORM: {
            auto accel = _value(builder, function, inst->operand(0u));
            auto index = _value(builder, function, inst->operand(1u));
            auto instance = _accel_instance_pointer(builder, accel, index);
            auto instance_type = _accel_instance();
            auto transform = builder.CreateStructGEP(
                instance_type, instance, accel_instance_transform_field);
            auto transform_type = llvm::cast<llvm::ArrayType>(
                instance_type->getElementType(accel_instance_transform_field));
            auto result = static_cast<llvm::Value *>(
                llvm::PoisonValue::get(_type(inst->type())->reg_type));
            for (auto column_index = 0u; column_index < 4u; column_index++) {
                auto column = static_cast<llvm::Value *>(
                    llvm::PoisonValue::get(
                        llvm::FixedVectorType::get(builder.getFloatTy(), 4u)));
                for (auto row_index = 0u; row_index < 3u; row_index++) {
                    auto element_index = column_index * 3u + row_index;
                    auto element_pointer = builder.CreateInBoundsGEP(
                        transform_type, transform,
                        {builder.getInt32(0u), builder.getInt32(element_index)});
                    auto element = builder.CreateAlignedLoad(
                        builder.getFloatTy(), element_pointer, llvm::Align{4u});
                    column = builder.CreateInsertElement(column, element, row_index);
                }
                column = builder.CreateInsertElement(
                    column,
                    llvm::ConstantFP::get(
                        builder.getFloatTy(), column_index == 3u ? 1.0 : 0.0),
                    3u);
                result = builder.CreateInsertValue(result, column, column_index);
            }
            return result;
        }
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_USER_ID: [[fallthrough]];
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_VISIBILITY_MASK: {
            auto accel = _value(builder, function, inst->operand(0u));
            auto index = _value(builder, function, inst->operand(1u));
            auto instance = _accel_instance_pointer(builder, accel, index);
            auto field = inst->op() == xir::ResourceQueryOp::RAY_TRACING_INSTANCE_USER_ID ?
                             accel_instance_user_id_field :
                             accel_instance_mask_field;
            auto pointer = builder.CreateStructGEP(
                _accel_instance(), instance, field);
            auto value = builder.CreateAlignedLoad(
                builder.getInt32Ty(), pointer, llvm::Align{4u});
            return builder.CreateZExtOrTrunc(
                value, _type(inst->type())->reg_type);
        }
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST: [[fallthrough]];
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY: [[fallthrough]];
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR: [[fallthrough]];
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR: {
            auto op = inst->op();
            auto motion =
                op == xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR ||
                op == xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR;
            auto any = op == xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY ||
                       op == xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR;
            auto config = _air_ray_tracing_config(inst, motion);
            auto time = motion ?
                            _value(builder, function, inst->operand(2u)) :
                            nullptr;
            auto mask_index = motion ? 3u : 2u;
            auto intersection = _air_trace(
                builder,
                _value(builder, function, inst->operand(0u)),
                _value(builder, function, inst->operand(1u)),
                _value(builder, function, inst->operand(mask_index)),
                time, config, any);
            auto intersection_type = builder.CreateExtractValue(
                intersection, 0u);
            auto hit = builder.CreateICmpNE(
                intersection_type, builder.getInt32(0u));
            if (any) { return hit; }

            auto result = static_cast<llvm::Value *>(
                llvm::PoisonValue::get(_type(inst->type())->reg_type));
            auto miss_id = builder.getInt32(~0u);
            auto instance_id = builder.CreateSelect(
                hit, builder.CreateExtractValue(intersection, 5u), miss_id);
            auto primitive_id = builder.CreateSelect(
                hit, builder.CreateExtractValue(intersection, 2u), miss_id);
            auto raw_barycentrics =
                builder.CreateExtractValue(intersection, 7u);
            if (config.curves) {
                auto curve_barycentrics = static_cast<llvm::Value *>(
                    llvm::Constant::getNullValue(
                        raw_barycentrics->getType()));
                curve_barycentrics = builder.CreateInsertElement(
                    curve_barycentrics,
                    builder.CreateExtractValue(intersection, 9u),
                    uint64_t{0u});
                curve_barycentrics = builder.CreateInsertElement(
                    curve_barycentrics,
                    llvm::ConstantFP::get(
                        builder.getFloatTy(), -1.0),
                    1u);
                auto is_curve = builder.CreateICmpEQ(
                    intersection_type, builder.getInt32(3u));
                raw_barycentrics = builder.CreateSelect(
                    is_curve, curve_barycentrics,
                    raw_barycentrics);
            }
            auto barycentrics = builder.CreateSelect(
                hit, raw_barycentrics,
                llvm::Constant::getNullValue(
                    llvm::FixedVectorType::get(builder.getFloatTy(), 2u)));
            auto distance = builder.CreateSelect(
                hit, builder.CreateExtractValue(intersection, 1u),
                llvm::ConstantFP::get(builder.getFloatTy(), 0.0));
            result = builder.CreateInsertValue(result, instance_id, 0u);
            result = builder.CreateInsertValue(result, primitive_id, 1u);
            result = builder.CreateInsertValue(result, barycentrics, 2u);
            result = builder.CreateInsertValue(result, distance, 3u);
            return result;
        }
        case xir::ResourceQueryOp::BUFFER_SIZE: {
            auto buffer = _value(builder, function, inst->operand(0u));
            auto size = builder.CreateExtractValue(buffer, 1u);
            size = builder.CreateUDiv(size, builder.getInt64(inst->operand(0u)->type()->element()->size()));
            return builder.CreateZExtOrTrunc(size, _type(inst->type())->reg_type);
        }
        case xir::ResourceQueryOp::BYTE_BUFFER_SIZE: {
            auto buffer = _value(builder, function, inst->operand(0u));
            auto size = builder.CreateExtractValue(buffer, 1u);
            return builder.CreateZExtOrTrunc(size, _type(inst->type())->reg_type);
        }
        case xir::ResourceQueryOp::BUFFER_DEVICE_ADDRESS: {
            auto buffer = _value(builder, function, inst->operand(0u));
            auto pointer = builder.CreateExtractValue(buffer, 0u);
            return builder.CreatePtrToInt(pointer, _type(inst->type())->reg_type);
        }
        case xir::ResourceQueryOp::BINDLESS_BUFFER_SIZE: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_BYTE_BUFFER_SIZE: {
            auto array = _value(builder, function, inst->operand(0u));
            auto index = _value(builder, function, inst->operand(1u));
            auto slot = _bindless_slot(builder, array, index);
            auto size = _bindless_buffer_size(builder, slot);
            if (inst->op() == xir::ResourceQueryOp::BINDLESS_BUFFER_SIZE) {
                auto stride = builder.CreateZExtOrTrunc(
                    _value(builder, function, inst->operand(2u)), builder.getInt64Ty());
                size = builder.CreateUDiv(size, stride);
            }
            return builder.CreateZExtOrTrunc(size, _type(inst->type())->reg_type);
        }
        case xir::ResourceQueryOp::BINDLESS_BUFFER_DEVICE_ADDRESS: {
            auto array = _value(builder, function, inst->operand(0u));
            auto index = _value(builder, function, inst->operand(1u));
            auto slot = _bindless_slot(builder, array, index);
            auto pointer = _bindless_slot_field(builder, slot, 0u);
            return builder.CreatePtrToInt(pointer, _type(inst->type())->reg_type);
        }
        case xir::ResourceQueryOp::TEXTURE2D_SIZE: [[fallthrough]];
        case xir::ResourceQueryOp::TEXTURE3D_SIZE: {
            auto texture_value = inst->operand(0u);
            auto texture_type = texture_value->type();
            auto texture = _value(builder, function, texture_value);
            auto dimension = texture_type->dimension();
            auto dimension_name = dimension == 2u ? llvm::StringRef{"2d"} : llvm::StringRef{"3d"};
            auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(_type(inst->type())->reg_type));
            constexpr std::array components{"width", "height", "depth"};
            for (auto i = 0u; i < dimension; i++) {
                auto function_name = "air.get_" + std::string{components[i]} +
                                     "_texture_" + dimension_name.str();
                auto function_type = llvm::FunctionType::get(
                    builder.getInt32Ty(), {texture->getType(), builder.getInt32Ty()}, false);
                auto callee = _module.getOrInsertFunction(function_name, function_type);
                if (auto f = llvm::dyn_cast<llvm::Function>(callee.getCallee())) {
                    std::array pointer_types{
                        std::pair<unsigned, llvm::Type *>{0u, _air_texture_handle(dimension)}};
                    _set_air_pointer_element_types(f, pointer_types);
                    f->setMustProgress();
                    f->setDoesNotFreeMemory();
                    f->setDoesNotThrow();
                    f->setWillReturn();
                    f->setOnlyReadsMemory();
                    f->setOnlyAccessesArgMemory();
                }
                auto component = builder.CreateCall(callee, {texture, builder.getInt32(0u)});
                result = builder.CreateInsertElement(result, component, i);
            }
            return result;
        }
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE_LEVEL: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE_LEVEL: {
            auto op = inst->op();
            auto is_3d = op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE ||
                         op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE_LEVEL;
            auto has_level = op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE_LEVEL ||
                             op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE_LEVEL;
            auto dimension = is_3d ? 3u : 2u;
            auto array = _value(builder, function, inst->operand(0u));
            auto index = _value(builder, function, inst->operand(1u));
            auto slot = _bindless_slot(builder, array, index);
            auto texture = _bindless_texture(builder, slot, dimension);
            auto level = has_level ?
                             builder.CreateZExtOrTrunc(
                                 _value(builder, function, inst->operand(2u)), builder.getInt32Ty()) :
                             builder.getInt32(0u);
            auto result = static_cast<llvm::Value *>(
                llvm::PoisonValue::get(_type(inst->type())->reg_type));
            constexpr std::array components{"width", "height", "depth"};
            for (auto i = 0u; i < dimension; i++) {
                auto function_name = "air.get_" + std::string{components[i]} +
                                     "_texture_" + std::to_string(dimension) + "d";
                auto function_type = llvm::FunctionType::get(
                    builder.getInt32Ty(), {texture->getType(), builder.getInt32Ty()}, false);
                auto callee = _module.getOrInsertFunction(function_name, function_type);
                if (auto f = llvm::dyn_cast<llvm::Function>(callee.getCallee())) {
                    std::array pointer_types{
                        std::pair<unsigned, llvm::Type *>{0u, _air_texture_handle(dimension)}};
                    _set_air_pointer_element_types(f, pointer_types);
                    f->setMustProgress();
                    f->setDoesNotFreeMemory();
                    f->setDoesNotThrow();
                    f->setWillReturn();
                    f->setOnlyReadsMemory();
                    f->setOnlyAccessesArgMemory();
                }
                auto component = builder.CreateCall(callee, {texture, level});
                result = builder.CreateInsertElement(result, component, i);
            }
            return result;
        }
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE: [[fallthrough]];
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE_LEVEL: [[fallthrough]];
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD: [[fallthrough]];
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD_LEVEL: [[fallthrough]];
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE: [[fallthrough]];
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE_LEVEL: [[fallthrough]];
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD: [[fallthrough]];
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD_LEVEL: {
            auto op = inst->op();
            auto is_3d = op == xir::ResourceQueryOp::TEXTURE3D_SAMPLE ||
                         op == xir::ResourceQueryOp::TEXTURE3D_SAMPLE_LEVEL ||
                         op == xir::ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD ||
                         op == xir::ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD_LEVEL;
            auto is_grad = op == xir::ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD ||
                           op == xir::ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD_LEVEL ||
                           op == xir::ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD ||
                           op == xir::ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD_LEVEL;
            auto has_level = op == xir::ResourceQueryOp::TEXTURE2D_SAMPLE_LEVEL ||
                             op == xir::ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD_LEVEL ||
                             op == xir::ResourceQueryOp::TEXTURE3D_SAMPLE_LEVEL ||
                             op == xir::ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD_LEVEL;
            auto texture = function.sampled_texture(inst->operand(0u));
            auto coordinate = _value(builder, function, inst->operand(1u));
            auto sampler_operand = is_grad ? (has_level ? 5u : 4u) : (has_level ? 3u : 2u);
            auto sampler = _air_sampler(
                builder, _value(builder, function, inst->operand(sampler_operand)),
                _value(builder, function, inst->operand(sampler_operand + 1u)));
            auto result_type = llvm::StructType::get(
                _context, {_type(inst->type())->reg_type, builder.getInt8Ty()});
            llvm::SmallVector<llvm::Value *, 10u> arguments{texture, sampler, coordinate};
            std::string function_name{"air.sample_texture_"};
            function_name.append(is_3d ? "3d" : "2d");
            if (is_grad) {
                function_name.append("_grad.v4f32");
                arguments.append({_value(builder, function, inst->operand(2u)),
                                  _value(builder, function, inst->operand(3u)),
                                  has_level ? _value(builder, function, inst->operand(4u)) :
                                              llvm::ConstantFP::get(builder.getFloatTy(), 0.0),
                                  builder.getInt1(true),
                                  llvm::Constant::getNullValue(
                                      llvm::FixedVectorType::get(builder.getInt32Ty(), is_3d ? 3u : 2u)),
                                  builder.getInt32(0u)});
            } else {
                function_name.append(".v4f32");
                arguments.append({builder.getInt1(true),
                                  llvm::Constant::getNullValue(
                                      llvm::FixedVectorType::get(builder.getInt32Ty(), is_3d ? 3u : 2u)),
                                  builder.getInt1(has_level),
                                  has_level ? _value(builder, function, inst->operand(2u)) :
                                              llvm::ConstantFP::get(builder.getFloatTy(), 0.0),
                                  llvm::ConstantFP::get(builder.getFloatTy(), 0.0),
                                  builder.getInt32(0u)});
            }
            llvm::SmallVector<llvm::Type *, 10u> parameter_types;
            for (auto argument : arguments) { parameter_types.emplace_back(argument->getType()); }
            auto function_type = llvm::FunctionType::get(result_type, parameter_types, false);
            auto callee = _module.getOrInsertFunction(function_name, function_type);
            if (auto f = llvm::dyn_cast<llvm::Function>(callee.getCallee())) {
                auto dimension = is_3d ? 3u : 2u;
                std::array pointer_types{
                    std::pair<unsigned, llvm::Type *>{0u, _air_texture_handle(dimension)},
                    std::pair<unsigned, llvm::Type *>{1u, _air_sampler_handle()}};
                _set_air_pointer_element_types(f, pointer_types);
                f->setMustProgress();
                f->setDoesNotFreeMemory();
                f->setDoesNotThrow();
                f->setWillReturn();
                f->setOnlyReadsMemory();
                f->setOnlyAccessesArgMemory();
                if (!is_grad) { f->addFnAttr(llvm::Attribute::Convergent); }
            }
            auto call = builder.CreateCall(callee, arguments);
            if (!is_grad) { call->setConvergent(); }
            return builder.CreateExtractValue(call, 0u);
        }
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER: [[fallthrough]];
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER: {
            auto op = inst->op();
            auto is_3d = op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE ||
                         op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL ||
                         op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD ||
                         op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL ||
                         op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER ||
                         op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER ||
                         op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER ||
                         op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER;
            auto is_grad = op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD ||
                           op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL ||
                           op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD ||
                           op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL ||
                           op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER ||
                           op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER ||
                           op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER ||
                           op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER;
            auto has_level = op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL ||
                             op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL ||
                             op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL ||
                             op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL ||
                             op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER ||
                             op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER ||
                             op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER ||
                             op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER;
            auto explicit_sampler = op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER ||
                                    op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER ||
                                    op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER ||
                                    op == xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER ||
                                    op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER ||
                                    op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER ||
                                    op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER ||
                                    op == xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER;
            auto dimension = is_3d ? 3u : 2u;
            auto array = _value(builder, function, inst->operand(0u));
            auto index = _value(builder, function, inst->operand(1u));
            auto slot = _bindless_slot(builder, array, index);
            auto texture = _bindless_texture(builder, slot, dimension);
            auto sampler = explicit_sampler ?
                               _air_sampler(
                                   builder,
                                   _value(builder, function, inst->operand(inst->operand_count() - 2u)),
                                   _value(builder, function, inst->operand(inst->operand_count() - 1u))) :
                               _air_sampler_code(
                                   builder, _bindless_sampler_code(builder, slot, dimension));
            auto coordinate = _value(builder, function, inst->operand(2u));
            auto result_type = llvm::StructType::get(
                _context, {_type(inst->type())->reg_type, builder.getInt8Ty()});
            llvm::SmallVector<llvm::Value *, 10u> arguments{texture, sampler, coordinate};
            auto function_name = "air.sample_texture_" + std::to_string(dimension) + "d";
            if (is_grad) {
                function_name.append("_grad.v4f32");
                arguments.append({_value(builder, function, inst->operand(3u)),
                                  _value(builder, function, inst->operand(4u)),
                                  has_level ? _value(builder, function, inst->operand(5u)) :
                                              llvm::ConstantFP::get(builder.getFloatTy(), 0.0),
                                  builder.getInt1(true),
                                  llvm::Constant::getNullValue(
                                      llvm::FixedVectorType::get(builder.getInt32Ty(), dimension)),
                                  builder.getInt32(0u)});
            } else {
                function_name.append(".v4f32");
                arguments.append({builder.getInt1(true),
                                  llvm::Constant::getNullValue(
                                      llvm::FixedVectorType::get(builder.getInt32Ty(), dimension)),
                                  builder.getInt1(has_level),
                                  has_level ? _value(builder, function, inst->operand(3u)) :
                                              llvm::ConstantFP::get(builder.getFloatTy(), 0.0),
                                  llvm::ConstantFP::get(builder.getFloatTy(), 0.0),
                                  builder.getInt32(0u)});
            }
            llvm::SmallVector<llvm::Type *, 10u> parameter_types;
            for (auto argument : arguments) { parameter_types.emplace_back(argument->getType()); }
            auto function_type = llvm::FunctionType::get(result_type, parameter_types, false);
            auto callee = _module.getOrInsertFunction(function_name, function_type);
            if (auto f = llvm::dyn_cast<llvm::Function>(callee.getCallee())) {
                std::array pointer_types{
                    std::pair<unsigned, llvm::Type *>{0u, _air_texture_handle(dimension)},
                    std::pair<unsigned, llvm::Type *>{1u, _air_sampler_handle()}};
                _set_air_pointer_element_types(f, pointer_types);
                f->setMustProgress();
                f->setDoesNotFreeMemory();
                f->setDoesNotThrow();
                f->setWillReturn();
                f->setOnlyReadsMemory();
                f->setOnlyAccessesArgMemory();
                if (!is_grad) { f->addFnAttr(llvm::Attribute::Convergent); }
            }
            auto call = builder.CreateCall(callee, arguments);
            if (!is_grad) { call->setConvergent(); }
            return builder.CreateExtractValue(call, 0u);
        }
        default: _unsupported_instruction(inst);
    }
}

llvm::Value *MetalCodegenLLVMImpl::_translate_ray_query_object_read(
    IB &builder, FunctionContext &function,
    const xir::RayQueryObjectReadInst *inst) noexcept {
    LUISA_ASSERT(inst->operand_count() == 1u,
                 "Metal AIR ray-query reads require one query object.");
    if (function.pipeline_handler != nullptr ||
        function.pipeline_query_results.contains(inst->operand(0u))) {
        return _translate_pipeline_ray_query_read(
            builder, function, inst);
    }
    auto config = _air_ray_tracing_config(inst->operand(0u));
    auto query = _value(builder, function, inst->operand(0u));
    auto getter = [&](luisa::string_view operation,
                      llvm::Type *return_type) noexcept {
        llvm::SmallVector<llvm::Value *, 1u> arguments{query};
        return static_cast<llvm::Value *>(_air_ray_query_call(
            builder, operation, return_type, arguments,
            config, true));
    };
    auto i32 = builder.getInt32Ty();
    auto f32 = builder.getFloatTy();
    auto f32x2 = llvm::FixedVectorType::get(f32, 2u);
    auto f32x3 = llvm::FixedVectorType::get(f32, 3u);
    auto make_ray = [&](luisa::string_view origin_operation,
                        luisa::string_view direction_operation) noexcept {
        auto origin = getter(origin_operation, f32x3);
        auto direction = getter(direction_operation, f32x3);
        auto t_min = getter(
            "get_ray_min_distance_intersection_query", f32);
        auto t_max = getter(
            "get_committed_distance_intersection_query", f32);
        auto result_type = llvm::cast<llvm::StructType>(
            _type(inst->type())->reg_type);
        auto vector_to_storage = [&builder](
                                     llvm::Value *value,
                                     llvm::Type *storage_type) noexcept {
            if (storage_type->isVectorTy()) { return value; }
            LUISA_ASSERT(
                storage_type->isArrayTy() &&
                    llvm::cast<llvm::ArrayType>(storage_type)
                            ->getNumElements() == 3u,
                "Invalid Metal AIR ray float3 storage type.");
            auto storage = static_cast<llvm::Value *>(
                llvm::PoisonValue::get(storage_type));
            for (auto i = 0u; i < 3u; i++) {
                storage = builder.CreateInsertValue(
                    storage,
                    builder.CreateExtractElement(value, i), i);
            }
            return storage;
        };
        auto result = static_cast<llvm::Value *>(
            llvm::PoisonValue::get(result_type));
        result = builder.CreateInsertValue(
            result,
            vector_to_storage(
                origin, result_type->getElementType(0u)),
            0u);
        result = builder.CreateInsertValue(result, t_min, 1u);
        result = builder.CreateInsertValue(
            result,
            vector_to_storage(
                direction, result_type->getElementType(2u)),
            2u);
        result = builder.CreateInsertValue(result, t_max, 3u);
        return result;
    };
    switch (inst->op()) {
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED: {
            auto iter = function.last_query_proceed.find(query);
            LUISA_ASSERT(
                iter != function.last_query_proceed.end(),
                "Metal AIR ray-query termination was read before proceed.");
            return builder.CreateNot(iter->second);
        }
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TRIANGLE_CANDIDATE: [[fallthrough]];
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_PROCEDURAL_CANDIDATE: {
            auto type = getter(
                "get_candidate_intersection_type_intersection_query",
                i32);
            auto expected =
                inst->op() == xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TRIANGLE_CANDIDATE ?
                    1u :
                    2u;
            auto matches = builder.CreateICmpEQ(
                type, builder.getInt32(expected));
            if (config.curves &&
                inst->op() == xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TRIANGLE_CANDIDATE) {
                matches = builder.CreateOr(
                    matches,
                    builder.CreateICmpEQ(
                        type, builder.getInt32(3u)));
            }
            return matches;
        }
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_WORLD_SPACE_RAY:
            return make_ray(
                "get_world_space_ray_origin_intersection_query",
                "get_world_space_ray_direction_intersection_query");
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_CANDIDATE_OBJECT_SPACE_RAY:
            return make_ray(
                "get_candidate_ray_origin_intersection_query",
                "get_candidate_ray_direction_intersection_query");
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_PROCEDURAL_CANDIDATE_HIT: {
            auto instance = getter(
                "get_candidate_instance_id_intersection_query", i32);
            auto primitive = getter(
                "get_candidate_primitive_id_intersection_query", i32);
            auto result = static_cast<llvm::Value *>(
                llvm::PoisonValue::get(_type(inst->type())->reg_type));
            result = builder.CreateInsertValue(result, instance, 0u);
            result = builder.CreateInsertValue(result, primitive, 1u);
            return result;
        }
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_TRIANGLE_CANDIDATE_HIT: {
            auto instance = getter(
                "get_candidate_instance_id_intersection_query", i32);
            auto primitive = getter(
                "get_candidate_primitive_id_intersection_query", i32);
            auto barycentrics = getter(
                "get_candidate_triangle_barycentric_coord_intersection_query",
                f32x2);
            auto distance = getter(
                "get_candidate_triangle_distance_intersection_query", f32);
            if (config.curves) {
                auto type = getter(
                    "get_candidate_intersection_type_intersection_query",
                    i32);
                auto is_curve = builder.CreateICmpEQ(
                    type, builder.getInt32(3u));
                auto curve_barycentrics = static_cast<llvm::Value *>(
                    llvm::Constant::getNullValue(f32x2));
                curve_barycentrics = builder.CreateInsertElement(
                    curve_barycentrics,
                    getter(
                        "get_candidate_curve_parameter_intersection_query",
                        f32),
                    uint64_t{0u});
                curve_barycentrics = builder.CreateInsertElement(
                    curve_barycentrics,
                    llvm::ConstantFP::get(f32, -1.0), 1u);
                barycentrics = builder.CreateSelect(
                    is_curve, curve_barycentrics, barycentrics);
                distance = builder.CreateSelect(
                    is_curve,
                    getter(
                        "get_candidate_curve_distance_intersection_query",
                        f32),
                    distance);
            }
            auto result = static_cast<llvm::Value *>(
                llvm::PoisonValue::get(_type(inst->type())->reg_type));
            result = builder.CreateInsertValue(result, instance, 0u);
            result = builder.CreateInsertValue(result, primitive, 1u);
            result = builder.CreateInsertValue(result, barycentrics, 2u);
            result = builder.CreateInsertValue(result, distance, 3u);
            return result;
        }
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_COMMITTED_HIT: {
            auto intersection_type = getter(
                "get_committed_intersection_type_intersection_query", i32);
            auto is_none = builder.CreateICmpEQ(
                intersection_type, builder.getInt32(0u));
            auto is_triangle = builder.CreateICmpEQ(
                intersection_type, builder.getInt32(1u));
            auto is_curve = builder.CreateICmpEQ(
                intersection_type, builder.getInt32(3u));
            auto is_surface = config.curves ?
                                  builder.CreateOr(
                                      is_triangle, is_curve) :
                                  is_triangle;
            auto kind = builder.CreateSelect(
                is_none, builder.getInt32(0u),
                builder.CreateSelect(
                    is_surface, builder.getInt32(1u),
                    builder.getInt32(2u)));
            auto raw_instance = getter(
                "get_committed_instance_id_intersection_query", i32);
            auto instance = builder.CreateSelect(
                is_none, builder.getInt32(~0u), raw_instance);
            auto primitive = getter(
                "get_committed_primitive_id_intersection_query", i32);
            auto raw_barycentrics = getter(
                "get_committed_triangle_barycentric_coord_intersection_query",
                f32x2);
            auto barycentrics = raw_barycentrics;
            if (config.curves) {
                auto curve_barycentrics = static_cast<llvm::Value *>(
                    llvm::Constant::getNullValue(f32x2));
                curve_barycentrics = builder.CreateInsertElement(
                    curve_barycentrics,
                    getter(
                        "get_committed_curve_parameter_intersection_query",
                        f32),
                    uint64_t{0u});
                curve_barycentrics = builder.CreateInsertElement(
                    curve_barycentrics,
                    llvm::ConstantFP::get(f32, -1.0), 1u);
                barycentrics = builder.CreateSelect(
                    is_curve, curve_barycentrics, barycentrics);
            }
            barycentrics = builder.CreateSelect(
                is_surface, barycentrics,
                llvm::Constant::getNullValue(f32x2));
            auto distance = getter(
                "get_committed_distance_intersection_query", f32);
            auto result = static_cast<llvm::Value *>(
                llvm::PoisonValue::get(_type(inst->type())->reg_type));
            result = builder.CreateInsertValue(result, instance, 0u);
            result = builder.CreateInsertValue(result, primitive, 1u);
            result = builder.CreateInsertValue(result, barycentrics, 2u);
            result = builder.CreateInsertValue(result, kind, 3u);
            result = builder.CreateInsertValue(result, distance, 4u);
            return result;
        }
    }
    _unsupported_instruction(inst);
}

void MetalCodegenLLVMImpl::_translate_ray_query_object_write(
    IB &builder, FunctionContext &function,
    const xir::RayQueryObjectWriteInst *inst) noexcept {
    if (function.pipeline_handler != nullptr) {
        _translate_pipeline_ray_query_write(
            builder, function, inst);
        return;
    }
    LUISA_ASSERT(inst->operand_count() >= 1u,
                 "Metal AIR ray-query writes require a query object.");
    auto config = _air_ray_tracing_config(inst->operand(0u));
    auto query = _value(builder, function, inst->operand(0u));
    auto call = [&](luisa::string_view operation,
                    llvm::ArrayRef<llvm::Value *> extra_arguments = {}) noexcept {
        llvm::SmallVector<llvm::Value *, 2u> arguments{query};
        arguments.append(extra_arguments.begin(), extra_arguments.end());
        return _air_ray_query_call(
            builder, operation, builder.getVoidTy(), arguments,
            config);
    };
    switch (inst->op()) {
        case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED: {
            llvm::SmallVector<llvm::Value *, 1u> arguments{query};
            auto next = _air_ray_query_call(
                builder, "next_intersection_query",
                builder.getInt1Ty(), arguments, config);
            function.last_query_proceed[query] = next;
            return;
        }
        case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_TRIANGLE: {
            if (!config.curves) {
                static_cast<void>(call(
                    "commit_triangle_intersection_intersection_query"));
                return;
            }
            llvm::SmallVector<llvm::Value *, 1u> arguments{query};
            auto candidate_type = _air_ray_query_call(
                builder,
                "get_candidate_intersection_type_intersection_query",
                builder.getInt32Ty(), arguments, config, true);
            auto curve_block = llvm::BasicBlock::Create(
                _context, "ray.query.commit.curve",
                function.function);
            auto triangle_block = llvm::BasicBlock::Create(
                _context, "ray.query.commit.triangle",
                function.function);
            auto merge_block = llvm::BasicBlock::Create(
                _context, "ray.query.commit.surface.merge",
                function.function);
            builder.CreateCondBr(
                builder.CreateICmpEQ(
                    candidate_type, builder.getInt32(3u)),
                curve_block, triangle_block);
            builder.SetInsertPoint(curve_block);
            static_cast<void>(call(
                "commit_curve_intersection_intersection_query"));
            builder.CreateBr(merge_block);
            builder.SetInsertPoint(triangle_block);
            static_cast<void>(call(
                "commit_triangle_intersection_intersection_query"));
            builder.CreateBr(merge_block);
            builder.SetInsertPoint(merge_block);
            return;
        }
        case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_TERMINATE:
            static_cast<void>(call("abort_intersection_query"));
            return;
        case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_PROCEDURAL: {
            LUISA_ASSERT(inst->operand_count() == 2u,
                         "Metal AIR procedural ray-query commit requires a distance.");
            auto distance = _value(builder, function, inst->operand(1u));
            auto getter = [&](luisa::string_view operation) noexcept {
                llvm::SmallVector<llvm::Value *, 1u> arguments{query};
                return static_cast<llvm::Value *>(_air_ray_query_call(
                    builder, operation, builder.getFloatTy(),
                    arguments, config, true));
            };
            auto t_min = getter(
                "get_ray_min_distance_intersection_query");
            auto t_max = getter(
                "get_committed_distance_intersection_query");
            auto valid = builder.CreateAnd(
                builder.CreateFCmpOGE(distance, t_min),
                builder.CreateFCmpOLE(distance, t_max));
            auto commit_block = llvm::BasicBlock::Create(
                _context, "ray.query.commit.procedural",
                function.function);
            auto merge_block = llvm::BasicBlock::Create(
                _context, "ray.query.commit.merge",
                function.function);
            builder.CreateCondBr(valid, commit_block, merge_block);
            builder.SetInsertPoint(commit_block);
            llvm::SmallVector<llvm::Value *, 1u> extra{
                distance};
            static_cast<void>(call(
                "commit_bounding_box_intersection_intersection_query",
                extra));
            builder.CreateBr(merge_block);
            builder.SetInsertPoint(merge_block);
            return;
        }
    }
    _unsupported_instruction(inst);
}

llvm::Value *MetalCodegenLLVMImpl::_translate_resource_read(IB &builder, FunctionContext &function, const xir::ResourceReadInst *inst) noexcept {
    auto load_cooperative_vector = [&](llvm::Value *pointer,
                                       const Type *vector_type) noexcept {
        LUISA_ASSERT(vector_type != nullptr &&
                         vector_type->is_cooperative_vector(),
                     "Metal cooperative load result is not a cooperative vector.");
        auto element_type = vector_type->element();
        auto result = static_cast<llvm::Value *>(
            llvm::PoisonValue::get(_type(vector_type)->reg_type));
        for (auto i = 0u; i < vector_type->dimension(); i++) {
            auto element_pointer = i == 0u ?
                                       pointer :
                                       builder.CreateInBoundsGEP(
                                           _type(element_type)->mem_type,
                                           pointer, builder.getInt32(i));
            result = builder.CreateInsertValue(
                result, _load(builder, element_pointer, element_type), i);
        }
        return result;
    };
    switch (inst->op()) {
        case xir::ResourceReadOp::BUFFER_READ: [[fallthrough]];
        case xir::ResourceReadOp::BUFFER_VOLATILE_READ: {
            auto is_volatile = inst->op() == xir::ResourceReadOp::BUFFER_VOLATILE_READ;
            auto buffer = _value(builder, function, inst->operand(0u));
            auto index = _value(builder, function, inst->operand(1u));
            auto pointer = _buffer_pointer(builder, buffer, index, inst->type()->size());
            if (is_volatile) { _air_atomic_fence(builder, 1u); }
            return _load(builder, pointer, inst->type(), is_volatile);
        }
        case xir::ResourceReadOp::BYTE_BUFFER_READ: [[fallthrough]];
        case xir::ResourceReadOp::BYTE_BUFFER_VOLATILE_READ: {
            auto is_volatile = inst->op() == xir::ResourceReadOp::BYTE_BUFFER_VOLATILE_READ;
            auto buffer = _value(builder, function, inst->operand(0u));
            auto index = _value(builder, function, inst->operand(1u));
            auto pointer = _buffer_pointer(builder, buffer, index, 1u);
            if (is_volatile) { _air_atomic_fence(builder, 1u); }
            return _load(builder, pointer, inst->type(), is_volatile);
        }
        case xir::ResourceReadOp::BINDLESS_BUFFER_READ: [[fallthrough]];
        case xir::ResourceReadOp::BINDLESS_BYTE_BUFFER_READ: {
            auto array = _value(builder, function, inst->operand(0u));
            auto slot_index = _value(builder, function, inst->operand(1u));
            auto element_index = _value(builder, function, inst->operand(2u));
            auto slot = _bindless_slot(builder, array, slot_index);
            auto buffer = _bindless_slot_field(builder, slot, 0u);
            auto stride = inst->op() == xir::ResourceReadOp::BINDLESS_BUFFER_READ ?
                              inst->type()->size() :
                              1u;
            auto pointer = _device_pointer_offset(builder, buffer, element_index, stride);
            return _load(builder, pointer, inst->type());
        }
        case xir::ResourceReadOp::DEVICE_ADDRESS_READ: {
            auto address = builder.CreateZExtOrTrunc(_value(builder, function, inst->operand(0u)), builder.getInt64Ty());
            auto pointer = builder.CreateIntToPtr(address, llvm::PointerType::get(_context, air_address_space_device));
            return _load(builder, pointer, inst->type());
        }
        case xir::ResourceReadOp::TEXTURE2D_READ: [[fallthrough]];
        case xir::ResourceReadOp::TEXTURE3D_READ: {
            auto texture_value = inst->operand(0u);
            auto texture_type = texture_value->type();
            auto texture = _value(builder, function, texture_value);
            auto coordinate = _value(builder, function, inst->operand(1u));
            auto dimension = texture_type->dimension();
            auto element = texture_type->element();
            auto suffix = element->is_float32() ? std::string_view{"v4f32"} :
                          element->is_int32()   ? std::string_view{"s.v4i32"} :
                                                  std::string_view{"u.v4i32"};
            auto function_name = "air.read_texture_" + std::to_string(dimension) + "d." + std::string{suffix};
            auto sampler_type = llvm::PointerType::get(_context, air_address_space_constant);
            auto sampler_function_type = llvm::FunctionType::get(sampler_type, {}, false);
            auto sampler_callee = _module.getOrInsertFunction("air.get_read_sampler", sampler_function_type);
            if (auto f = llvm::dyn_cast<llvm::Function>(sampler_callee.getCallee())) {
                _set_air_pointer_element_types(f, {}, _air_sampler_handle());
                f->setMustProgress();
                f->setDoesNotFreeMemory();
                f->setDoesNotThrow();
                f->setWillReturn();
                f->setOnlyReadsMemory();
                f->setOnlyAccessesInaccessibleMemory();
            }
            auto sampler = builder.CreateCall(sampler_callee);
            auto result_type = llvm::StructType::get(
                _context, {_type(inst->type())->reg_type, builder.getInt8Ty()});
            auto function_type = llvm::FunctionType::get(
                result_type,
                {texture->getType(), sampler_type, coordinate->getType(),
                 coordinate->getType(), builder.getInt32Ty(), builder.getInt32Ty()},
                false);
            auto callee = _module.getOrInsertFunction(function_name, function_type);
            if (auto f = llvm::dyn_cast<llvm::Function>(callee.getCallee())) {
                std::array pointer_types{
                    std::pair<unsigned, llvm::Type *>{0u, _air_texture_handle(dimension)},
                    std::pair<unsigned, llvm::Type *>{1u, _air_sampler_handle()}};
                _set_air_pointer_element_types(f, pointer_types);
                f->setMustProgress();
                f->setDoesNotFreeMemory();
                f->setDoesNotThrow();
                f->setWillReturn();
                f->setOnlyReadsMemory();
                f->setOnlyAccessesArgMemory();
            }
            auto result = builder.CreateCall(
                callee,
                {texture, sampler, coordinate,
                 llvm::Constant::getNullValue(coordinate->getType()),
                 builder.getInt32(0u), builder.getInt32(_texture_access(texture_value))});
            return builder.CreateExtractValue(result, 0u);
        }
        case xir::ResourceReadOp::BINDLESS_TEXTURE2D_READ: [[fallthrough]];
        case xir::ResourceReadOp::BINDLESS_TEXTURE3D_READ: [[fallthrough]];
        case xir::ResourceReadOp::BINDLESS_TEXTURE2D_READ_LEVEL: [[fallthrough]];
        case xir::ResourceReadOp::BINDLESS_TEXTURE3D_READ_LEVEL: {
            auto op = inst->op();
            auto is_3d = op == xir::ResourceReadOp::BINDLESS_TEXTURE3D_READ ||
                         op == xir::ResourceReadOp::BINDLESS_TEXTURE3D_READ_LEVEL;
            auto has_level = op == xir::ResourceReadOp::BINDLESS_TEXTURE2D_READ_LEVEL ||
                             op == xir::ResourceReadOp::BINDLESS_TEXTURE3D_READ_LEVEL;
            auto dimension = is_3d ? 3u : 2u;
            auto array = _value(builder, function, inst->operand(0u));
            auto slot_index = _value(builder, function, inst->operand(1u));
            auto coordinate = _value(builder, function, inst->operand(2u));
            auto level = has_level ?
                             builder.CreateZExtOrTrunc(
                                 _value(builder, function, inst->operand(3u)), builder.getInt32Ty()) :
                             builder.getInt32(0u);
            auto slot = _bindless_slot(builder, array, slot_index);
            auto texture = _bindless_texture(builder, slot, dimension);
            auto sampler_type = llvm::PointerType::get(_context, air_address_space_constant);
            auto sampler_function_type = llvm::FunctionType::get(sampler_type, {}, false);
            auto sampler_callee = _module.getOrInsertFunction(
                "air.get_read_sampler", sampler_function_type);
            if (auto f = llvm::dyn_cast<llvm::Function>(sampler_callee.getCallee())) {
                _set_air_pointer_element_types(f, {}, _air_sampler_handle());
                f->setMustProgress();
                f->setDoesNotFreeMemory();
                f->setDoesNotThrow();
                f->setWillReturn();
                f->setOnlyReadsMemory();
                f->setOnlyAccessesInaccessibleMemory();
            }
            auto sampler = builder.CreateCall(sampler_callee);
            auto result_type = llvm::StructType::get(
                _context, {_type(inst->type())->reg_type, builder.getInt8Ty()});
            auto function_name = "air.read_texture_" + std::to_string(dimension) + "d.v4f32";
            auto function_type = llvm::FunctionType::get(
                result_type,
                {texture->getType(), sampler_type, coordinate->getType(),
                 coordinate->getType(), builder.getInt32Ty(), builder.getInt32Ty()},
                false);
            auto callee = _module.getOrInsertFunction(function_name, function_type);
            if (auto f = llvm::dyn_cast<llvm::Function>(callee.getCallee())) {
                std::array pointer_types{
                    std::pair<unsigned, llvm::Type *>{0u, _air_texture_handle(dimension)},
                    std::pair<unsigned, llvm::Type *>{1u, _air_sampler_handle()}};
                _set_air_pointer_element_types(f, pointer_types);
                f->setMustProgress();
                f->setDoesNotFreeMemory();
                f->setDoesNotThrow();
                f->setWillReturn();
                f->setOnlyReadsMemory();
                f->setOnlyAccessesArgMemory();
            }
            auto result = builder.CreateCall(
                callee,
                {texture, sampler, coordinate,
                 llvm::Constant::getNullValue(coordinate->getType()),
                 level, builder.getInt32(0u)});
            return builder.CreateExtractValue(result, 0u);
        }
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_LOAD: {
            auto buffer = _value(builder, function, inst->operand(0u));
            auto offset = _value(builder, function, inst->operand(1u));
            auto pointer = _buffer_pointer(builder, buffer, offset, 1u);
            return load_cooperative_vector(pointer, inst->type());
        }
        case xir::ResourceReadOp::BINDLESS_COOPERATIVE_VECTOR_LOAD: {
            auto array = _value(builder, function, inst->operand(0u));
            auto slot_index = _value(builder, function, inst->operand(1u));
            auto offset = _value(builder, function, inst->operand(2u));
            auto slot = _bindless_slot(builder, array, slot_index);
            auto buffer = _bindless_slot_field(builder, slot, 0u);
            auto pointer = _device_pointer_offset(
                builder, buffer, offset, 1u);
            return load_cooperative_vector(pointer, inst->type());
        }
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_WORKGROUP_LOAD: {
            auto shared = _value(builder, function, inst->operand(0u));
            auto index = _value(builder, function, inst->operand(1u));
            auto element_type = inst->type()->element();
            auto pointer = builder.CreateInBoundsGEP(
                _type(element_type)->mem_type, shared, index);
            return load_cooperative_vector(pointer, inst->type());
        }
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_SPLAT: {
            auto scalar = _value(builder, function, inst->operand(0u));
            auto result = static_cast<llvm::Value *>(
                llvm::PoisonValue::get(_type(inst->type())->reg_type));
            for (auto i = 0u; i < inst->type()->dimension(); i++) {
                result = builder.CreateInsertValue(result, scalar, i);
            }
            return result;
        }
        case xir::ResourceReadOp::COOPERATIVE_VECTOR_CAST: {
            auto source_type = inst->operand(0u)->type();
            auto target_type = inst->type();
            auto source = _value(builder, function, inst->operand(0u));
            LUISA_ASSERT(source_type->is_cooperative_vector() &&
                             target_type->is_cooperative_vector() &&
                             source_type->dimension() == target_type->dimension(),
                         "Invalid Metal cooperative-vector cast.");
            auto result = static_cast<llvm::Value *>(
                llvm::PoisonValue::get(_type(target_type)->reg_type));
            for (auto i = 0u; i < target_type->dimension(); i++) {
                auto source_element = builder.CreateExtractValue(source, i);
                auto target_element = _static_cast(
                    builder, source_element, source_type->element(),
                    target_type->element());
                result = builder.CreateInsertValue(result, target_element, i);
            }
            return result;
        }
        default: _unsupported_instruction(inst);
    }
}

void MetalCodegenLLVMImpl::_translate_resource_write(IB &builder, FunctionContext &function, const xir::ResourceWriteInst *inst) noexcept {
    auto store_cooperative_vector = [&](llvm::Value *pointer,
                                        llvm::Value *value,
                                        const Type *vector_type) noexcept {
        LUISA_ASSERT(vector_type != nullptr &&
                         vector_type->is_cooperative_vector() &&
                         value->getType() == _type(vector_type)->reg_type,
                     "Invalid Metal cooperative-vector store value.");
        auto element_type = vector_type->element();
        for (auto i = 0u; i < vector_type->dimension(); i++) {
            auto element_pointer = i == 0u ?
                                       pointer :
                                       builder.CreateInBoundsGEP(
                                           _type(element_type)->mem_type,
                                           pointer, builder.getInt32(i));
            _store(builder, element_pointer,
                   builder.CreateExtractValue(value, i), element_type);
        }
    };
    auto indirect_dispatch_buffer = [&]() noexcept {
        auto value = _value(builder, function, inst->operand(0u));
        if (value->getType()->isPointerTy()) {
            value = builder.CreateAlignedLoad(
                _indirect_dispatch_buffer(), value,
                llvm::Align{kernel_argument_alignment});
        }
        LUISA_ASSERT(
            value->getType() == _indirect_dispatch_buffer(),
            "Invalid Metal indirect-dispatch buffer LLVM value.");
        return value;
    };
    switch (inst->op()) {
        case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_TRANSFORM: {
            auto accel = _value(builder, function, inst->operand(0u));
            auto index = _value(builder, function, inst->operand(1u));
            auto matrix = _value(builder, function, inst->operand(2u));
            auto instance = _accel_instance_pointer(builder, accel, index);
            auto instance_type = _accel_instance();
            auto transform = builder.CreateStructGEP(
                instance_type, instance, accel_instance_transform_field);
            auto transform_type = llvm::cast<llvm::ArrayType>(
                instance_type->getElementType(accel_instance_transform_field));
            for (auto column_index = 0u; column_index < 4u; column_index++) {
                auto column = builder.CreateExtractValue(matrix, column_index);
                for (auto row_index = 0u; row_index < 3u; row_index++) {
                    auto element_index = column_index * 3u + row_index;
                    auto element_pointer = builder.CreateInBoundsGEP(
                        transform_type, transform,
                        {builder.getInt32(0u), builder.getInt32(element_index)});
                    builder.CreateAlignedStore(
                        builder.CreateExtractElement(column, row_index),
                        element_pointer, llvm::Align{4u});
                }
            }
            return;
        }
        case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_VISIBILITY_MASK: [[fallthrough]];
        case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_USER_ID: {
            auto accel = _value(builder, function, inst->operand(0u));
            auto index = _value(builder, function, inst->operand(1u));
            auto value = _value(builder, function, inst->operand(2u));
            auto instance = _accel_instance_pointer(builder, accel, index);
            auto field = inst->op() == xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_USER_ID ?
                             accel_instance_user_id_field :
                             accel_instance_mask_field;
            auto pointer = builder.CreateStructGEP(
                _accel_instance(), instance, field);
            value = builder.CreateZExtOrTrunc(value, builder.getInt32Ty());
            builder.CreateAlignedStore(value, pointer, llvm::Align{4u});
            return;
        }
        case xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_OPACITY: {
            auto accel = _value(builder, function, inst->operand(0u));
            auto index = _value(builder, function, inst->operand(1u));
            auto opaque = _value(builder, function, inst->operand(2u));
            auto instance = _accel_instance_pointer(builder, accel, index);
            auto pointer = builder.CreateStructGEP(
                _accel_instance(), instance, accel_instance_options_field);
            llvm::Value *options = builder.CreateAlignedLoad(
                builder.getInt32Ty(), pointer, llvm::Align{4u});
            options = builder.CreateAnd(options, builder.getInt32(~(4u | 8u)));
            auto opacity = builder.CreateSelect(
                opaque, builder.getInt32(4u), builder.getInt32(8u));
            builder.CreateAlignedStore(
                builder.CreateOr(options, opacity), pointer, llvm::Align{4u});
            return;
        }
        case xir::ResourceWriteOp::BUFFER_WRITE: [[fallthrough]];
        case xir::ResourceWriteOp::BUFFER_VOLATILE_WRITE: [[fallthrough]];
        case xir::ResourceWriteOp::BYTE_BUFFER_WRITE: [[fallthrough]];
        case xir::ResourceWriteOp::BYTE_BUFFER_VOLATILE_WRITE: {
            auto buffer = _value(builder, function, inst->operand(0u));
            auto index = _value(builder, function, inst->operand(1u));
            auto value = _value(builder, function, inst->operand(2u));
            auto byte_addressed = inst->op() == xir::ResourceWriteOp::BYTE_BUFFER_WRITE ||
                                  inst->op() == xir::ResourceWriteOp::BYTE_BUFFER_VOLATILE_WRITE;
            auto is_volatile = inst->op() == xir::ResourceWriteOp::BUFFER_VOLATILE_WRITE ||
                               inst->op() == xir::ResourceWriteOp::BYTE_BUFFER_VOLATILE_WRITE;
            auto pointer = _buffer_pointer(builder, buffer, index, byte_addressed ? 1u : inst->operand(2u)->type()->size());
            _store(builder, pointer, value, inst->operand(2u)->type(), is_volatile);
            if (is_volatile) { _air_atomic_fence(builder, 1u); }
            return;
        }
        case xir::ResourceWriteOp::BINDLESS_BUFFER_WRITE: [[fallthrough]];
        case xir::ResourceWriteOp::BINDLESS_BYTE_BUFFER_WRITE: {
            auto array = _value(builder, function, inst->operand(0u));
            auto slot_index = _value(builder, function, inst->operand(1u));
            auto element_index = _value(builder, function, inst->operand(2u));
            auto value = _value(builder, function, inst->operand(3u));
            auto value_type = inst->operand(3u)->type();
            auto slot = _bindless_slot(builder, array, slot_index);
            auto buffer = _bindless_slot_field(builder, slot, 0u);
            auto stride = inst->op() == xir::ResourceWriteOp::BINDLESS_BUFFER_WRITE ?
                              value_type->size() :
                              1u;
            auto pointer = _device_pointer_offset(builder, buffer, element_index, stride);
            _store(builder, pointer, value, value_type);
            return;
        }
        case xir::ResourceWriteOp::DEVICE_ADDRESS_WRITE: {
            auto address = builder.CreateZExtOrTrunc(_value(builder, function, inst->operand(0u)), builder.getInt64Ty());
            auto value = _value(builder, function, inst->operand(1u));
            auto pointer = builder.CreateIntToPtr(address, llvm::PointerType::get(_context, air_address_space_device));
            _store(builder, pointer, value, inst->operand(1u)->type());
            return;
        }
        case xir::ResourceWriteOp::COOPERATIVE_VECTOR_STORE: {
            auto buffer = _value(builder, function, inst->operand(0u));
            auto offset = _value(builder, function, inst->operand(1u));
            auto value = _value(builder, function, inst->operand(2u));
            auto value_type = inst->operand(2u)->type();
            auto pointer = _buffer_pointer(builder, buffer, offset, 1u);
            store_cooperative_vector(pointer, value, value_type);
            return;
        }
        case xir::ResourceWriteOp::BINDLESS_COOPERATIVE_VECTOR_STORE: {
            auto array = _value(builder, function, inst->operand(0u));
            auto slot_index = _value(builder, function, inst->operand(1u));
            auto offset = _value(builder, function, inst->operand(2u));
            auto value = _value(builder, function, inst->operand(3u));
            auto value_type = inst->operand(3u)->type();
            auto slot = _bindless_slot(builder, array, slot_index);
            auto buffer = _bindless_slot_field(builder, slot, 0u);
            auto pointer = _device_pointer_offset(
                builder, buffer, offset, 1u);
            store_cooperative_vector(pointer, value, value_type);
            return;
        }
        case xir::ResourceWriteOp::COOPERATIVE_VECTOR_WORKGROUP_STORE: {
            auto shared = _value(builder, function, inst->operand(0u));
            auto index = _value(builder, function, inst->operand(1u));
            auto value = _value(builder, function, inst->operand(2u));
            auto value_type = inst->operand(2u)->type();
            auto pointer = builder.CreateInBoundsGEP(
                _type(value_type->element())->mem_type, shared, index);
            store_cooperative_vector(pointer, value, value_type);
            return;
        }
        case xir::ResourceWriteOp::COOPERATIVE_VECTOR_ACCUMULATE: {
            auto buffer = _value(builder, function, inst->operand(0u));
            auto offset = _value(builder, function, inst->operand(1u));
            auto value = _value(builder, function, inst->operand(2u));
            auto value_type = inst->operand(2u)->type();
            auto element_type = value_type->element();
            LUISA_ASSERT(element_type->is_int32() ||
                             element_type->is_uint32() ||
                             element_type->is_float32(),
                         "Metal cooperative-vector accumulation currently requires i32, u32, or f32 elements.");
            auto base = _buffer_pointer(builder, buffer, offset, 1u);
            auto suffix = element_type->is_float32() ?
                              luisa::string_view{"f32"} :
                          element_type->is_int32() ?
                              luisa::string_view{"s.i32"} :
                              luisa::string_view{"u.i32"};
            auto name = luisa::format(
                "air.atomic.global.add.{}", suffix);
            for (auto i = 0u; i < value_type->dimension(); i++) {
                auto pointer = i == 0u ?
                                   base :
                                   builder.CreateInBoundsGEP(
                                       _type(element_type)->mem_type,
                                       base, builder.getInt32(i));
                auto element = builder.CreateExtractValue(value, i);
                auto function_type = llvm::FunctionType::get(
                    element->getType(),
                    {pointer->getType(), element->getType(),
                     builder.getInt32Ty(), builder.getInt32Ty(),
                     builder.getInt1Ty()},
                    false);
                auto callee = _module.getOrInsertFunction(
                    llvm::StringRef{name.data(), name.size()}, function_type);
                if (auto intrinsic = llvm::dyn_cast<llvm::Function>(
                        callee.getCallee())) {
                    intrinsic->setMustProgress();
                    intrinsic->setDoesNotThrow();
                    intrinsic->setWillReturn();
                }
                builder.CreateCall(
                    callee,
                    {pointer, element, builder.getInt32(0u),
                     builder.getInt32(2u), builder.getInt1(true)});
            }
            return;
        }
        case xir::ResourceWriteOp::INDIRECT_DISPATCH_SET_COUNT: {
            auto buffer = indirect_dispatch_buffer();
            auto pointer = builder.CreateExtractValue(buffer, 0u);
            auto count = builder.CreateZExtOrTrunc(
                _value(builder, function, inst->operand(1u)),
                builder.getInt32Ty());
            // The host/device ABI gives the count header 16-byte alignment.
            builder.CreateAlignedStore(count, pointer, llvm::Align{16u});
            return;
        }
        case xir::ResourceWriteOp::INDIRECT_DISPATCH_SET_KERNEL: {
            auto buffer = indirect_dispatch_buffer();
            auto base = builder.CreateExtractValue(buffer, 0u);
            auto offset = builder.CreateExtractValue(buffer, 1u);
            auto capacity = builder.CreateExtractValue(buffer, 2u);
            auto local_index = builder.CreateZExtOrTrunc(
                _value(builder, function, inst->operand(1u)),
                builder.getInt32Ty());
            auto index = builder.CreateAdd(offset, local_index);
            auto in_bounds = builder.CreateICmpULT(index, capacity);
            auto llvm_function = builder.GetInsertBlock()->getParent();
            auto store_block = llvm::BasicBlock::Create(
                _context, "indirect.store", llvm_function);
            auto continue_block = llvm::BasicBlock::Create(
                _context, "indirect.continue", llvm_function);
            builder.CreateCondBr(in_bounds, store_block, continue_block);

            builder.SetInsertPoint(store_block);
            auto slots = builder.CreateInBoundsGEP(
                builder.getInt8Ty(), base, builder.getInt64(16u));
            auto slot_type = _indirect_dispatch_slot();
            auto slot = builder.CreateInBoundsGEP(
                slot_type, slots,
                builder.CreateZExt(index, builder.getInt64Ty()));
            auto block_size = _value(builder, function, inst->operand(2u));
            auto block_size_pointer = builder.CreateStructGEP(
                slot_type, slot, 0u);
            builder.CreateAlignedStore(
                block_size, block_size_pointer, llvm::Align{16u});

            auto dispatch_size = _value(
                builder, function, inst->operand(3u));
            auto dispatch_and_kernel = static_cast<llvm::Value *>(
                llvm::PoisonValue::get(
                    llvm::FixedVectorType::get(
                        builder.getInt32Ty(), 4u)));
            for (auto i = 0u; i < 3u; i++) {
                dispatch_and_kernel = builder.CreateInsertElement(
                    dispatch_and_kernel,
                    builder.CreateExtractElement(dispatch_size, i), i);
            }
            dispatch_and_kernel = builder.CreateInsertElement(
                dispatch_and_kernel,
                builder.CreateZExtOrTrunc(
                    _value(builder, function, inst->operand(4u)),
                    builder.getInt32Ty()),
                3u);
            auto dispatch_pointer = builder.CreateStructGEP(
                slot_type, slot, 1u);
            builder.CreateAlignedStore(
                dispatch_and_kernel, dispatch_pointer,
                llvm::Align{16u});
            builder.CreateBr(continue_block);
            builder.SetInsertPoint(continue_block);
            return;
        }
        case xir::ResourceWriteOp::TEXTURE2D_WRITE: [[fallthrough]];
        case xir::ResourceWriteOp::TEXTURE3D_WRITE: {
            auto texture_value = inst->operand(0u);
            auto texture_type = texture_value->type();
            auto texture = _value(builder, function, texture_value);
            auto coordinate = _value(builder, function, inst->operand(1u));
            auto value = _value(builder, function, inst->operand(2u));
            if (!value->getType()->isVectorTy()) {
                value = builder.CreateVectorSplat(4u, value);
            }
            auto dimension = texture_type->dimension();
            auto element = texture_type->element();
            auto access = _texture_access(texture_value);
            auto suffix = element->is_float32() ? std::string_view{"v4f32"} :
                          element->is_int32()   ? std::string_view{"s.v4i32"} :
                                                  std::string_view{"u.v4i32"};
            auto function_name = "air.write_texture_" + std::to_string(dimension) + "d." + std::string{suffix};
            auto function_type = llvm::FunctionType::get(
                builder.getVoidTy(),
                {texture->getType(), coordinate->getType(), value->getType(),
                 builder.getInt32Ty(), builder.getInt32Ty()},
                false);
            auto callee = _module.getOrInsertFunction(function_name, function_type);
            if (auto f = llvm::dyn_cast<llvm::Function>(callee.getCallee())) {
                std::array pointer_types{
                    std::pair<unsigned, llvm::Type *>{0u, _air_texture_handle(dimension)}};
                _set_air_pointer_element_types(f, pointer_types);
                f->setMustProgress();
                f->setDoesNotThrow();
                f->setWillReturn();
                f->setOnlyAccessesArgMemory();
            }
            builder.CreateCall(
                callee,
                {texture, coordinate, value,
                 builder.getInt32(0u), builder.getInt32(access)});
            if (access == air_texture_access_read_write) {
                auto fence_type = llvm::FunctionType::get(builder.getVoidTy(), {texture->getType()}, false);
                auto fence = _module.getOrInsertFunction(
                    "air.fence_texture_" + std::to_string(dimension) + "d", fence_type);
                if (auto f = llvm::dyn_cast<llvm::Function>(fence.getCallee())) {
                    std::array pointer_types{
                        std::pair<unsigned, llvm::Type *>{0u, _air_texture_handle(dimension)}};
                    _set_air_pointer_element_types(f, pointer_types);
                    f->setMustProgress();
                    f->setDoesNotThrow();
                    f->setWillReturn();
                }
                builder.CreateCall(fence, {texture});
            }
            return;
        }
        default: _unsupported_instruction(inst);
    }
}

}// namespace luisa::compute::metal::detail
