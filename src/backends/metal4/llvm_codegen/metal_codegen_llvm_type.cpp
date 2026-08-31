#include "metal_codegen_llvm_impl.h"

namespace luisa::compute::metal::detail {

[[noreturn]] void MetalCodegenLLVMImpl::_unsupported_instruction(const xir::Instruction *inst) noexcept {
    LUISA_ERROR_WITH_LOCATION(
        "Metal AIR LLVM codegen does not support XIR instruction '{}' yet (intrinsic '{}').",
        xir::to_string(inst->derived_instruction_tag()), inst->intrinsic_identifier());
}

[[noreturn]] void MetalCodegenLLVMImpl::_unsupported_type(const Type *type) noexcept {
    LUISA_ERROR_WITH_LOCATION("Metal AIR LLVM codegen does not support type '{}' yet.", type->description());
}

size_t MetalCodegenLLVMImpl::_type_alignment(const Type *type) noexcept {
    if (type->is_basic() || type->is_array() || type->is_structure() ||
        type->is_cooperative_vector()) {
        return type->alignment();
    }
    return kernel_argument_alignment;
}

const MetalCodegenLLVMImpl::LLVMTypeInfo *MetalCodegenLLVMImpl::_type(const Type *type) noexcept {
    if (auto iter = _types.find(type); iter != _types.end()) { return iter->second.get(); }
    auto make = [this](llvm::Type *mem_type, llvm::Type *reg_type,
                       luisa::vector<size_t> member_indices = {},
                       luisa::vector<size_t> member_offsets = {}) noexcept {
        return std::make_unique<LLVMTypeInfo>(LLVMTypeInfo{
            .mem_type = mem_type,
            .reg_type = reg_type,
            .member_indices = std::move(member_indices),
            .member_offsets = std::move(member_offsets)});
    };
    auto info = [&]() noexcept -> std::unique_ptr<LLVMTypeInfo> {
        if (type == nullptr) {
            auto void_type = llvm::Type::getVoidTy(_context);
            return make(void_type, void_type);
        }
        switch (type->tag()) {
            case Type::Tag::BOOL: {
                auto t = llvm::Type::getInt1Ty(_context);
                return make(t, t);
            }
            case Type::Tag::INT8: [[fallthrough]];
            case Type::Tag::UINT8: {
                auto t = llvm::Type::getInt8Ty(_context);
                return make(t, t);
            }
            case Type::Tag::INT16: [[fallthrough]];
            case Type::Tag::UINT16: {
                auto t = llvm::Type::getInt16Ty(_context);
                return make(t, t);
            }
            case Type::Tag::INT32: [[fallthrough]];
            case Type::Tag::UINT32: {
                auto t = llvm::Type::getInt32Ty(_context);
                return make(t, t);
            }
            case Type::Tag::INT64: [[fallthrough]];
            case Type::Tag::UINT64: {
                auto t = llvm::Type::getInt64Ty(_context);
                return make(t, t);
            }
            case Type::Tag::FLOAT16: {
                auto t = llvm::Type::getHalfTy(_context);
                return make(t, t);
            }
            case Type::Tag::FLOAT32: {
                auto t = llvm::Type::getFloatTy(_context);
                return make(t, t);
            }
            case Type::Tag::FLOAT64: [[fallthrough]];
            case Type::Tag::FLOAT8_E4M3: [[fallthrough]];
            case Type::Tag::FLOAT8_E5M2: [[fallthrough]];
            case Type::Tag::INT4: [[fallthrough]];
            case Type::Tag::FP4_E2M1: _unsupported_type(type);
            case Type::Tag::VECTOR: {
                auto element = _type(type->element());
                auto dimension = type->dimension();
                auto reg_type = llvm::FixedVectorType::get(element->reg_type, dimension);
                // Native non-bool LLVM vectors have the same padded size and
                // alignment as Luisa vectors under the AIR data layout, and
                // preserving them in memory enables vector loads/stores. LLVM
                // i1 vectors are bit-packed, so bool vectors deliberately use
                // byte-aligned i1 array storage instead (bool4 is four bytes).
                auto mem_type = type->element()->is_bool() ?
                                    static_cast<llvm::Type *>(llvm::ArrayType::get(
                                        element->mem_type, luisa::align(dimension, 2u))) :
                                    static_cast<llvm::Type *>(reg_type);
                return make(mem_type, reg_type);
            }
            case Type::Tag::MATRIX: {
                auto dimension = type->dimension();
                auto column = _type(Type::vector(type->element(), dimension));
                return make(llvm::ArrayType::get(column->mem_type, dimension),
                            llvm::ArrayType::get(column->reg_type, dimension));
            }
            case Type::Tag::ARRAY: {
                auto element = _type(type->element());
                return make(llvm::ArrayType::get(element->mem_type, type->dimension()),
                            llvm::ArrayType::get(element->reg_type, type->dimension()));
            }
            case Type::Tag::STRUCTURE: {
                llvm::SmallVector<llvm::Type *> memory_members;
                llvm::SmallVector<llvm::Type *> register_members;
                luisa::vector<size_t> member_indices;
                luisa::vector<size_t> member_offsets;
                auto byte_type = llvm::Type::getInt8Ty(_context);
                auto offset = static_cast<size_t>(0u);
                for (auto member : type->members()) {
                    auto member_type = _type(member);
                    register_members.emplace_back(member_type->reg_type);
                    auto aligned_offset = luisa::align(offset, member->alignment());
                    if (aligned_offset != offset) {
                        memory_members.emplace_back(llvm::ArrayType::get(byte_type, aligned_offset - offset));
                    }
                    member_indices.emplace_back(memory_members.size());
                    member_offsets.emplace_back(aligned_offset);
                    memory_members.emplace_back(member_type->mem_type);
                    offset = aligned_offset + member->size();
                }
                if (offset < type->size()) {
                    memory_members.emplace_back(llvm::ArrayType::get(byte_type, type->size() - offset));
                }
                return make(llvm::StructType::get(_context, memory_members, false),
                            llvm::StructType::get(_context, register_members, false),
                            std::move(member_indices), std::move(member_offsets));
            }
            case Type::Tag::BUFFER: {
                auto t = _buffer(type->element());
                return make(t, t);
            }
            case Type::Tag::TEXTURE: {
                auto t = llvm::PointerType::get(_context, air_address_space_device);
                return make(t, t);
            }
            case Type::Tag::BINDLESS_ARRAY: {
                auto t = _bindless_array();
                return make(t, t);
            }
            case Type::Tag::ACCEL: {
                auto t = _accel();
                return make(t, t);
            }
            case Type::Tag::COOPERATIVE_VECTOR: {
                auto element = _type(type->element());
                return make(
                    llvm::ArrayType::get(
                        element->mem_type, type->dimension()),
                    llvm::ArrayType::get(
                        element->reg_type, type->dimension()));
            }
            case Type::Tag::COOPERATIVE_VECTOR_REF: [[fallthrough]];
            case Type::Tag::COOPERATIVE_MATRIX_REF: {
                // AST-to-XIR normalizes cooperative references to byte offsets.
                // Keep the same representation here so hand-built XIR and
                // external ABI diagnostics remain deterministic.
                auto offset = llvm::Type::getInt32Ty(_context);
                return make(offset, offset);
            }
            case Type::Tag::CUSTOM: {
                if (is_indirect_dispatch_buffer_type(type)) {
                    auto t = _indirect_dispatch_buffer();
                    return make(t, t);
                }
                if (is_ray_query_type(type)) {
                    auto t = llvm::PointerType::get(
                        _context, air_address_space_generic);
                    return make(t, t);
                }
                _unsupported_type(type);
            }
        }
        _unsupported_type(type);
    }();
    if (type != nullptr && !type->is_resource() && !type->is_custom() &&
        !type->is_cooperative_vector_ref() &&
        !type->is_cooperative_matrix_ref()) {
        auto llvm_size = _data_layout.getTypeAllocSize(info->mem_type).getFixedValue();
        LUISA_ASSERT(
            llvm_size == type->size(),
            "Metal AIR memory type size mismatch for '{}': LLVM has {} bytes, Luisa has {} bytes.",
            type->description(), llvm_size, type->size());
    }
    auto [iter, inserted] = _types.try_emplace(type, std::move(info));
    LUISA_ASSERT(inserted, "Failed to cache a Metal LLVM type.");
    return iter->second.get();
}

llvm::StructType *MetalCodegenLLVMImpl::_buffer(const Type *element) noexcept {
    if (auto iter = _buffer_types.find(element); iter != _buffer_types.end()) { return iter->second; }
    auto device_pointer = llvm::PointerType::get(_context, air_address_space_device);
    auto size = llvm::Type::getInt64Ty(_context);
    auto name = "luisa.buffer." + std::to_string(_buffer_types.size());
    auto type = llvm::StructType::create(_context, {device_pointer, size}, name);
    _set_struct_pointer_element_type(type, 0u, llvm::Type::getInt8Ty(_context));
    _buffer_types.try_emplace(element, type);
    return type;
}

llvm::StructType *MetalCodegenLLVMImpl::_indirect_dispatch_buffer() noexcept {
    if (_indirect_dispatch_buffer_type == nullptr) {
        auto device_pointer = llvm::PointerType::get(
            _context, air_address_space_device);
        auto i32 = llvm::Type::getInt32Ty(_context);
        _indirect_dispatch_buffer_type = llvm::StructType::create(
            _context, {device_pointer, i32, i32},
            "luisa.indirect.dispatch.buffer");
        _set_struct_pointer_element_type(
            _indirect_dispatch_buffer_type, 0u,
            llvm::Type::getInt8Ty(_context));
        LUISA_ASSERT(
            _data_layout.getTypeAllocSize(
                            _indirect_dispatch_buffer_type)
                    .getFixedValue() == 16u,
            "Unexpected Metal indirect-dispatch buffer LLVM ABI size.");
    }
    return _indirect_dispatch_buffer_type;
}

llvm::StructType *MetalCodegenLLVMImpl::_indirect_dispatch_slot() noexcept {
    if (_indirect_dispatch_slot_type == nullptr) {
        auto i32 = llvm::Type::getInt32Ty(_context);
        _indirect_dispatch_slot_type = llvm::StructType::create(
            _context,
            {llvm::FixedVectorType::get(i32, 3u),
             llvm::FixedVectorType::get(i32, 4u)},
            "luisa.indirect.dispatch.slot");
        LUISA_ASSERT(
            _data_layout.getTypeAllocSize(
                            _indirect_dispatch_slot_type)
                    .getFixedValue() == 32u,
            "Unexpected Metal indirect-dispatch slot LLVM ABI size.");
    }
    return _indirect_dispatch_slot_type;
}

llvm::StructType *MetalCodegenLLVMImpl::_bindless_item() noexcept {
    if (_bindless_item_type == nullptr) {
        auto device_pointer = llvm::PointerType::get(_context, air_address_space_device);
        _bindless_item_type = llvm::StructType::create(
            _context,
            {device_pointer, llvm::Type::getInt64Ty(_context),
             _air_texture_wrapper(2u), _air_texture_wrapper(3u)},
            "luisa.bindless.item");
        _set_struct_pointer_element_type(
            _bindless_item_type, 0u, llvm::Type::getInt8Ty(_context));
        LUISA_ASSERT(_data_layout.getTypeAllocSize(_bindless_item_type).getFixedValue() == 32u,
                     "Unexpected Metal bindless-item LLVM ABI size.");
    }
    return _bindless_item_type;
}

llvm::StructType *MetalCodegenLLVMImpl::_bindless_array() noexcept {
    if (_bindless_array_type == nullptr) {
        auto device_pointer = llvm::PointerType::get(_context, air_address_space_device);
        _bindless_array_type = llvm::StructType::create(
            _context, {device_pointer}, "luisa.bindless.array");
        _set_struct_pointer_element_type(
            _bindless_array_type, 0u, _bindless_item());
        LUISA_ASSERT(_data_layout.getTypeAllocSize(_bindless_array_type).getFixedValue() == 8u,
                     "Unexpected Metal bindless-array LLVM ABI size.");
    }
    return _bindless_array_type;
}

llvm::StructType *MetalCodegenLLVMImpl::_air_texture_handle(unsigned dimension) noexcept {
    LUISA_ASSERT(dimension == 2u || dimension == 3u,
                 "AIR texture handles must be two- or three-dimensional.");
    auto &type = _air_texture_handle_types[dimension];
    if (type == nullptr) {
        type = llvm::StructType::create(
            _context, "struct._texture_" + std::to_string(dimension) + "d_t");
    }
    return type;
}

llvm::StructType *MetalCodegenLLVMImpl::_air_texture_wrapper(
    unsigned dimension) noexcept {
    LUISA_ASSERT(dimension == 2u || dimension == 3u,
                 "AIR texture wrappers must be two- or three-dimensional.");
    auto &type = _air_texture_wrapper_types[dimension];
    if (type == nullptr) {
        auto pointer = llvm::PointerType::get(
            _context, air_address_space_device);
        type = llvm::StructType::create(
            _context, {pointer},
            "luisa.air.texture." + std::to_string(dimension) + "d");
        _set_struct_pointer_element_type(
            type, 0u, _air_texture_handle(dimension));
    }
    return type;
}

llvm::StructType *MetalCodegenLLVMImpl::_air_sampler_handle() noexcept {
    if (_air_sampler_handle_type == nullptr) {
        _air_sampler_handle_type = llvm::StructType::create(
            _context, "struct._sampler_t");
    }
    return _air_sampler_handle_type;
}

llvm::StructType *MetalCodegenLLVMImpl::_air_accel_handle() noexcept {
    if (_air_accel_handle_type == nullptr) {
        _air_accel_handle_type = llvm::StructType::create(
            _context, "struct._instance_acceleration_structure_t");
    }
    return _air_accel_handle_type;
}

llvm::StructType *MetalCodegenLLVMImpl::_air_accel_wrapper() noexcept {
    if (_air_accel_wrapper_type == nullptr) {
        auto pointer = llvm::PointerType::get(
            _context, air_address_space_device);
        _air_accel_wrapper_type = llvm::StructType::create(
            _context, {pointer}, "luisa.air.accel.handle");
        _set_struct_pointer_element_type(
            _air_accel_wrapper_type, 0u, _air_accel_handle());
    }
    return _air_accel_wrapper_type;
}

llvm::StructType *MetalCodegenLLVMImpl::_accel_instance() noexcept {
    if (_accel_instance_type == nullptr) {
        auto f32 = llvm::Type::getFloatTy(_context);
        auto i32 = llvm::Type::getInt32Ty(_context);
        auto i64 = llvm::Type::getInt64Ty(_context);
        _accel_instance_type = llvm::StructType::create(
            _context,
            {llvm::ArrayType::get(f32, 12u), i32, i32, i32, i32, i64},
            "luisa.accel.instance");
        LUISA_ASSERT(
            _data_layout.getTypeAllocSize(_accel_instance_type).getFixedValue() == 72u,
            "Unexpected Metal acceleration-instance LLVM ABI size.");
    }
    return _accel_instance_type;
}

llvm::StructType *MetalCodegenLLVMImpl::_accel() noexcept {
    if (_accel_type == nullptr) {
        auto device_pointer = llvm::PointerType::get(
            _context, air_address_space_device);
        _accel_type = llvm::StructType::create(
            _context, {_air_accel_wrapper(), device_pointer},
            "luisa.accel");
        _set_struct_pointer_element_type(
            _accel_type, 1u, _accel_instance());
        LUISA_ASSERT(
            _data_layout.getTypeAllocSize(_accel_type).getFixedValue() == 16u,
            "Unexpected Metal acceleration-structure LLVM ABI size.");
    }
    return _accel_type;
}

llvm::StructType *MetalCodegenLLVMImpl::_air_intersection_function_table() noexcept {
    if (_air_intersection_function_table_type == nullptr) {
        _air_intersection_function_table_type = llvm::StructType::create(
            _context, "struct._intersection_function_table_t");
    }
    return _air_intersection_function_table_type;
}

llvm::StructType *MetalCodegenLLVMImpl::_air_intersection_result(
    bool curves) noexcept {
    auto &result_type = curves ?
                            _air_curve_intersection_result_type :
                            _air_intersection_result_type;
    if (result_type == nullptr) {
        auto i32 = llvm::Type::getInt32Ty(_context);
        auto f32 = llvm::Type::getFloatTy(_context);
        auto device_pointer = llvm::PointerType::get(
            _context, air_address_space_device);
        llvm::SmallVector<llvm::Type *, 10u> fields{
            i32, f32, i32, i32, device_pointer, i32, i32,
            llvm::FixedVectorType::get(f32, 2u),
            llvm::Type::getInt1Ty(_context)};
        if (curves) { fields.emplace_back(f32); }
        result_type = llvm::StructType::create(
            _context, fields,
            curves ? "luisa.air.curve.intersection.result" :
                     "luisa.air.intersection.result");
        _set_struct_pointer_element_type(
            result_type, 4u,
            llvm::Type::getInt8Ty(_context));
    }
    return result_type;
}

llvm::StructType *MetalCodegenLLVMImpl::_air_intersection_query() noexcept {
    if (_air_intersection_query_type == nullptr) {
        _air_intersection_query_type = llvm::StructType::create(
            _context, "struct._intersection_query_t");
    }
    return _air_intersection_query_type;
}

llvm::Constant *MetalCodegenLLVMImpl::_constant_string(
    luisa::string_view value, luisa::string_view name) noexcept {
    auto data = llvm::ConstantDataArray::getString(
        _context, llvm::StringRef{value.data(), value.size()}, true);
    auto global = new llvm::GlobalVariable(
        _module, data->getType(), true,
        llvm::GlobalValue::PrivateLinkage, data,
        llvm::StringRef{name.data(), name.size()}, nullptr,
        llvm::GlobalVariable::NotThreadLocal,
        air_address_space_constant);
    global->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);
    global->setAlignment(llvm::Align{1u});
    auto zero = llvm::ConstantInt::get(
        llvm::Type::getInt64Ty(_context), 0u);
    std::array<llvm::Constant *, 2u> indices{zero, zero};
    return llvm::ConstantExpr::getInBoundsGetElementPtr(
        data->getType(), global,
        llvm::ArrayRef<llvm::Constant *>{
            indices.data(), indices.size()});
}

llvm::Function *MetalCodegenLLVMImpl::_shader_log() noexcept {
    if (_shader_log_helper != nullptr) { return _shader_log_helper; }
    auto constant_pointer = llvm::PointerType::get(
        _context, air_address_space_constant);
    auto generic_pointer = llvm::PointerType::get(
        _context, air_address_space_generic);
    auto i32 = llvm::Type::getInt32Ty(_context);
    auto i64 = llvm::Type::getInt64Ty(_context);
    auto function_type = llvm::FunctionType::get(
        llvm::Type::getVoidTy(_context),
        {constant_pointer, i64}, true);
    _shader_log_helper = llvm::Function::Create(
        function_type, llvm::GlobalValue::LinkOnceODRLinkage,
        "__luisa_metal_shader_log", _module);
    _shader_log_helper->addFnAttr(llvm::Attribute::AlwaysInline);
    _shader_log_helper->setMustProgress();
    _shader_log_helper->setDoesNotThrow();
    _set_float_control_attributes(_shader_log_helper);

    auto argument = _shader_log_helper->arg_begin();
    auto format = argument++;
    auto argument_size = argument++;
    format->setName("format");
    argument_size->setName("argument.size");
    auto entry = llvm::BasicBlock::Create(
        _context, "entry", _shader_log_helper);
    IB builder{entry};
    auto va_list = builder.CreateAlloca(generic_pointer, nullptr, "va.list");
    auto va_start = llvm::Intrinsic::getOrInsertDeclaration(
        &_module, llvm::Intrinsic::vastart, {generic_pointer});
    auto va_end = llvm::Intrinsic::getOrInsertDeclaration(
        &_module, llvm::Intrinsic::vaend, {generic_pointer});
    builder.CreateCall(va_start, {va_list});
    auto arguments = builder.CreateAlignedLoad(
        generic_pointer, va_list, llvm::Align{8u});
    auto os_log_type = llvm::FunctionType::get(
        llvm::Type::getVoidTy(_context),
        {constant_pointer, constant_pointer, i32,
         constant_pointer, generic_pointer, i64}, false);
    auto os_log = _module.getOrInsertFunction("air.os_log", os_log_type);
    if (auto function = llvm::dyn_cast<llvm::Function>(os_log.getCallee())) {
        function->setMustProgress();
        function->setDoesNotThrow();
        function->setWillReturn();
    }
    auto subsystem = _constant_string(
        shader_log_subsystem, "luisa.shader.log.subsystem");
    auto category = _constant_string(
        shader_log_category, "luisa.shader.log.category");
    builder.CreateCall(
        os_log,
        {subsystem, category, builder.getInt32(1u),
         format, arguments, argument_size});
    builder.CreateCall(va_end, {va_list});
    builder.CreateRetVoid();
    return _shader_log_helper;
}

void MetalCodegenLLVMImpl::_append_shader_log_type(
    luisa::string &format, const Type *type) const noexcept {
    if (type->is_scalar()) {
        switch (type->tag()) {
            case Type::Tag::BOOL:
                format.append(shader_log_bool_prefix)
                    .append("%d")
                    .append(shader_log_bool_suffix);
                return;
            case Type::Tag::INT8: [[fallthrough]];
            case Type::Tag::INT16: [[fallthrough]];
            case Type::Tag::INT32: format.append("%d"); return;
            case Type::Tag::UINT8: [[fallthrough]];
            case Type::Tag::UINT16: [[fallthrough]];
            case Type::Tag::UINT32: format.append("%u"); return;
            case Type::Tag::INT64: format.append("%ld"); return;
            case Type::Tag::UINT64: format.append("%lu"); return;
            case Type::Tag::FLOAT16: [[fallthrough]];
            case Type::Tag::FLOAT32: format.append("%g"); return;
            default: _unsupported_type(type);
        }
    }
    auto append_sequence = [&](luisa::string_view open,
                               luisa::string_view close,
                               auto &&element) noexcept {
        format.append(open);
        for (auto i = 0u; i < type->dimension(); i++) {
            if (i != 0u) { format.append(", "); }
            element(i);
        }
        format.append(close);
    };
    if (type->is_vector()) {
        append_sequence("(", ")", [&](auto) noexcept {
            _append_shader_log_type(format, type->element());
        });
        return;
    }
    if (type->is_array()) {
        append_sequence("[", "]", [&](auto) noexcept {
            _append_shader_log_type(format, type->element());
        });
        return;
    }
    if (type->is_matrix()) {
        auto column = Type::vector(type->element(), type->dimension());
        append_sequence("<", ">", [&](auto) noexcept {
            _append_shader_log_type(format, column);
        });
        return;
    }
    if (type->is_structure()) {
        format.push_back('{');
        for (auto i = 0u; i < type->members().size(); i++) {
            if (i != 0u) { format.append(", "); }
            _append_shader_log_type(format, type->members()[i]);
        }
        format.push_back('}');
        return;
    }
    _unsupported_type(type);
}

luisa::string MetalCodegenLLVMImpl::_shader_log_format(
    luisa::string_view format,
    luisa::span<const Type *const> arguments) const noexcept {
    luisa::string native;
    native.reserve(format.size() + arguments.size() * 8u);
    auto argument_index = 0u;
    for (auto i = 0u; i < format.size();) {
        auto c = format[i++];
        if (c == '{') {
            LUISA_ASSERT(i < format.size(),
                         "Invalid Metal shader-log format string '{}'.", format);
            if (format[i] == '{') {
                native.push_back('{');
                i++;
            } else {
                LUISA_ASSERT(format[i] == '}',
                             "Unsupported Metal shader-log format string '{}'.", format);
                i++;
                LUISA_ASSERT(argument_index < arguments.size(),
                             "Metal shader-log format has too few operands.");
                _append_shader_log_type(native, arguments[argument_index++]);
            }
        } else if (c == '}') {
            LUISA_ASSERT(i < format.size() && format[i] == '}',
                         "Invalid Metal shader-log format string '{}'.", format);
            native.push_back('}');
            i++;
        } else {
            if (c == '%') { native.push_back('%'); }
            native.push_back(c);
        }
    }
    LUISA_ASSERT(argument_index == arguments.size(),
                 "Metal shader-log format has too many operands.");
    LUISA_ASSERT(native.size() < 1024u,
                 "Metal shader-log format exceeds the native 1023-byte limit.");
    return native;
}

void MetalCodegenLLVMImpl::_append_shader_log_arguments(
    IB &builder, llvm::Value *value, const Type *type,
    llvm::SmallVectorImpl<llvm::Value *> &arguments,
    size_t &argument_size) noexcept {
    if (type->is_scalar()) {
        auto argument = value;
        auto size = 0u;
        auto alignment = 0u;
        switch (type->tag()) {
            case Type::Tag::BOOL:
                argument = builder.CreateZExt(argument, builder.getInt32Ty());
                size = alignment = 4u;
                break;
            case Type::Tag::INT8: [[fallthrough]];
            case Type::Tag::INT16:
                argument = builder.CreateSExt(argument, builder.getInt32Ty());
                size = alignment = 4u;
                break;
            case Type::Tag::UINT8: [[fallthrough]];
            case Type::Tag::UINT16:
                argument = builder.CreateZExt(argument, builder.getInt32Ty());
                size = alignment = 4u;
                break;
            case Type::Tag::INT32: [[fallthrough]];
            case Type::Tag::UINT32:
                size = alignment = 4u;
                break;
            case Type::Tag::INT64: [[fallthrough]];
            case Type::Tag::UINT64:
                size = alignment = 8u;
                break;
            case Type::Tag::FLOAT16: [[fallthrough]];
            case Type::Tag::FLOAT32:
                argument = builder.CreateFPExt(
                    argument, llvm::Type::getDoubleTy(_context));
                size = alignment = 8u;
                break;
            default: _unsupported_type(type);
        }
        argument_size = luisa::align(argument_size, alignment) + size;
        arguments.emplace_back(argument);
        return;
    }
    if (type->is_vector()) {
        for (auto i = 0u; i < type->dimension(); i++) {
            _append_shader_log_arguments(
                builder, builder.CreateExtractElement(value, i),
                type->element(), arguments, argument_size);
        }
        return;
    }
    if (type->is_array() || type->is_matrix()) {
        auto element = type->is_matrix() ?
                           Type::vector(type->element(), type->dimension()) :
                           type->element();
        for (auto i = 0u; i < type->dimension(); i++) {
            _append_shader_log_arguments(
                builder, builder.CreateExtractValue(value, i),
                element, arguments, argument_size);
        }
        return;
    }
    if (type->is_structure()) {
        for (auto i = 0u; i < type->members().size(); i++) {
            _append_shader_log_arguments(
                builder, builder.CreateExtractValue(value, i),
                type->members()[i], arguments, argument_size);
        }
        return;
    }
    _unsupported_type(type);
}

void MetalCodegenLLVMImpl::_set_air_pointer_element_types(
    llvm::Function *function,
    llvm::ArrayRef<std::pair<unsigned, llvm::Type *>> arguments,
    llvm::Type *return_element) noexcept {
    if (!arguments.empty()) {
        llvm::SmallVector<llvm::Metadata *> metadata;
        metadata.reserve(arguments.size() * 2u);
        for (auto [index, element] : arguments) {
            LUISA_ASSERT(index < function->arg_size() &&
                             function->getFunctionType()->getParamType(index)->isPointerTy(),
                         "Invalid AIR pointer-element metadata argument index.");
            metadata.emplace_back(md_i32(_context, index));
            metadata.emplace_back(llvm::ValueAsMetadata::get(
                llvm::UndefValue::get(element)));
        }
        function->setMetadata("arg_eltypes", llvm::MDNode::get(_context, metadata));
    }
    if (return_element != nullptr) {
        LUISA_ASSERT(function->getReturnType()->isPointerTy(),
                     "AIR return-element metadata requires a pointer return type.");
        function->setMetadata(
            "ret_eltype",
            llvm::MDNode::get(
                _context,
                {llvm::ValueAsMetadata::get(llvm::UndefValue::get(return_element))}));
    }
}

void MetalCodegenLLVMImpl::_set_struct_pointer_element_type(
    llvm::StructType *structure, unsigned field,
    llvm::Type *element) noexcept {
    LUISA_ASSERT(structure->hasName() && field < structure->getNumElements() &&
                     structure->getElementType(field)->isPointerTy(),
                 "Invalid LLVM struct pointer-element metadata field.");
    auto metadata = llvm::MDNode::get(
        _context,
        {llvm::MDString::get(_context, structure->getName()),
         md_i32(_context, field),
         llvm::ValueAsMetadata::get(llvm::UndefValue::get(element))});
    _module.getOrInsertNamedMetadata("llvm.struct_eltypes")->addOperand(metadata);
}

const MetalCodegenLLVMImpl::KernelArguments &MetalCodegenLLVMImpl::_root_argument_layout() noexcept {
    if (_root_argument_layout_initialized) { return _root_argument_layout_cache; }
    KernelArguments layout;
    llvm::SmallVector<llvm::Type *> members;
    auto byte_type = llvm::Type::getInt8Ty(_context);
    auto offset = static_cast<size_t>(0u);
    auto append_member = [&](llvm::Type *type) noexcept {
        auto aligned_offset = luisa::align(offset, kernel_argument_alignment);
        if (aligned_offset != offset) {
            members.emplace_back(llvm::ArrayType::get(
                byte_type, aligned_offset - offset));
        }
        auto member_index = static_cast<unsigned>(members.size());
        members.emplace_back(type);
        offset = aligned_offset +
                 _data_layout.getTypeAllocSize(type).getFixedValue();
        return std::pair{aligned_offset, member_index};
    };
    for (auto argument : _root_arguments) {
        LUISA_ASSERT(
            !argument->is_reference() ||
                is_indirect_dispatch_buffer_type(argument->type()),
            "Only the opaque indirect-dispatch buffer may use an XIR reference kernel argument in the Metal host ABI.");
        auto llvm_type = _type(argument->type())->mem_type;
        auto [argument_offset, member_index] = append_member(llvm_type);
        layout.offsets.emplace_back(argument_offset);
        layout.member_indices.emplace_back(member_index);
        if (argument->type()->is_texture() &&
            _texture_needs_sampled_split(argument)) {
            auto [sampled_offset, sampled_member_index] =
                append_member(llvm_type);
            layout.sampled_texture_offsets.emplace_back(sampled_offset);
            layout.sampled_texture_member_indices.emplace_back(
                sampled_member_index);
        } else {
            layout.sampled_texture_offsets.emplace_back(
                std::numeric_limits<size_t>::max());
            layout.sampled_texture_member_indices.emplace_back(
                std::numeric_limits<unsigned>::max());
        }
    }
    layout.size = std::max<size_t>(kernel_argument_alignment, luisa::align(offset, kernel_argument_alignment));
    if (layout.size != offset) {
        members.emplace_back(llvm::ArrayType::get(byte_type, layout.size - offset));
    }
    layout.type = llvm::StructType::create(_context, members, "luisa.arguments");
    auto argument_index = 0u;
    for (auto argument : _root_arguments) {
        auto argument_type = argument->type();
        if (argument_type->is_texture()) {
            _set_struct_pointer_element_type(
                layout.type, layout.member_indices[argument_index],
                _air_texture_handle(argument_type->dimension()));
            if (layout.sampled_texture_member_indices[argument_index] !=
                std::numeric_limits<unsigned>::max()) {
                _set_struct_pointer_element_type(
                    layout.type,
                    layout.sampled_texture_member_indices[argument_index],
                    _air_texture_handle(argument_type->dimension()));
            }
        }
        argument_index++;
    }
    _root_argument_layout_cache = std::move(layout);
    _root_argument_layout_initialized = true;
    _result.root_argument_size = _root_argument_layout_cache.size;
    return _root_argument_layout_cache;
}

MetalCodegenLLVMImpl::RasterVertexInput
MetalCodegenLLVMImpl::_raster_vertex_input(PixelFormat format) noexcept {
    enum class Scalar : uint8_t { SINT,
                                  UINT,
                                  HALF,
                                  FLOAT };
    auto scalar = Scalar::FLOAT;
    auto dimension = pixel_format_channel_count(format);
    switch (format) {
        case PixelFormat::R8SInt: [[fallthrough]];
        case PixelFormat::RG8SInt: [[fallthrough]];
        case PixelFormat::RGBA8SInt: [[fallthrough]];
        case PixelFormat::R16SInt: [[fallthrough]];
        case PixelFormat::RG16SInt: [[fallthrough]];
        case PixelFormat::RGBA16SInt: [[fallthrough]];
        case PixelFormat::R32SInt: [[fallthrough]];
        case PixelFormat::RG32SInt: [[fallthrough]];
        case PixelFormat::RGBA32SInt: scalar = Scalar::SINT; break;
        case PixelFormat::R8UInt: [[fallthrough]];
        case PixelFormat::RG8UInt: [[fallthrough]];
        case PixelFormat::RGBA8UInt: [[fallthrough]];
        case PixelFormat::R16UInt: [[fallthrough]];
        case PixelFormat::RG16UInt: [[fallthrough]];
        case PixelFormat::RGBA16UInt: [[fallthrough]];
        case PixelFormat::R32UInt: [[fallthrough]];
        case PixelFormat::RG32UInt: [[fallthrough]];
        case PixelFormat::RGBA32UInt: scalar = Scalar::UINT; break;
        case PixelFormat::R16F: [[fallthrough]];
        case PixelFormat::RG16F: [[fallthrough]];
        case PixelFormat::RGBA16F: scalar = Scalar::HALF; break;
        case PixelFormat::R8UNorm: [[fallthrough]];
        case PixelFormat::RG8UNorm: [[fallthrough]];
        case PixelFormat::RGBA8UNorm: [[fallthrough]];
        case PixelFormat::R16UNorm: [[fallthrough]];
        case PixelFormat::RG16UNorm: [[fallthrough]];
        case PixelFormat::RGBA16UNorm: [[fallthrough]];
        case PixelFormat::R32F: [[fallthrough]];
        case PixelFormat::RG32F: [[fallthrough]];
        case PixelFormat::RGBA32F: [[fallthrough]];
        case PixelFormat::R10G10B10A2UNorm: [[fallthrough]];
        case PixelFormat::R11G11B10F: scalar = Scalar::FLOAT; break;
        case PixelFormat::R10G10B10A2UInt: [[fallthrough]];
        case PixelFormat::RGBA8SRGB:
            LUISA_ERROR_WITH_LOCATION(
                "Pixel format {} has no semantics-preserving Metal vertex format.",
                static_cast<uint32_t>(format));
        case PixelFormat::BC1UNorm: [[fallthrough]];
        case PixelFormat::BC2UNorm: [[fallthrough]];
        case PixelFormat::BC3UNorm: [[fallthrough]];
        case PixelFormat::BC4UNorm: [[fallthrough]];
        case PixelFormat::BC5UNorm: [[fallthrough]];
        case PixelFormat::BC6HUF16: [[fallthrough]];
        case PixelFormat::BC7UNorm: [[fallthrough]];
        case PixelFormat::BC7SRGB:
            LUISA_ERROR_WITH_LOCATION(
                "Block-compressed format {} cannot be a Metal raster vertex attribute.",
                static_cast<uint32_t>(format));
    }
    llvm::Type *element = nullptr;
    luisa::string_view name;
    switch (scalar) {
        case Scalar::SINT:
            element = llvm::Type::getInt32Ty(_context);
            name = "int";
            break;
        case Scalar::UINT:
            element = llvm::Type::getInt32Ty(_context);
            name = "uint";
            break;
        case Scalar::HALF:
            element = llvm::Type::getHalfTy(_context);
            name = "half";
            break;
        case Scalar::FLOAT:
            element = llvm::Type::getFloatTy(_context);
            name = "float";
            break;
    }
    auto type = dimension == 1u ?
                    element :
                    static_cast<llvm::Type *>(llvm::FixedVectorType::get(element, dimension));
    auto air_name = luisa::string{name};
    if (dimension != 1u) { air_name.append(std::to_string(dimension)); }
    return {.type = type,
            .air_type_name = std::move(air_name),
            .dimension = dimension,
            .signed_integer = scalar == Scalar::SINT};
}

llvm::Value *MetalCodegenLLVMImpl::_load_root_argument(
    IB &builder, llvm::Value *root, const xir::Argument *argument,
    size_t root_index, bool sampled_texture) noexcept {
    auto layout = _root_argument_layout();
    LUISA_ASSERT(root_index < _root_arguments.size(),
                 "Metal AIR root argument index is out of bounds.");
    auto member_index = layout.member_indices[root_index];
    if (sampled_texture) {
        LUISA_ASSERT(
            argument->type()->is_texture() &&
                layout.sampled_texture_member_indices[root_index] !=
                    std::numeric_limits<unsigned>::max(),
            "Metal AIR sampled texture root argument was not split.");
        member_index =
            layout.sampled_texture_member_indices[root_index];
    }
    auto field_pointer = builder.CreateStructGEP(
        layout.type, root, member_index);
    if (argument->is_reference()) {
        LUISA_ASSERT(
            is_indirect_dispatch_buffer_type(argument->type()),
            "Unsupported Metal AIR reference root argument.");
        return _load(builder, field_pointer, argument->type());
    }
    if (argument->type()->is_buffer()) {
        auto buffer_type = _buffer(argument->type()->element());
        auto device_pointer_field = builder.CreateStructGEP(
            buffer_type, field_pointer, 0u);
        auto device_pointer = builder.CreateAlignedLoad(
            buffer_type->getElementType(0u), device_pointer_field,
            llvm::Align{16u});
        auto size_pointer = builder.CreateStructGEP(
            buffer_type, field_pointer, 1u);
        auto size = builder.CreateAlignedLoad(
            buffer_type->getElementType(1u), size_pointer,
            llvm::Align{8u});
        auto buffer = static_cast<llvm::Value *>(
            llvm::PoisonValue::get(buffer_type));
        buffer = builder.CreateInsertValue(buffer, device_pointer, 0u);
        return builder.CreateInsertValue(buffer, size, 1u);
    }
    return _load(builder, field_pointer, argument->type());
}

llvm::Value *MetalCodegenLLVMImpl::_reg_to_mem(IB &builder, llvm::Value *value, const Type *type) noexcept {
    auto type_info = _type(type);
    LUISA_ASSERT(value->getType() == type_info->reg_type, "Invalid register value type in Metal LLVM codegen.");
    if (type_info->reg_type == type_info->mem_type) { return value; }
    if (type_info->reg_type->isVectorTy()) {
        auto result = static_cast<llvm::Value *>(llvm::Constant::getNullValue(type_info->mem_type));
        auto dimension = llvm::cast<llvm::FixedVectorType>(type_info->reg_type)->getNumElements();
        for (auto i = 0u; i < dimension; i++) {
            result = builder.CreateInsertValue(result, builder.CreateExtractElement(value, i), i);
        }
        return result;
    }
    if (type_info->reg_type->isArrayTy()) {
        auto result = static_cast<llvm::Value *>(llvm::Constant::getNullValue(type_info->mem_type));
        auto dimension = type->dimension();
        auto element = type->is_matrix() ? Type::vector(type->element(), dimension) : type->element();
        for (auto i = 0u; i < dimension; i++) {
            auto item = _reg_to_mem(builder, builder.CreateExtractValue(value, i), element);
            result = builder.CreateInsertValue(result, item, i);
        }
        return result;
    }
    LUISA_ASSERT(type_info->reg_type->isStructTy(), "Invalid aggregate register type.");
    // Memory aggregate types contain explicit byte arrays for ABI padding.
    // Start from zero so bitwise operations such as PACK never observe poison
    // in those padding bytes (notably scalar bool/byte/short wrappers).
    auto result = static_cast<llvm::Value *>(llvm::Constant::getNullValue(type_info->mem_type));
    for (auto i = 0u; i < type->members().size(); i++) {
        auto member = _reg_to_mem(builder, builder.CreateExtractValue(value, i), type->members()[i]);
        result = builder.CreateInsertValue(result, member, type_info->member_indices[i]);
    }
    return result;
}

llvm::Value *MetalCodegenLLVMImpl::_mem_to_reg(IB &builder, llvm::Value *value, const Type *type) noexcept {
    auto type_info = _type(type);
    LUISA_ASSERT(value->getType() == type_info->mem_type, "Invalid memory value type in Metal LLVM codegen.");
    if (type_info->reg_type == type_info->mem_type) { return value; }
    if (type_info->reg_type->isVectorTy()) {
        auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(type_info->reg_type));
        auto dimension = llvm::cast<llvm::FixedVectorType>(type_info->reg_type)->getNumElements();
        for (auto i = 0u; i < dimension; i++) {
            result = builder.CreateInsertElement(result, builder.CreateExtractValue(value, i), i);
        }
        return result;
    }
    if (type_info->reg_type->isArrayTy()) {
        auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(type_info->reg_type));
        auto dimension = type->dimension();
        auto element = type->is_matrix() ? Type::vector(type->element(), dimension) : type->element();
        for (auto i = 0u; i < dimension; i++) {
            auto item = _mem_to_reg(builder, builder.CreateExtractValue(value, i), element);
            result = builder.CreateInsertValue(result, item, i);
        }
        return result;
    }
    LUISA_ASSERT(type_info->reg_type->isStructTy(), "Invalid aggregate memory type.");
    auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(type_info->reg_type));
    for (auto i = 0u; i < type->members().size(); i++) {
        auto member = _mem_to_reg(builder, builder.CreateExtractValue(value, type_info->member_indices[i]), type->members()[i]);
        result = builder.CreateInsertValue(result, member, i);
    }
    return result;
}

llvm::Value *MetalCodegenLLVMImpl::_load(IB &builder, llvm::Value *pointer, const Type *type, bool is_volatile) noexcept {
    auto type_info = _type(type);
    auto load = builder.CreateAlignedLoad(type_info->mem_type, pointer, llvm::Align{_type_alignment(type)});
    load->setVolatile(is_volatile);
    return _mem_to_reg(builder, load, type);
}

void MetalCodegenLLVMImpl::_store(IB &builder, llvm::Value *pointer, llvm::Value *value, const Type *type, bool is_volatile) noexcept {
    auto store = builder.CreateAlignedStore(_reg_to_mem(builder, value, type), pointer, llvm::Align{_type_alignment(type)});
    store->setVolatile(is_volatile);
}

llvm::Value *MetalCodegenLLVMImpl::_temporary(const FunctionContext &function, llvm::Type *type, size_t alignment) noexcept {
    IB builder{function.alloca_block->getTerminator()};
    auto allocation = builder.CreateAlloca(type);
    allocation->setAlignment(llvm::Align{alignment});
    return allocation;
}

llvm::Value *MetalCodegenLLVMImpl::_literal(IB &builder, const Type *type, const void *data) noexcept {
    if (data == nullptr) { return llvm::Constant::getNullValue(_type(type)->reg_type); }
    auto scalar = [data](auto value) noexcept {
        std::memcpy(&value, data, sizeof(value));
        return value;
    };
    switch (type->tag()) {
        case Type::Tag::BOOL: return builder.getInt1(scalar(bool{}));
        case Type::Tag::INT8: [[fallthrough]];
        case Type::Tag::UINT8: return builder.getInt8(scalar(uint8_t{}));
        case Type::Tag::INT16: [[fallthrough]];
        case Type::Tag::UINT16: return builder.getInt16(scalar(uint16_t{}));
        case Type::Tag::INT32: [[fallthrough]];
        case Type::Tag::UINT32: return builder.getInt32(scalar(uint32_t{}));
        case Type::Tag::INT64: [[fallthrough]];
        case Type::Tag::UINT64: return builder.getInt64(scalar(uint64_t{}));
        case Type::Tag::FLOAT16: return llvm::ConstantFP::get(builder.getHalfTy(), scalar(luisa::half{}));
        case Type::Tag::FLOAT32: return llvm::ConstantFP::get(builder.getFloatTy(), scalar(float{}));
        case Type::Tag::VECTOR: {
            llvm::SmallVector<llvm::Constant *> elements;
            auto element_type = type->element();
            for (auto i = 0u; i < type->dimension(); i++) {
                auto element_data = static_cast<const std::byte *>(data) + i * element_type->size();
                elements.emplace_back(llvm::cast<llvm::Constant>(_literal(builder, element_type, element_data)));
            }
            return llvm::ConstantVector::get(elements);
        }
        case Type::Tag::MATRIX: {
            llvm::SmallVector<llvm::Constant *> columns;
            auto column_type = Type::vector(type->element(), type->dimension());
            for (auto i = 0u; i < type->dimension(); i++) {
                auto column_data = static_cast<const std::byte *>(data) + i * column_type->size();
                columns.emplace_back(llvm::cast<llvm::Constant>(_literal(builder, column_type, column_data)));
            }
            return llvm::ConstantArray::get(llvm::cast<llvm::ArrayType>(_type(type)->reg_type), columns);
        }
        case Type::Tag::ARRAY: {
            llvm::SmallVector<llvm::Constant *> elements;
            auto element_type = type->element();
            for (auto i = 0u; i < type->dimension(); i++) {
                auto element_data = static_cast<const std::byte *>(data) + i * element_type->size();
                elements.emplace_back(llvm::cast<llvm::Constant>(_literal(builder, element_type, element_data)));
            }
            return llvm::ConstantArray::get(llvm::cast<llvm::ArrayType>(_type(type)->reg_type), elements);
        }
        case Type::Tag::STRUCTURE: {
            llvm::SmallVector<llvm::Constant *> members;
            auto type_info = _type(type);
            for (auto i = 0u; i < type->members().size(); i++) {
                auto member_data = static_cast<const std::byte *>(data) + type_info->member_offsets[i];
                members.emplace_back(llvm::cast<llvm::Constant>(_literal(builder, type->members()[i], member_data)));
            }
            return llvm::ConstantStruct::get(llvm::cast<llvm::StructType>(type_info->reg_type), members);
        }
        default: _unsupported_type(type);
    }
}

llvm::Value *MetalCodegenLLVMImpl::_constant(IB &builder, const xir::Constant *constant) noexcept {
    if (auto iter = _constants.find(constant); iter != _constants.end()) { return iter->second; }
    auto value = llvm::cast<llvm::Constant>(_literal(builder, constant->type(), constant->data()));
    _constants.try_emplace(constant, value);
    return value;
}

llvm::Value *MetalCodegenLLVMImpl::_special_register(const FunctionContext &function, xir::DerivedSpecialRegisterTag tag) noexcept {
    if (_config.program != MetalAIRProgram::COMPUTE) {
        switch (tag) {
            case xir::DerivedSpecialRegisterTag::KERNEL_ID: return function.kernel_id;
            case xir::DerivedSpecialRegisterTag::RASTER_OBJECT_ID: return function.raster_object_id;
            case xir::DerivedSpecialRegisterTag::RASTER_BARYCENTRICS: return function.raster_barycentrics;
            case xir::DerivedSpecialRegisterTag::RASTER_FRONT_FACING: return function.raster_front_facing;
            case xir::DerivedSpecialRegisterTag::RASTER_BASE_INSTANCE: return function.raster_base_instance;
            default:
                LUISA_ERROR_WITH_LOCATION(
                    "Metal raster AIR codegen does not support compute special register '{}'.",
                    xir::to_string(tag));
        }
    }
    switch (tag) {
        case xir::DerivedSpecialRegisterTag::THREAD_ID: return function.thread_id;
        case xir::DerivedSpecialRegisterTag::BLOCK_ID: return function.block_id;
        case xir::DerivedSpecialRegisterTag::WARP_LANE_ID: return function.warp_lane_id;
        case xir::DerivedSpecialRegisterTag::DISPATCH_ID: return function.dispatch_id;
        case xir::DerivedSpecialRegisterTag::KERNEL_ID: return function.kernel_id;
        case xir::DerivedSpecialRegisterTag::BLOCK_SIZE: return function.block_size;
        case xir::DerivedSpecialRegisterTag::WARP_SIZE: return function.warp_size;
        case xir::DerivedSpecialRegisterTag::DISPATCH_SIZE: return function.dispatch_size;
        case xir::DerivedSpecialRegisterTag::RASTER_OBJECT_ID: [[fallthrough]];
        case xir::DerivedSpecialRegisterTag::RASTER_BARYCENTRICS: [[fallthrough]];
        case xir::DerivedSpecialRegisterTag::RASTER_FRONT_FACING: [[fallthrough]];
        case xir::DerivedSpecialRegisterTag::RASTER_BASE_INSTANCE:
            LUISA_ERROR_WITH_LOCATION("Metal compute AIR codegen does not support raster special register '{}'.", xir::to_string(tag));
    }
    LUISA_ERROR_WITH_LOCATION("Invalid XIR special register '{}'.", xir::to_string(tag));
}

llvm::Value *MetalCodegenLLVMImpl::_value(IB &builder, const FunctionContext &function, const xir::Value *value) noexcept {
    LUISA_ASSERT(value != nullptr, "Cannot translate a null XIR value.");
    switch (value->derived_value_tag()) {
        case xir::DerivedValueTag::UNDEFINED: return llvm::PoisonValue::get(_type(value->type())->reg_type);
        case xir::DerivedValueTag::FUNCTION: return _function(static_cast<const xir::Function *>(value));
        case xir::DerivedValueTag::BASIC_BLOCK: [[fallthrough]];
        case xir::DerivedValueTag::INSTRUCTION: [[fallthrough]];
        case xir::DerivedValueTag::ARGUMENT: return function.value(value);
        case xir::DerivedValueTag::CONSTANT: return _constant(builder, static_cast<const xir::Constant *>(value));
        case xir::DerivedValueTag::SPECIAL_REGISTER: {
            auto special = static_cast<const xir::SpecialRegister *>(value);
            return _special_register(function, special->derived_special_register_tag());
        }
    }
    LUISA_ERROR_WITH_LOCATION("Unsupported XIR value kind in Metal AIR LLVM codegen.");
}

}// namespace luisa::compute::metal::detail
