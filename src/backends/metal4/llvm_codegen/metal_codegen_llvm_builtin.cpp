#include "metal_codegen_llvm_builtin.h"

#include "metal_codegen_llvm_impl.h"

#include <array>
#include <string>

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>

namespace luisa::compute::metal {
namespace {

using IB = llvm::IRBuilder<>;

class MetalBuiltinLLVMCodegen {

private:
    struct AccelTypes {
        llvm::StructType *instance;
        llvm::StructType *modification;
    };

    struct BindlessTypes {
        llvm::StructType *texture_2d_handle;
        llvm::StructType *texture_3d_handle;
        llvm::StructType *texture_2d;
        llvm::StructType *texture_3d;
        llvm::StructType *sampler;
        llvm::StructType *slot;
        llvm::StructType *buffer_modification;
        llvm::StructType *texture_2d_modification;
        llvm::StructType *texture_3d_modification;
        llvm::StructType *modification;
    };

    struct IndirectTypes {
        llvm::StructType *command_buffer_handle;
        llvm::StructType *pipeline_state_handle;
        llvm::StructType *command_buffer;
        llvm::StructType *pipeline_state;
        llvm::StructType *icb;
        llvm::StructType *header;
        llvm::StructType *slot;
    };

private:
    MetalCodegenLLVMConfig _config;
    MetalCodegenLLVMResult _result;
    llvm::LLVMContext &_context;
    llvm::Module &_module;
    llvm::DataLayout _data_layout;

private:
    [[nodiscard]] llvm::Metadata *_md_i32(uint32_t value) noexcept {
        return detail::md_i32(_context, value);
    }

    [[nodiscard]] llvm::Metadata *_md_string(
        luisa::string_view value) noexcept {
        return detail::md_string(_context, value);
    }

    [[nodiscard]] llvm::MDNode *_node(
        llvm::ArrayRef<llvm::Metadata *> operands) noexcept {
        return llvm::MDNode::get(_context, operands);
    }

    [[nodiscard]] llvm::PointerType *_pointer(unsigned address_space) noexcept {
        return llvm::PointerType::get(_context, address_space);
    }

    void _set_struct_pointer_element_type(
        llvm::StructType *structure, unsigned field,
        llvm::Type *element) noexcept {
        auto metadata = _node({llvm::MDString::get(_context, structure->getName()),
                               _md_i32(field),
                               llvm::ValueAsMetadata::get(llvm::UndefValue::get(element))});
        _module.getOrInsertNamedMetadata(
                   "llvm.struct_eltypes")
            ->addOperand(metadata);
    }

    void _set_pointer_element_types(
        llvm::Function *function,
        llvm::ArrayRef<std::pair<unsigned, llvm::Type *>> arguments,
        llvm::Type *return_element = nullptr) noexcept {
        if (!arguments.empty()) {
            llvm::SmallVector<llvm::Metadata *> metadata;
            metadata.reserve(arguments.size() * 2u);
            for (auto [index, element] : arguments) {
                metadata.emplace_back(_md_i32(index));
                metadata.emplace_back(llvm::ValueAsMetadata::get(
                    llvm::UndefValue::get(element)));
            }
            function->setMetadata(
                "arg_eltypes", llvm::MDNode::get(_context, metadata));
        }
        if (return_element != nullptr) {
            function->setMetadata(
                "ret_eltype",
                _node({llvm::ValueAsMetadata::get(
                    llvm::UndefValue::get(return_element))}));
        }
    }

    void _set_fast_math_attributes(llvm::Function *function) noexcept {
        function->setMustProgress();
        function->setDoesNotThrow();
        function->setWillReturn();
        function->addFnAttr("no-builtins");
        function->addFnAttr("frame-pointer", "all");
        function->addFnAttr("stack-protector-buffer-size", "8");
        auto value = _config.enable_fast_math ? "true" : "false";
        function->addFnAttr("approx-func-fp-math", value);
        function->addFnAttr("no-infs-fp-math", value);
        function->addFnAttr("no-nans-fp-math", value);
        function->addFnAttr("no-signed-zeros-fp-math", value);
        function->addFnAttr("unsafe-fp-math", value);
    }

    [[nodiscard]] llvm::Function *_create_function(
        luisa::string_view name, llvm::Type *return_type,
        llvm::ArrayRef<llvm::Type *> arguments) noexcept {
        auto type = llvm::FunctionType::get(
            return_type, arguments, false);
        auto function = llvm::Function::Create(
            type, llvm::GlobalValue::ExternalLinkage,
            llvm::StringRef{name.data(), name.size()}, _module);
        _set_fast_math_attributes(function);
        return function;
    }

    template<typename F>
    void _emit_if(IB &builder, llvm::Value *condition,
                  luisa::string_view name, F &&body) noexcept {
        auto function = builder.GetInsertBlock()->getParent();
        auto then_name = std::string{name} + ".then";
        auto end_name = std::string{name} + ".end";
        auto then_block = llvm::BasicBlock::Create(
            _context, then_name, function);
        auto end_block = llvm::BasicBlock::Create(
            _context, end_name, function);
        builder.CreateCondBr(condition, then_block, end_block);
        builder.SetInsertPoint(then_block);
        body(builder);
        if (builder.GetInsertBlock()->getTerminator() == nullptr) {
            builder.CreateBr(end_block);
        }
        builder.SetInsertPoint(end_block);
    }

    void _add_module_metadata() noexcept {
        auto sdk_values = std::array<uint32_t, 2u>{
            _config.sdk_version.major,
            _config.sdk_version.minor};
        auto sdk = llvm::ConstantDataArray::get(
            _context, sdk_values);
        _module.addModuleFlag(
            llvm::Module::ModFlagBehavior::Warning,
            "SDK Version", llvm::ConstantAsMetadata::get(sdk));
        _module.addModuleFlag(
            llvm::Module::ModFlagBehavior::Error, "wchar_size", 4u);
        _module.addModuleFlag(
            llvm::Module::ModFlagBehavior::Max, "frame-pointer", 2u);
        _module.addModuleFlag(
            llvm::Module::ModFlagBehavior::Max,
            "air.max_device_buffers", 31u);
        _module.addModuleFlag(
            llvm::Module::ModFlagBehavior::Max,
            "air.max_constant_buffers", 31u);
        _module.addModuleFlag(
            llvm::Module::ModFlagBehavior::Max,
            "air.max_threadgroup_buffers", 31u);
        _module.addModuleFlag(
            llvm::Module::ModFlagBehavior::Max,
            "air.max_textures", 128u);
        _module.addModuleFlag(
            llvm::Module::ModFlagBehavior::Max,
            "air.max_read_write_textures", 8u);
        _module.addModuleFlag(
            llvm::Module::ModFlagBehavior::Max,
            "air.max_samplers", 16u);
        _module.getOrInsertNamedMetadata("llvm.ident")
            ->addOperand(_node({_md_string(
                "LuisaCompute Metal4 runtime builtin LLVM codegen")}));
        _module.getOrInsertNamedMetadata("air.version")
            ->addOperand(_node({_md_i32(_config.air_version.major),
                                _md_i32(_config.air_version.minor),
                                _md_i32(_config.air_version.patch)}));
        _module.getOrInsertNamedMetadata("air.language_version")
            ->addOperand(_node({_md_string("Metal"),
                                _md_i32(_config.metal_version.major),
                                _md_i32(_config.metal_version.minor),
                                _md_i32(_config.metal_version.patch)}));
        auto compile_options =
            _module.getOrInsertNamedMetadata("air.compile_options");
        compile_options->addOperand(_node({_md_string("air.compile.denorms_disable")}));
        compile_options->addOperand(_node({_md_string(
            _config.enable_fast_math ?
                "air.compile.fast_math_enable" :
                "air.compile.fast_math_disable")}));
        compile_options->addOperand(_node({_md_string("air.compile.framebuffer_fetch_enable")}));
        if (!_config.source_file.empty()) {
            _module.getOrInsertNamedMetadata("air.source_file_name")
                ->addOperand(_node({_md_string(_config.source_file)}));
        }
    }

    [[nodiscard]] AccelTypes _accel_types() noexcept {
        auto f32 = llvm::Type::getFloatTy(_context);
        auto i8 = llvm::Type::getInt8Ty(_context);
        auto i32 = llvm::Type::getInt32Ty(_context);
        auto i64 = llvm::Type::getInt64Ty(_context);
        auto f32x4 = llvm::FixedVectorType::get(f32, 4u);
        auto instance = llvm::StructType::create(
            _context,
            {llvm::ArrayType::get(f32, 12u),
             i32, i32, i32, i32, i64},
            "luisa.metal4.builtin.accel.instance");
        auto modification = llvm::StructType::create(
            _context,
            {i32, i32, i32, i32,
             llvm::ArrayType::get(f32x4, 3u),
             i64, llvm::ArrayType::get(i8, 8u)},
            "luisa.metal4.builtin.accel.modification");
        return {instance, modification};
    }

    [[nodiscard]] BindlessTypes _bindless_types() noexcept {
        auto i8 = llvm::Type::getInt8Ty(_context);
        auto i32 = llvm::Type::getInt32Ty(_context);
        auto i64 = llvm::Type::getInt64Ty(_context);
        auto device_pointer = _pointer(detail::air_address_space_device);
        auto texture_2d_handle = llvm::StructType::create(
            _context, "struct._texture_2d_t");
        auto texture_3d_handle = llvm::StructType::create(
            _context, "struct._texture_3d_t");
        auto texture_2d = llvm::StructType::create(
            _context, {device_pointer},
            "luisa.metal4.builtin.texture2d");
        auto texture_3d = llvm::StructType::create(
            _context, {device_pointer},
            "luisa.metal4.builtin.texture3d");
        _set_struct_pointer_element_type(
            texture_2d, 0u, texture_2d_handle);
        _set_struct_pointer_element_type(
            texture_3d, 0u, texture_3d_handle);
        auto sampler = llvm::StructType::create(
            _context, {i8, i8},
            "luisa.metal4.builtin.sampler");
        auto slot = llvm::StructType::create(
            _context,
            {device_pointer, i64, texture_2d, texture_3d},
            "luisa.metal4.builtin.bindless.slot");
        _set_struct_pointer_element_type(slot, 0u, i8);
        auto buffer_modification = llvm::StructType::create(
            _context, {device_pointer, i64, i32},
            "luisa.metal4.builtin.bindless.buffer.modification");
        _set_struct_pointer_element_type(
            buffer_modification, 0u, i8);
        auto texture_2d_modification = llvm::StructType::create(
            _context, {texture_2d, sampler, i32},
            "luisa.metal4.builtin.bindless.texture2d.modification");
        auto texture_3d_modification = llvm::StructType::create(
            _context, {texture_3d, sampler, i32},
            "luisa.metal4.builtin.bindless.texture3d.modification");
        auto modification = llvm::StructType::create(
            _context,
            {i64, buffer_modification,
             texture_2d_modification,
             texture_3d_modification},
            "luisa.metal4.builtin.bindless.modification");
        return {
            texture_2d_handle, texture_3d_handle,
            texture_2d, texture_3d, sampler, slot,
            buffer_modification,
            texture_2d_modification,
            texture_3d_modification, modification};
    }

    [[nodiscard]] IndirectTypes _indirect_types() noexcept {
        auto i8 = llvm::Type::getInt8Ty(_context);
        auto i32 = llvm::Type::getInt32Ty(_context);
        auto device_pointer = _pointer(detail::air_address_space_device);
        auto command_buffer_handle = llvm::StructType::create(
            _context, "struct._command_buffer_t");
        auto pipeline_state_handle = llvm::StructType::create(
            _context, "struct._compute_pipeline_state_t");
        auto command_buffer = llvm::StructType::create(
            _context, {device_pointer},
            "luisa.metal4.builtin.command_buffer");
        auto pipeline_state = llvm::StructType::create(
            _context, {device_pointer},
            "luisa.metal4.builtin.compute_pipeline_state");
        _set_struct_pointer_element_type(
            command_buffer, 0u, command_buffer_handle);
        _set_struct_pointer_element_type(
            pipeline_state, 0u, pipeline_state_handle);
        auto icb = llvm::StructType::create(
            _context,
            {device_pointer, i32, i32,
             command_buffer, pipeline_state},
            "luisa.metal4.builtin.icb");
        _set_struct_pointer_element_type(icb, 0u, i8);
        auto header = llvm::StructType::create(
            _context,
            {i32, llvm::ArrayType::get(i8, 12u)},
            "luisa.metal4.builtin.icb.header");
        auto slot = llvm::StructType::create(
            _context,
            {llvm::FixedVectorType::get(i32, 3u),
             llvm::FixedVectorType::get(i32, 4u)},
            "luisa.metal4.builtin.icb.slot");
        return {
            command_buffer_handle, pipeline_state_handle,
            command_buffer, pipeline_state, icb, header, slot};
    }

    void _add_update_accel_metadata(
        llvm::Function *function) noexcept;
    void _add_update_bindless_metadata(
        llvm::Function *function,
        llvm::GlobalVariable *sampler = nullptr) noexcept;
    void _add_prepare_indirect_metadata(
        llvm::Function *function) noexcept;
    void _add_swapchain_vertex_metadata(
        llvm::Function *function) noexcept;
    void _add_swapchain_fragment_metadata(
        llvm::Function *function,
        llvm::GlobalVariable *sampler) noexcept;

    void _build_update_accel_instances() noexcept {
        auto types = _accel_types();
        auto i32 = llvm::Type::getInt32Ty(_context);
        auto i64 = llvm::Type::getInt64Ty(_context);
        auto function = _create_function(
            "update_accel_instances",
            llvm::Type::getVoidTy(_context),
            {_pointer(detail::air_address_space_device),
             _pointer(detail::air_address_space_device),
             _pointer(detail::air_address_space_constant), i32});
        std::array pointer_types{
            std::pair<unsigned, llvm::Type *>{0u, types.instance},
            std::pair<unsigned, llvm::Type *>{1u, types.modification},
            std::pair<unsigned, llvm::Type *>{2u, i32}};
        _set_pointer_element_types(function, pointer_types);
        auto instances = function->getArg(0u);
        auto modifications = function->getArg(1u);
        auto count_pointer = function->getArg(2u);
        auto tid = function->getArg(3u);
        instances->setName("instances");
        modifications->setName("modifications");
        count_pointer->setName("count");
        tid->setName("tid");

        auto entry = llvm::BasicBlock::Create(
            _context, "entry", function);
        auto body = llvm::BasicBlock::Create(
            _context, "body", function);
        auto exit = llvm::BasicBlock::Create(
            _context, "exit", function);
        IB builder{entry};
        auto count = builder.CreateAlignedLoad(
            i32, count_pointer, llvm::Align{4u}, "count.value");
        builder.CreateCondBr(
            builder.CreateICmpULT(tid, count), body, exit);
        builder.SetInsertPoint(body);
        auto tid64 = builder.CreateZExt(tid, i64);
        auto modification = builder.CreateInBoundsGEP(
            types.modification, modifications, tid64,
            "modification");
        auto index = builder.CreateLoad(
            i32, builder.CreateStructGEP(types.modification, modification, 0u),
            "instance.index");
        auto user_id = builder.CreateLoad(
            i32, builder.CreateStructGEP(types.modification, modification, 1u),
            "user.id");
        auto flags = builder.CreateLoad(
            i32, builder.CreateStructGEP(types.modification, modification, 2u),
            "flags");
        auto visibility = builder.CreateLoad(
            i32, builder.CreateStructGEP(types.modification, modification, 3u),
            "visibility");
        auto instance = builder.CreateInBoundsGEP(
            types.instance, instances,
            builder.CreateZExt(index, i64), "instance");
        auto flag = [&](uint32_t value) noexcept {
            return builder.CreateICmpNE(
                builder.CreateAnd(flags, builder.getInt32(value)),
                builder.getInt32(0u));
        };

        _emit_if(builder, flag(1u), "primitive", [&](IB &b) noexcept {
            auto primitive = b.CreateLoad(
                i64, b.CreateStructGEP(
                         types.modification, modification, 5u));
            b.CreateStore(
                b.getInt32(0u),
                b.CreateStructGEP(types.instance, instance, 3u));
            b.CreateStore(
                primitive,
                b.CreateStructGEP(types.instance, instance, 5u));
        });
        _emit_if(builder, flag(16u), "visibility", [&](IB &b) noexcept {
            b.CreateStore(
                visibility,
                b.CreateStructGEP(types.instance, instance, 2u));
        });
        _emit_if(
            builder,
            builder.CreateICmpNE(
                builder.CreateAnd(flags, builder.getInt32(12u)),
                builder.getInt32(0u)),
            "opaque", [&](IB &b) noexcept {
                auto opaque_on = b.CreateICmpNE(
                    b.CreateAnd(flags, b.getInt32(4u)),
                    b.getInt32(0u));
                auto options = b.CreateSelect(
                    opaque_on, b.getInt32(5u), b.getInt32(9u));
                b.CreateStore(
                    options,
                    b.CreateStructGEP(types.instance, instance, 1u));
            });
        _emit_if(builder, flag(2u), "transform", [&](IB &b) noexcept {
            auto affine = b.CreateStructGEP(
                types.modification, modification, 4u);
            auto transform = b.CreateStructGEP(
                types.instance, instance, 0u);
            std::array<llvm::Value *, 3u> rows{};
            auto affine_type = llvm::cast<llvm::ArrayType>(
                types.modification->getElementType(4u));
            auto transform_type = llvm::cast<llvm::ArrayType>(
                types.instance->getElementType(0u));
            for (auto row = 0u; row < rows.size(); row++) {
                auto row_pointer = b.CreateInBoundsGEP(
                    affine_type, affine,
                    {b.getInt32(0u), b.getInt32(row)});
                rows[row] = b.CreateLoad(
                    affine_type->getElementType(), row_pointer);
            }
            for (auto column = 0u; column < 4u; column++) {
                for (auto row = 0u; row < rows.size(); row++) {
                    auto value = b.CreateExtractElement(
                        rows[row], b.getInt32(column));
                    auto destination = b.CreateInBoundsGEP(
                        transform_type, transform,
                        {b.getInt32(0u),
                         b.getInt32(column * 3u + row)});
                    b.CreateStore(value, destination);
                }
            }
        });
        _emit_if(builder, flag(32u), "user_id", [&](IB &b) noexcept {
            b.CreateStore(
                user_id,
                b.CreateStructGEP(types.instance, instance, 4u));
        });
        builder.CreateBr(exit);
        builder.SetInsertPoint(exit);
        builder.CreateRetVoid();
        _add_update_accel_metadata(function);
    }

    void _build_update_bindless_array() noexcept;
    void _build_prepare_indirect_dispatches() noexcept;
    void _build_swapchain_vertex() noexcept;
    void _build_swapchain_fragment() noexcept;

public:
    explicit MetalBuiltinLLVMCodegen(
        MetalCodegenLLVMConfig config) noexcept
        : _config{std::move(config)},
          _result{},
          _context{*(_result.context =
                         std::make_unique<llvm::LLVMContext>())},
          _module{*(_result.module = std::make_unique<llvm::Module>(
                        "luisa.metal4.runtime.builtin", _context))},
          _data_layout{detail::air_data_layout} {
        _module.setDataLayout(_data_layout);
        _module.setTargetTriple(
            llvm::Triple{detail::air_target_triple(_config)});
        _module.setSourceFileName(
            std::string_view{_config.source_file});
    }

    [[nodiscard]] MetalCodegenLLVMResult generate(
        MetalBuiltinLLVMProgram program) noexcept {
        switch (program) {
            case MetalBuiltinLLVMProgram::UPDATE_ACCEL_INSTANCES:
                _build_update_accel_instances();
                break;
            case MetalBuiltinLLVMProgram::UPDATE_BINDLESS_ARRAY:
                _build_update_bindless_array();
                break;
            case MetalBuiltinLLVMProgram::PREPARE_INDIRECT_DISPATCHES:
                _build_prepare_indirect_dispatches();
                break;
            case MetalBuiltinLLVMProgram::SWAPCHAIN_VERTEX:
                _build_swapchain_vertex();
                break;
            case MetalBuiltinLLVMProgram::SWAPCHAIN_FRAGMENT:
                _build_swapchain_fragment();
                break;
        }
        _add_module_metadata();
        return std::move(_result);
    }
};

void MetalBuiltinLLVMCodegen::_build_update_bindless_array() noexcept {
    auto types = _bindless_types();
    auto i8 = llvm::Type::getInt8Ty(_context);
    auto i32 = llvm::Type::getInt32Ty(_context);
    auto i64 = llvm::Type::getInt64Ty(_context);
    auto device_pointer = _pointer(detail::air_address_space_device);
    auto function = _create_function(
        "update_bindless_array", llvm::Type::getVoidTy(_context),
        {device_pointer, device_pointer,
         _pointer(detail::air_address_space_constant), i32});
    std::array pointer_types{
        std::pair<unsigned, llvm::Type *>{0u, types.slot},
        std::pair<unsigned, llvm::Type *>{1u, types.modification},
        std::pair<unsigned, llvm::Type *>{2u, i32}};
    _set_pointer_element_types(function, pointer_types);
    auto slots = function->getArg(0u);
    auto modifications = function->getArg(1u);
    auto count_pointer = function->getArg(2u);
    auto tid = function->getArg(3u);
    slots->setName("slots");
    modifications->setName("modifications");
    count_pointer->setName("count");
    tid->setName("tid");

    auto entry = llvm::BasicBlock::Create(
        _context, "entry", function);
    auto body = llvm::BasicBlock::Create(
        _context, "body", function);
    auto exit = llvm::BasicBlock::Create(
        _context, "exit", function);
    IB builder{entry};
    auto count = builder.CreateAlignedLoad(
        i32, count_pointer, llvm::Align{4u}, "count.value");
    builder.CreateCondBr(
        builder.CreateICmpULT(tid, count), body, exit);
    builder.SetInsertPoint(body);
    auto modification = builder.CreateInBoundsGEP(
        types.modification, modifications,
        builder.CreateZExt(tid, i64), "modification");
    auto slot_index = builder.CreateLoad(
        i64, builder.CreateStructGEP(types.modification, modification, 0u),
        "slot.index");
    auto slot = builder.CreateInBoundsGEP(
        types.slot, slots, slot_index, "slot");
    auto buffer_pointer = builder.CreateStructGEP(
        types.slot, slot, 0u, "slot.buffer");
    auto packed_pointer = builder.CreateStructGEP(
        types.slot, slot, 1u, "slot.packed");
    auto texture_2d_wrapper = builder.CreateStructGEP(
        types.slot, slot, 2u, "slot.texture2d.wrapper");
    auto texture_3d_wrapper = builder.CreateStructGEP(
        types.slot, slot, 3u, "slot.texture3d.wrapper");
    auto texture_2d_pointer = builder.CreateStructGEP(
        types.texture_2d, texture_2d_wrapper, 0u,
        "slot.texture2d");
    auto texture_3d_pointer = builder.CreateStructGEP(
        types.texture_3d, texture_3d_wrapper, 0u,
        "slot.texture3d");
    auto current_buffer = builder.CreateLoad(
        device_pointer, buffer_pointer, "buffer.current");
    llvm::Value *packed = builder.CreateLoad(
        i64, packed_pointer, "packed.current");
    auto current_texture_2d = builder.CreateLoad(
        device_pointer, texture_2d_pointer,
        "texture2d.current");
    auto current_texture_3d = builder.CreateLoad(
        device_pointer, texture_3d_pointer,
        "texture3d.current");

    auto buffer_modification = builder.CreateStructGEP(
        types.modification, modification, 1u,
        "buffer.modification");
    auto new_buffer = builder.CreateLoad(
        device_pointer,
        builder.CreateStructGEP(
            types.buffer_modification,
            buffer_modification, 0u),
        "buffer.new");
    auto new_buffer_size = builder.CreateLoad(
        i64, builder.CreateStructGEP(types.buffer_modification, buffer_modification, 1u),
        "buffer.size");
    auto buffer_op = builder.CreateLoad(
        i32, builder.CreateStructGEP(types.buffer_modification, buffer_modification, 2u),
        "buffer.op");
    auto buffer_update = builder.CreateICmpEQ(
        buffer_op, builder.getInt32(1u));
    auto buffer_remove = builder.CreateICmpEQ(
        buffer_op, builder.getInt32(2u));
    auto output_buffer = builder.CreateSelect(
        buffer_update, new_buffer,
        builder.CreateSelect(
            buffer_remove,
            llvm::ConstantPointerNull::get(device_pointer),
            current_buffer));
    constexpr auto size_mask = 0x0000ffffffffffffull;
    auto packed_buffer_update = builder.CreateOr(
        builder.CreateAnd(
            packed, builder.getInt64(~size_mask)),
        builder.CreateAnd(
            new_buffer_size, builder.getInt64(size_mask)));
    auto packed_buffer_remove = builder.CreateAnd(
        packed, builder.getInt64(~size_mask));
    packed = builder.CreateSelect(
        buffer_update, packed_buffer_update,
        builder.CreateSelect(
            buffer_remove, packed_buffer_remove, packed),
        "packed.after.buffer");

    auto texture_2d_modification = builder.CreateStructGEP(
        types.modification, modification, 2u,
        "texture2d.modification");
    auto new_texture_2d_wrapper = builder.CreateStructGEP(
        types.texture_2d_modification,
        texture_2d_modification, 0u);
    auto new_texture_2d = builder.CreateLoad(
        device_pointer,
        builder.CreateStructGEP(
            types.texture_2d, new_texture_2d_wrapper, 0u),
        "texture2d.new");
    auto sampler_2d = builder.CreateStructGEP(
        types.texture_2d_modification,
        texture_2d_modification, 1u);
    auto sampler_2d_filter = builder.CreateLoad(
        i8, builder.CreateStructGEP(
                types.sampler, sampler_2d, 0u));
    auto sampler_2d_address = builder.CreateLoad(
        i8, builder.CreateStructGEP(
                types.sampler, sampler_2d, 1u));
    auto texture_2d_op = builder.CreateLoad(
        i32, builder.CreateStructGEP(types.texture_2d_modification, texture_2d_modification, 2u),
        "texture2d.op");
    auto texture_2d_update = builder.CreateICmpEQ(
        texture_2d_op, builder.getInt32(1u));
    auto texture_2d_remove = builder.CreateICmpEQ(
        texture_2d_op, builder.getInt32(2u));
    auto null_texture_2d = _create_function(
        "air.get_null_texture_2d", device_pointer, {});
    _set_pointer_element_types(
        null_texture_2d, {}, types.texture_2d_handle);
    auto removed_texture_2d = builder.CreateCall(
        null_texture_2d, {}, "texture2d.null");
    auto output_texture_2d = builder.CreateSelect(
        texture_2d_update, new_texture_2d,
        builder.CreateSelect(
            texture_2d_remove,
            removed_texture_2d,
            current_texture_2d));
    auto sampler_2d_code = builder.CreateOr(
        builder.CreateShl(
            builder.CreateZExt(sampler_2d_filter, i64),
            builder.getInt64(2u)),
        builder.CreateZExt(sampler_2d_address, i64));
    constexpr auto sampler_2d_mask = 0x00ff000000000000ull;
    auto packed_texture_2d_update = builder.CreateOr(
        builder.CreateAnd(
            packed, builder.getInt64(~sampler_2d_mask)),
        builder.CreateShl(sampler_2d_code, 48u));
    auto packed_texture_2d_remove = builder.CreateAnd(
        packed, builder.getInt64(~sampler_2d_mask));
    packed = builder.CreateSelect(
        texture_2d_update, packed_texture_2d_update,
        builder.CreateSelect(
            texture_2d_remove,
            packed_texture_2d_remove, packed),
        "packed.after.texture2d");

    auto texture_3d_modification = builder.CreateStructGEP(
        types.modification, modification, 3u,
        "texture3d.modification");
    auto new_texture_3d_wrapper = builder.CreateStructGEP(
        types.texture_3d_modification,
        texture_3d_modification, 0u);
    auto new_texture_3d = builder.CreateLoad(
        device_pointer,
        builder.CreateStructGEP(
            types.texture_3d, new_texture_3d_wrapper, 0u),
        "texture3d.new");
    auto sampler_3d = builder.CreateStructGEP(
        types.texture_3d_modification,
        texture_3d_modification, 1u);
    auto sampler_3d_filter = builder.CreateLoad(
        i8, builder.CreateStructGEP(
                types.sampler, sampler_3d, 0u));
    auto sampler_3d_address = builder.CreateLoad(
        i8, builder.CreateStructGEP(
                types.sampler, sampler_3d, 1u));
    auto texture_3d_op = builder.CreateLoad(
        i32, builder.CreateStructGEP(types.texture_3d_modification, texture_3d_modification, 2u),
        "texture3d.op");
    auto texture_3d_update = builder.CreateICmpEQ(
        texture_3d_op, builder.getInt32(1u));
    auto texture_3d_remove = builder.CreateICmpEQ(
        texture_3d_op, builder.getInt32(2u));
    auto null_texture_3d = _create_function(
        "air.get_null_texture_3d", device_pointer, {});
    _set_pointer_element_types(
        null_texture_3d, {}, types.texture_3d_handle);
    auto removed_texture_3d = builder.CreateCall(
        null_texture_3d, {}, "texture3d.null");
    auto output_texture_3d = builder.CreateSelect(
        texture_3d_update, new_texture_3d,
        builder.CreateSelect(
            texture_3d_remove,
            removed_texture_3d,
            current_texture_3d));
    auto sampler_3d_code = builder.CreateOr(
        builder.CreateShl(
            builder.CreateZExt(sampler_3d_filter, i64),
            builder.getInt64(2u)),
        builder.CreateZExt(sampler_3d_address, i64));
    constexpr auto sampler_3d_mask = 0xff00000000000000ull;
    auto packed_texture_3d_update = builder.CreateOr(
        builder.CreateAnd(
            packed, builder.getInt64(~sampler_3d_mask)),
        builder.CreateShl(sampler_3d_code, 56u));
    auto packed_texture_3d_remove = builder.CreateAnd(
        packed, builder.getInt64(~sampler_3d_mask));
    packed = builder.CreateSelect(
        texture_3d_update, packed_texture_3d_update,
        builder.CreateSelect(
            texture_3d_remove,
            packed_texture_3d_remove, packed),
        "packed.after.texture3d");

    builder.CreateStore(output_buffer, buffer_pointer);
    builder.CreateStore(packed, packed_pointer);
    builder.CreateStore(output_texture_2d, texture_2d_pointer);
    builder.CreateStore(output_texture_3d, texture_3d_pointer);
    builder.CreateBr(exit);
    builder.SetInsertPoint(exit);
    builder.CreateRetVoid();
    _add_update_bindless_metadata(function);
}

void MetalBuiltinLLVMCodegen::_build_swapchain_vertex() noexcept {
    auto f32 = llvm::Type::getFloatTy(_context);
    auto i32 = llvm::Type::getInt32Ty(_context);
    auto f32x2 = llvm::FixedVectorType::get(f32, 2u);
    auto f32x4 = llvm::FixedVectorType::get(f32, 4u);
    auto output = llvm::StructType::get(
        _context, {f32x4, f32x2}, true);
    auto function = _create_function(
        "swapchain_vertex_shader", output,
        {_pointer(detail::air_address_space_constant), i32});
    std::array pointer_types{
        std::pair<unsigned, llvm::Type *>{0u, f32x2}};
    _set_pointer_element_types(function, pointer_types);
    auto input = function->getArg(0u);
    auto vertex_id = function->getArg(1u);
    input->setName("input");
    vertex_id->setName("vertex.id");
    auto entry = llvm::BasicBlock::Create(
        _context, "entry", function);
    IB builder{entry};
    if (_config.enable_fast_math) {
        llvm::FastMathFlags flags;
        flags.setFast();
        builder.setFastMathFlags(flags);
    }
    auto p = builder.CreateLoad(
        f32x2,
        builder.CreateInBoundsGEP(
            f32x2, input,
            builder.CreateZExt(vertex_id,
                               llvm::Type::getInt64Ty(_context))),
        "position.xy");
    llvm::Value *position = llvm::PoisonValue::get(f32x4);
    position = builder.CreateInsertElement(
        position,
        builder.CreateExtractElement(p, uint64_t{0u}),
        uint64_t{0u});
    position = builder.CreateInsertElement(
        position,
        builder.CreateExtractElement(p, uint64_t{1u}),
        uint64_t{1u});
    position = builder.CreateInsertElement(
        position, llvm::ConstantFP::get(f32, 0.0),
        uint64_t{2u});
    position = builder.CreateInsertElement(
        position, llvm::ConstantFP::get(f32, 1.0),
        uint64_t{3u});
    auto scale = llvm::ConstantVector::get({llvm::ConstantFP::get(f32, 0.5),
                                            llvm::ConstantFP::get(f32, -0.5)});
    auto bias = llvm::ConstantVector::getSplat(
        llvm::ElementCount::getFixed(2u),
        llvm::ConstantFP::get(f32, 0.5));
    auto uv_unclamped = builder.CreateFAdd(
        builder.CreateFMul(p, scale), bias);
    auto saturate = _create_function(
        "air.fast_saturate.v2f32", f32x2, {f32x2});
    saturate->setDoesNotAccessMemory();
    auto uv = builder.CreateCall(
        saturate, {uv_unclamped}, "uv");
    auto result = builder.CreateInsertValue(
        llvm::PoisonValue::get(output), position, 0u);
    result = builder.CreateInsertValue(result, uv, 1u);
    builder.CreateRet(result);
    _add_swapchain_vertex_metadata(function);
}

void MetalBuiltinLLVMCodegen::_build_swapchain_fragment() noexcept {
    auto types = _bindless_types();
    auto f32 = llvm::Type::getFloatTy(_context);
    auto i1 = llvm::Type::getInt1Ty(_context);
    auto i8 = llvm::Type::getInt8Ty(_context);
    auto i32 = llvm::Type::getInt32Ty(_context);
    auto i64 = llvm::Type::getInt64Ty(_context);
    auto f32x2 = llvm::FixedVectorType::get(f32, 2u);
    auto f32x4 = llvm::FixedVectorType::get(f32, 4u);
    auto i32x2 = llvm::FixedVectorType::get(i32, 2u);
    auto device_pointer = _pointer(detail::air_address_space_device);
    auto constant_pointer = _pointer(detail::air_address_space_constant);
    auto sampler_handle = llvm::StructType::create(
        _context, "struct._sampler_t");
    auto sampler_storage_type = llvm::ArrayType::get(i64, 2u);
    auto sampler_storage = llvm::ConstantArray::get(
        sampler_storage_type,
        {llvm::ConstantInt::get(i64, 34901797601020489ull),
         llvm::ConstantInt::get(i64, 0u)});
    auto sampler = new llvm::GlobalVariable(
        _module, sampler_storage_type, true,
        llvm::GlobalValue::InternalLinkage,
        sampler_storage, "__air_sampler_state", nullptr,
        llvm::GlobalValue::NotThreadLocal,
        detail::air_address_space_constant);
    sampler->setAlignment(llvm::Align{8u});

    auto function = _create_function(
        "swapchain_fragment_shader", f32x4,
        {f32x4, f32x2, device_pointer});
    std::array pointer_types{
        std::pair<unsigned, llvm::Type *>{
            2u, types.texture_2d_handle}};
    _set_pointer_element_types(function, pointer_types);
    auto position = function->getArg(0u);
    auto uv = function->getArg(1u);
    auto texture = function->getArg(2u);
    position->setName("position");
    uv->setName("uv");
    texture->setName("image");
    auto entry = llvm::BasicBlock::Create(
        _context, "entry", function);
    IB builder{entry};
    if (_config.enable_fast_math) {
        llvm::FastMathFlags flags;
        flags.setFast();
        builder.setFastMathFlags(flags);
    }
    auto sample_result_type = llvm::StructType::get(
        _context, {f32x4, i8});
    auto sample = _create_function(
        "air.sample_texture_2d.v4f32", sample_result_type,
        {device_pointer, constant_pointer, f32x2,
         i1, i32x2, i1, f32, f32, i32});
    std::array sample_pointer_types{
        std::pair<unsigned, llvm::Type *>{
            0u, types.texture_2d_handle},
        std::pair<unsigned, llvm::Type *>{
            1u, sampler_handle}};
    _set_pointer_element_types(sample, sample_pointer_types);
    sample->setOnlyReadsMemory();
    sample->setConvergent();
    auto sampled = builder.CreateCall(
        sample,
        {texture, sampler, uv,
         builder.getInt1(true),
         llvm::Constant::getNullValue(i32x2),
         builder.getInt1(false),
         llvm::ConstantFP::get(f32, 0.0),
         llvm::ConstantFP::get(f32, 0.0),
         builder.getInt32(0u)},
        "sample");
    sampled->setConvergent();
    auto color = builder.CreateExtractValue(sampled, 0u);
    color = builder.CreateInsertElement(
        color, llvm::ConstantFP::get(f32, 1.0), 3u);
    builder.CreateRet(color);
    _add_swapchain_fragment_metadata(function, sampler);
}

void MetalBuiltinLLVMCodegen::_build_prepare_indirect_dispatches() noexcept {
    auto types = _indirect_types();
    auto i1 = llvm::Type::getInt1Ty(_context);
    auto i8 = llvm::Type::getInt8Ty(_context);
    auto i32 = llvm::Type::getInt32Ty(_context);
    auto i64 = llvm::Type::getInt64Ty(_context);
    auto i1x3 = llvm::FixedVectorType::get(i1, 3u);
    auto i32x3 = llvm::FixedVectorType::get(i32, 3u);
    auto i32x4 = llvm::FixedVectorType::get(i32, 4u);
    auto device_pointer = _pointer(detail::air_address_space_device);
    auto constant_pointer = _pointer(detail::air_address_space_constant);
    auto function = _create_function(
        "prepare_indirect_dispatches",
        llvm::Type::getVoidTy(_context),
        {constant_pointer, constant_pointer, i32});
    std::array pointer_types{
        std::pair<unsigned, llvm::Type *>{0u, types.icb},
        std::pair<unsigned, llvm::Type *>{1u, i8}};
    _set_pointer_element_types(function, pointer_types);
    auto icb = function->getArg(0u);
    auto kernel_arguments = function->getArg(1u);
    auto tid = function->getArg(2u);
    icb->setName("icb");
    kernel_arguments->setName("kernel.arguments");
    tid->setName("tid");

    auto reset_command = _create_function(
        "air.reset_compute_command",
        llvm::Type::getVoidTy(_context),
        {device_pointer, i32});
    std::array reset_pointer_types{
        std::pair<unsigned, llvm::Type *>{
            0u, types.command_buffer_handle}};
    _set_pointer_element_types(reset_command, reset_pointer_types);
    auto set_pipeline = _create_function(
        "air.set_pipeline_state_compute_command",
        llvm::Type::getVoidTy(_context),
        {device_pointer, i32, device_pointer});
    std::array pipeline_pointer_types{
        std::pair<unsigned, llvm::Type *>{
            0u, types.command_buffer_handle},
        std::pair<unsigned, llvm::Type *>{
            2u, types.pipeline_state_handle}};
    _set_pointer_element_types(set_pipeline, pipeline_pointer_types);
    auto set_kernel_constant = _create_function(
        "air.set_kernel_buffer_compute_command.p2i8",
        llvm::Type::getVoidTy(_context),
        {device_pointer, i32, constant_pointer, i64, i32});
    std::array constant_pointer_types{
        std::pair<unsigned, llvm::Type *>{
            0u, types.command_buffer_handle},
        std::pair<unsigned, llvm::Type *>{2u, i8}};
    _set_pointer_element_types(
        set_kernel_constant, constant_pointer_types);
    auto set_kernel_device = _create_function(
        "air.set_kernel_buffer_compute_command.p1i8",
        llvm::Type::getVoidTy(_context),
        {device_pointer, i32, device_pointer, i64, i32});
    std::array device_pointer_types{
        std::pair<unsigned, llvm::Type *>{
            0u, types.command_buffer_handle},
        std::pair<unsigned, llvm::Type *>{2u, i8}};
    _set_pointer_element_types(
        set_kernel_device, device_pointer_types);
    auto dispatch = _create_function(
        "air.concurrent_dispatch_threadgroups_compute_command",
        llvm::Type::getVoidTy(_context),
        {device_pointer, i32, i32x3, i32x3});
    std::array dispatch_pointer_types{
        std::pair<unsigned, llvm::Type *>{
            0u, types.command_buffer_handle}};
    _set_pointer_element_types(dispatch, dispatch_pointer_types);
    auto all = _create_function(
        "air.all.v3i1", i1, {i1x3});
    all->setDoesNotAccessMemory();

    auto entry = llvm::BasicBlock::Create(
        _context, "entry", function);
    IB builder{entry};
    auto offset = builder.CreateLoad(
        i32, builder.CreateStructGEP(types.icb, icb, 1u),
        "offset");
    auto capacity = builder.CreateLoad(
        i32, builder.CreateStructGEP(types.icb, icb, 2u),
        "capacity");
    auto index = builder.CreateAdd(offset, tid, "index");
    _emit_if(
        builder, builder.CreateICmpULT(index, capacity),
        "in_capacity", [&](IB &capacity_builder) noexcept {
            auto command_wrapper = capacity_builder.CreateStructGEP(
                types.icb, icb, 3u);
            auto command_buffer = capacity_builder.CreateLoad(
                device_pointer,
                capacity_builder.CreateStructGEP(
                    types.command_buffer,
                    command_wrapper, 0u),
                "command.buffer");
            capacity_builder.CreateCall(
                reset_command, {command_buffer, index});
            auto dispatch_buffer = capacity_builder.CreateLoad(
                device_pointer,
                capacity_builder.CreateStructGEP(
                    types.icb, icb, 0u),
                "dispatch.buffer");
            auto count = capacity_builder.CreateLoad(
                i32,
                capacity_builder.CreateStructGEP(
                    types.header, dispatch_buffer, 0u),
                "dispatch.count");
            _emit_if(
                capacity_builder,
                capacity_builder.CreateICmpULT(tid, count),
                "in_count", [&](IB &count_builder) noexcept {
                    auto slots = count_builder.CreateInBoundsGEP(
                        types.header, dispatch_buffer,
                        count_builder.getInt64(1u), "slots");
                    auto slot = count_builder.CreateInBoundsGEP(
                        types.slot, slots,
                        count_builder.CreateZExt(index, i64),
                        "slot");
                    auto dispatch_size_4 = count_builder.CreateLoad(
                        i32x4,
                        count_builder.CreateStructGEP(
                            types.slot, slot, 1u),
                        "dispatch.size.and.kernel");
                    auto dispatch_size =
                        count_builder.CreateShuffleVector(
                            dispatch_size_4, {0, 1, 2},
                            "dispatch.size");
                    auto nonzero = count_builder.CreateCall(
                        all,
                        {count_builder.CreateICmpNE(
                            dispatch_size,
                            llvm::Constant::getNullValue(i32x3))},
                        "dispatch.nonzero");
                    _emit_if(
                        count_builder, nonzero,
                        "nonzero", [&](IB &dispatch_builder) noexcept {
                            auto block_size = dispatch_builder.CreateLoad(
                                i32x3,
                                dispatch_builder.CreateStructGEP(
                                    types.slot, slot, 0u),
                                "block.size");
                            auto pipeline_wrapper =
                                dispatch_builder.CreateStructGEP(
                                    types.icb, icb, 4u);
                            auto pipeline = dispatch_builder.CreateLoad(
                                device_pointer,
                                dispatch_builder.CreateStructGEP(
                                    types.pipeline_state,
                                    pipeline_wrapper, 0u),
                                "pipeline.state");
                            dispatch_builder.CreateCall(
                                set_pipeline,
                                {command_buffer, index, pipeline});
                            dispatch_builder.CreateCall(
                                set_kernel_constant,
                                {command_buffer, index,
                                 kernel_arguments,
                                 dispatch_builder.getInt64(~0ull),
                                 dispatch_builder.getInt32(0u)});
                            auto dispatch_size_pointer =
                                dispatch_builder.CreateStructGEP(
                                    types.slot, slot, 1u);
                            dispatch_builder.CreateCall(
                                set_kernel_device,
                                {command_buffer, index,
                                 dispatch_size_pointer,
                                 dispatch_builder.getInt64(~0ull),
                                 dispatch_builder.getInt32(1u)});
                            auto one = llvm::ConstantVector::getSplat(
                                llvm::ElementCount::getFixed(3u),
                                dispatch_builder.getInt32(1u));
                            auto block_count = dispatch_builder.CreateUDiv(
                                dispatch_builder.CreateAdd(
                                    dispatch_size,
                                    dispatch_builder.CreateSub(
                                        block_size, one)),
                                block_size, "block.count");
                            dispatch_builder.CreateCall(
                                dispatch,
                                {command_buffer, index,
                                 block_count, block_size});
                        });
                });
        });
    builder.CreateRetVoid();
    _add_prepare_indirect_metadata(function);
}

void MetalBuiltinLLVMCodegen::_add_update_accel_metadata(
    llvm::Function *function) noexcept {
    auto transform_info = _node({_md_i32(0u), _md_i32(4u), _md_i32(12u),
                                 _md_string("float"), _md_string("transform"),
                                 _md_i32(48u), _md_i32(4u), _md_i32(0u),
                                 _md_string("uint"), _md_string("options"),
                                 _md_i32(52u), _md_i32(4u), _md_i32(0u),
                                 _md_string("uint"), _md_string("mask"),
                                 _md_i32(56u), _md_i32(4u), _md_i32(0u),
                                 _md_string("uint"),
                                 _md_string("intersection_function_offset"),
                                 _md_i32(60u), _md_i32(4u), _md_i32(0u),
                                 _md_string("uint"), _md_string("user_id"),
                                 _md_i32(64u), _md_i32(8u), _md_i32(0u),
                                 _md_string("ulong"),
                                 _md_string("acceleration_structure_id")});
    auto modification_info = _node({_md_i32(0u), _md_i32(4u), _md_i32(0u),
                                    _md_string("uint"), _md_string("index"),
                                    _md_i32(4u), _md_i32(4u), _md_i32(0u),
                                    _md_string("uint"), _md_string("user_id"),
                                    _md_i32(8u), _md_i32(4u), _md_i32(0u),
                                    _md_string("uint"), _md_string("flags"),
                                    _md_i32(12u), _md_i32(4u), _md_i32(0u),
                                    _md_string("uint"), _md_string("vis_mask"),
                                    _md_i32(16u), _md_i32(16u), _md_i32(3u),
                                    _md_string("float4"), _md_string("affine"),
                                    _md_i32(64u), _md_i32(8u), _md_i32(0u),
                                    _md_string("ulong"), _md_string("primitive")});
    auto instances = _node({_md_i32(0u), _md_string("air.buffer"),
                            _md_string("air.location_index"), _md_i32(0u), _md_i32(1u),
                            _md_string("air.read_write"),
                            _md_string("air.address_space"), _md_i32(1u),
                            _md_string("air.struct_type_info"), transform_info,
                            _md_string("air.arg_type_size"), _md_i32(72u),
                            _md_string("air.arg_type_align_size"), _md_i32(8u),
                            _md_string("air.arg_type_name"), _md_string("AccelInstance"),
                            _md_string("air.arg_name"), _md_string("instances")});
    auto modifications = _node({_md_i32(1u), _md_string("air.buffer"),
                                _md_string("air.location_index"), _md_i32(1u), _md_i32(1u),
                                _md_string("air.read"),
                                _md_string("air.address_space"), _md_i32(1u),
                                _md_string("air.struct_type_info"), modification_info,
                                _md_string("air.arg_type_size"), _md_i32(80u),
                                _md_string("air.arg_type_align_size"), _md_i32(16u),
                                _md_string("air.arg_type_name"),
                                _md_string("AccelInstanceModification"),
                                _md_string("air.arg_name"), _md_string("mods")});
    auto count = _node({_md_i32(2u), _md_string("air.buffer"),
                        _md_string("air.buffer_size"), _md_i32(4u),
                        _md_string("air.location_index"), _md_i32(2u), _md_i32(1u),
                        _md_string("air.read"),
                        _md_string("air.address_space"), _md_i32(2u),
                        _md_string("air.arg_type_size"), _md_i32(4u),
                        _md_string("air.arg_type_align_size"), _md_i32(4u),
                        _md_string("air.arg_type_name"), _md_string("uint"),
                        _md_string("air.arg_name"), _md_string("n")});
    auto tid = _node({_md_i32(3u), _md_string("air.thread_position_in_grid"),
                      _md_string("air.arg_type_name"), _md_string("uint"),
                      _md_string("air.arg_name"), _md_string("tid")});
    _module.getOrInsertNamedMetadata("air.kernel")
        ->addOperand(_node({llvm::ValueAsMetadata::get(function), _node({}),
                            _node({instances, modifications, count, tid})}));
}

void MetalBuiltinLLVMCodegen::_add_update_bindless_metadata(
    llvm::Function *function,
    [[maybe_unused]] llvm::GlobalVariable *sampler) noexcept {
    auto buffer = _node({_md_i32(0u), _md_string("air.buffer"),
                         _md_string("air.location_index"), _md_i32(0u), _md_i32(1u),
                         _md_string("air.read_write"),
                         _md_string("air.address_space"), _md_i32(1u),
                         _md_string("air.arg_type_name"), _md_string("void"),
                         _md_string("air.arg_name"), _md_string("buffer")});
    auto buffer_size = _node({_md_i32(1u), _md_string("air.indirect_constant"),
                              _md_string("air.location_index"), _md_i32(1u), _md_i32(1u),
                              _md_string("air.arg_type_name"), _md_string("ulong"),
                              _md_string("air.arg_name"), _md_string("buffer_size")});
    auto sampler_2d = _node({_md_i32(2u), _md_string("air.indirect_constant"),
                             _md_string("air.location_index"), _md_i32(2u), _md_i32(1u),
                             _md_string("air.arg_type_name"), _md_string("uint"),
                             _md_string("air.arg_name"), _md_string("sampler2d")});
    auto sampler_3d = _node({_md_i32(3u), _md_string("air.indirect_constant"),
                             _md_string("air.location_index"), _md_i32(3u), _md_i32(1u),
                             _md_string("air.arg_type_name"), _md_string("uint"),
                             _md_string("air.arg_name"), _md_string("sampler3d")});
    auto texture_2d = _node({_md_i32(4u), _md_string("air.texture"),
                             _md_string("air.location_index"), _md_i32(4u), _md_i32(1u),
                             _md_string("air.sample"),
                             _md_string("air.arg_type_name"),
                             _md_string("texture2d<float, sample>"),
                             _md_string("air.arg_name"), _md_string("tex2d")});
    auto texture_3d = _node({_md_i32(5u), _md_string("air.texture"),
                             _md_string("air.location_index"), _md_i32(5u), _md_i32(1u),
                             _md_string("air.sample"),
                             _md_string("air.arg_type_name"),
                             _md_string("texture3d<float, sample>"),
                             _md_string("air.arg_name"), _md_string("tex3d")});
    auto slot_info = _node({_md_i32(0u), _md_i32(8u), _md_i32(0u),
                            _md_string("void"), _md_string("buffer"),
                            _md_string("air.indirect_argument"), buffer,
                            _md_i32(8u), _md_i32(8u), _md_i32(0u),
                            _md_string("ulong"), _md_string("buffer_size"),
                            _md_string("air.indirect_argument"), buffer_size,
                            _md_i32(14u), _md_i32(4u), _md_i32(0u),
                            _md_string("uint"), _md_string("sampler2d"),
                            _md_string("air.indirect_argument"), sampler_2d,
                            _md_i32(15u), _md_i32(4u), _md_i32(0u),
                            _md_string("uint"), _md_string("sampler3d"),
                            _md_string("air.indirect_argument"), sampler_3d,
                            _md_i32(16u), _md_i32(8u), _md_i32(0u),
                            _md_string("texture2d<float, sample>"), _md_string("tex2d"),
                            _md_string("air.indirect_argument"), texture_2d,
                            _md_i32(24u), _md_i32(8u), _md_i32(0u),
                            _md_string("texture3d<float, sample>"), _md_string("tex3d"),
                            _md_string("air.indirect_argument"), texture_3d});
    auto slots = _node({_md_i32(0u), _md_string("air.indirect_buffer"),
                        _md_string("air.location_index"), _md_i32(0u), _md_i32(1u),
                        _md_string("air.read_write"),
                        _md_string("air.address_space"), _md_i32(1u),
                        _md_string("air.struct_type_info"), slot_info,
                        _md_string("air.arg_type_size"), _md_i32(32u),
                        _md_string("air.arg_type_align_size"), _md_i32(16u),
                        _md_string("air.arg_type_name"), _md_string("BindlessSlot"),
                        _md_string("air.arg_name"), _md_string("slots")});

    auto modification_slot = _node({_md_i32(0u), _md_string("air.indirect_constant"),
                                    _md_string("air.location_index"), _md_i32(0u), _md_i32(1u),
                                    _md_string("air.arg_type_name"), _md_string("ulong"),
                                    _md_string("air.arg_name"), _md_string("slot")});
    auto modification_buffer_handle = _node({_md_i32(0u), _md_string("air.buffer"),
                                             _md_string("air.location_index"), _md_i32(0u), _md_i32(1u),
                                             _md_string("air.read_write"),
                                             _md_string("air.address_space"), _md_i32(1u),
                                             _md_string("air.arg_type_name"), _md_string("void"),
                                             _md_string("air.arg_name"), _md_string("handle")});
    auto modification_buffer_size = _node({_md_i32(1u), _md_string("air.indirect_constant"),
                                           _md_string("air.location_index"), _md_i32(1u), _md_i32(1u),
                                           _md_string("air.arg_type_name"), _md_string("ulong"),
                                           _md_string("air.arg_name"), _md_string("size")});
    auto modification_buffer_op = _node({_md_i32(2u), _md_string("air.indirect_constant"),
                                         _md_string("air.location_index"), _md_i32(2u), _md_i32(1u),
                                         _md_string("air.arg_type_name"), _md_string("uint"),
                                         _md_string("air.arg_name"), _md_string("op")});
    auto modification_buffer_info = _node({_md_i32(0u), _md_i32(8u), _md_i32(0u),
                                           _md_string("void"), _md_string("handle"),
                                           _md_string("air.indirect_argument"),
                                           modification_buffer_handle,
                                           _md_i32(8u), _md_i32(8u), _md_i32(0u),
                                           _md_string("ulong"), _md_string("size"),
                                           _md_string("air.indirect_argument"),
                                           modification_buffer_size,
                                           _md_i32(16u), _md_i32(4u), _md_i32(0u),
                                           _md_string("uint"), _md_string("op"),
                                           _md_string("air.indirect_argument"),
                                           modification_buffer_op});
    auto sampler_filter = _node({_md_i32(0u), _md_string("air.indirect_constant"),
                                 _md_string("air.location_index"), _md_i32(0u), _md_i32(1u),
                                 _md_string("air.arg_type_name"), _md_string("uchar"),
                                 _md_string("air.arg_name"), _md_string("filter")});
    auto sampler_address = _node({_md_i32(1u), _md_string("air.indirect_constant"),
                                  _md_string("air.location_index"), _md_i32(1u), _md_i32(1u),
                                  _md_string("air.arg_type_name"), _md_string("uchar"),
                                  _md_string("air.arg_name"), _md_string("address")});
    auto sampler_info = _node({_md_i32(0u), _md_i32(1u), _md_i32(0u),
                               _md_string("uchar"), _md_string("filter"),
                               _md_string("air.indirect_argument"), sampler_filter,
                               _md_i32(1u), _md_i32(1u), _md_i32(0u),
                               _md_string("uchar"), _md_string("address"),
                               _md_string("air.indirect_argument"), sampler_address});
    auto modification_texture_2d_handle = _node({_md_i32(0u), _md_string("air.texture"),
                                                 _md_string("air.location_index"), _md_i32(0u), _md_i32(1u),
                                                 _md_string("air.sample"),
                                                 _md_string("air.arg_type_name"),
                                                 _md_string("texture2d<float, sample>"),
                                                 _md_string("air.arg_name"), _md_string("handle")});
    auto modification_texture_3d_handle = _node({_md_i32(0u), _md_string("air.texture"),
                                                 _md_string("air.location_index"), _md_i32(0u), _md_i32(1u),
                                                 _md_string("air.sample"),
                                                 _md_string("air.arg_type_name"),
                                                 _md_string("texture3d<float, sample>"),
                                                 _md_string("air.arg_name"), _md_string("handle")});
    auto modification_texture_op = _node({_md_i32(2u), _md_string("air.indirect_constant"),
                                          _md_string("air.location_index"), _md_i32(3u), _md_i32(1u),
                                          _md_string("air.arg_type_name"), _md_string("uint"),
                                          _md_string("air.arg_name"), _md_string("op")});
    auto texture_modification_info = [&](llvm::MDNode *handle,
                                         luisa::string_view type) noexcept {
        return _node({_md_i32(0u), _md_i32(8u), _md_i32(0u),
                      _md_string(type), _md_string("handle"),
                      _md_string("air.indirect_argument"), handle,
                      _md_string("air.struct_type_info"), sampler_info,
                      _md_i32(8u), _md_i32(2u), _md_i32(0u),
                      _md_string("Sampler"), _md_string("sampler"),
                      _md_string("air.indirect_argument"), _md_i32(1u),
                      _md_i32(12u), _md_i32(4u), _md_i32(0u),
                      _md_string("uint"), _md_string("op"),
                      _md_string("air.indirect_argument"),
                      modification_texture_op});
    };
    auto texture_2d_modification_info = texture_modification_info(
        modification_texture_2d_handle,
        "texture2d<float, sample>");
    auto texture_3d_modification_info = texture_modification_info(
        modification_texture_3d_handle,
        "texture3d<float, sample>");
    auto modification_info = _node({_md_i32(0u), _md_i32(8u), _md_i32(0u),
                                    _md_string("ulong"), _md_string("slot"),
                                    _md_string("air.indirect_argument"), modification_slot,
                                    _md_string("air.struct_type_info"), modification_buffer_info,
                                    _md_i32(8u), _md_i32(24u), _md_i32(0u),
                                    _md_string("BindlessSlotModification::Buffer"),
                                    _md_string("buffer"),
                                    _md_string("air.indirect_argument"), _md_i32(1u),
                                    _md_string("air.struct_type_info"),
                                    texture_2d_modification_info,
                                    _md_i32(32u), _md_i32(16u), _md_i32(0u),
                                    _md_string("BindlessSlotModification::Texture2D"),
                                    _md_string("tex2d"),
                                    _md_string("air.indirect_argument"), _md_i32(4u),
                                    _md_string("air.struct_type_info"),
                                    texture_3d_modification_info,
                                    _md_i32(48u), _md_i32(16u), _md_i32(0u),
                                    _md_string("BindlessSlotModification::Texture3D"),
                                    _md_string("tex3d"),
                                    _md_string("air.indirect_argument"), _md_i32(8u)});
    auto modifications = _node({_md_i32(1u), _md_string("air.indirect_buffer"),
                                _md_string("air.location_index"), _md_i32(1u), _md_i32(1u),
                                _md_string("air.read"),
                                _md_string("air.address_space"), _md_i32(1u),
                                _md_string("air.struct_type_info"), modification_info,
                                _md_string("air.arg_type_size"), _md_i32(64u),
                                _md_string("air.arg_type_align_size"), _md_i32(16u),
                                _md_string("air.arg_type_name"),
                                _md_string("BindlessSlotModification"),
                                _md_string("air.arg_name"), _md_string("mods")});
    auto count = _node({_md_i32(2u), _md_string("air.buffer"),
                        _md_string("air.buffer_size"), _md_i32(4u),
                        _md_string("air.location_index"), _md_i32(2u), _md_i32(1u),
                        _md_string("air.read"),
                        _md_string("air.address_space"), _md_i32(2u),
                        _md_string("air.arg_type_size"), _md_i32(4u),
                        _md_string("air.arg_type_align_size"), _md_i32(4u),
                        _md_string("air.arg_type_name"), _md_string("uint"),
                        _md_string("air.arg_name"), _md_string("n")});
    auto tid = _node({_md_i32(3u), _md_string("air.thread_position_in_grid"),
                      _md_string("air.arg_type_name"), _md_string("uint"),
                      _md_string("air.arg_name"), _md_string("tid")});
    _module.getOrInsertNamedMetadata("air.kernel")
        ->addOperand(_node({llvm::ValueAsMetadata::get(function), _node({}),
                            _node({slots, modifications, count, tid})}));
}

void MetalBuiltinLLVMCodegen::_add_prepare_indirect_metadata(
    llvm::Function *function) noexcept {
    auto buffer = _node({_md_i32(0u), _md_string("air.buffer"),
                         _md_string("air.location_index"), _md_i32(0u), _md_i32(1u),
                         _md_string("air.read_write"),
                         _md_string("air.address_space"), _md_i32(1u),
                         _md_string("air.arg_type_name"), _md_string("void"),
                         _md_string("air.arg_name"), _md_string("buffer")});
    auto offset = _node({_md_i32(1u), _md_string("air.indirect_constant"),
                         _md_string("air.location_index"), _md_i32(1u), _md_i32(1u),
                         _md_string("air.arg_type_name"), _md_string("uint"),
                         _md_string("air.arg_name"), _md_string("offset")});
    auto capacity = _node({_md_i32(2u), _md_string("air.indirect_constant"),
                           _md_string("air.location_index"), _md_i32(2u), _md_i32(1u),
                           _md_string("air.arg_type_name"), _md_string("uint"),
                           _md_string("air.arg_name"), _md_string("capacity")});
    auto command_buffer = _node({_md_i32(3u), _md_string("air.command_buffer"),
                                 _md_string("air.location_index"), _md_i32(3u), _md_i32(1u),
                                 _md_string("air.arg_type_name"), _md_string("command_buffer"),
                                 _md_string("air.arg_name"), _md_string("command_buffer")});
    auto pipeline = _node({_md_i32(4u), _md_string("air.compute_pipeline_state"),
                           _md_string("air.location_index"), _md_i32(4u), _md_i32(1u),
                           _md_string("air.arg_type_name"),
                           _md_string("compute_pipeline_state"),
                           _md_string("air.arg_name"), _md_string("pipeline_state")});
    auto icb_info = _node({_md_i32(0u), _md_i32(8u), _md_i32(0u),
                           _md_string("void"), _md_string("buffer"),
                           _md_string("air.indirect_argument"), buffer,
                           _md_i32(8u), _md_i32(4u), _md_i32(0u),
                           _md_string("uint"), _md_string("offset"),
                           _md_string("air.indirect_argument"), offset,
                           _md_i32(12u), _md_i32(4u), _md_i32(0u),
                           _md_string("uint"), _md_string("capacity"),
                           _md_string("air.indirect_argument"), capacity,
                           _md_i32(16u), _md_i32(8u), _md_i32(0u),
                           _md_string("command_buffer"), _md_string("command_buffer"),
                           _md_string("air.indirect_argument"), command_buffer,
                           _md_i32(24u), _md_i32(8u), _md_i32(0u),
                           _md_string("compute_pipeline_state"),
                           _md_string("pipeline_state"),
                           _md_string("air.indirect_argument"), pipeline});
    auto icb = _node({_md_i32(0u), _md_string("air.indirect_buffer"),
                      _md_string("air.buffer_size"), _md_i32(32u),
                      _md_string("air.location_index"), _md_i32(0u), _md_i32(1u),
                      _md_string("air.read"),
                      _md_string("air.address_space"), _md_i32(2u),
                      _md_string("air.struct_type_info"), icb_info,
                      _md_string("air.arg_type_size"), _md_i32(32u),
                      _md_string("air.arg_type_align_size"), _md_i32(8u),
                      _md_string("air.arg_type_name"), _md_string("ICB"),
                      _md_string("air.arg_name"), _md_string("icb")});
    auto arguments = _node({_md_i32(1u), _md_string("air.buffer"),
                            _md_string("air.location_index"), _md_i32(1u), _md_i32(1u),
                            _md_string("air.read"),
                            _md_string("air.address_space"), _md_i32(2u),
                            _md_string("air.arg_type_name"), _md_string("void"),
                            _md_string("air.arg_name"), _md_string("kernel_args")});
    auto tid = _node({_md_i32(2u), _md_string("air.thread_position_in_grid"),
                      _md_string("air.arg_type_name"), _md_string("uint"),
                      _md_string("air.arg_name"), _md_string("tid")});
    _module.getOrInsertNamedMetadata("air.kernel")
        ->addOperand(_node({llvm::ValueAsMetadata::get(function), _node({}),
                            _node({icb, arguments, tid})}));
}

void MetalBuiltinLLVMCodegen::_add_swapchain_vertex_metadata(
    llvm::Function *function) noexcept {
    auto position = _node({_md_string("air.position"),
                           _md_string("air.arg_type_name"), _md_string("float4"),
                           _md_string("air.arg_name"), _md_string("p")});
    auto uv_output = _node({_md_string("air.vertex_output"), _md_string("generated(2uvDv2_f)"),
                            _md_string("air.arg_type_name"), _md_string("float2"),
                            _md_string("air.arg_name"), _md_string("uv")});
    auto input = _node({_md_i32(0u), _md_string("air.buffer"),
                        _md_string("air.location_index"), _md_i32(0u), _md_i32(1u),
                        _md_string("air.read"),
                        _md_string("air.address_space"), _md_i32(2u),
                        _md_string("air.arg_type_size"), _md_i32(8u),
                        _md_string("air.arg_type_align_size"), _md_i32(8u),
                        _md_string("air.arg_type_name"), _md_string("float2"),
                        _md_string("air.arg_name"), _md_string("in")});
    auto vertex_id = _node({_md_i32(1u), _md_string("air.vertex_id"),
                            _md_string("air.arg_type_name"), _md_string("uint"),
                            _md_string("air.arg_name"), _md_string("vid")});
    _module.getOrInsertNamedMetadata("air.vertex")
        ->addOperand(_node({llvm::ValueAsMetadata::get(function),
                            _node({position, uv_output}),
                            _node({input, vertex_id})}));
}

void MetalBuiltinLLVMCodegen::_add_swapchain_fragment_metadata(
    llvm::Function *function,
    llvm::GlobalVariable *sampler) noexcept {
    auto output = _node({_md_string("air.render_target"), _md_i32(0u), _md_i32(0u),
                         _md_string("air.arg_type_name"), _md_string("float4")});
    auto position = _node({_md_i32(0u), _md_string("air.position"),
                           _md_string("air.center"), _md_string("air.no_perspective"),
                           _md_string("air.arg_type_name"), _md_string("float4"),
                           _md_string("air.arg_name"), _md_string("p"),
                           _md_string("air.arg_unused")});
    auto uv = _node({_md_i32(1u), _md_string("air.fragment_input"),
                     _md_string("generated(2uvDv2_f)"),
                     _md_string("air.center"), _md_string("air.perspective"),
                     _md_string("air.arg_type_name"), _md_string("float2"),
                     _md_string("air.arg_name"), _md_string("uv")});
    auto texture = _node({_md_i32(2u), _md_string("air.texture"),
                          _md_string("air.location_index"), _md_i32(0u), _md_i32(1u),
                          _md_string("air.sample"),
                          _md_string("air.arg_type_name"),
                          _md_string("texture2d<float, sample>"),
                          _md_string("air.arg_name"), _md_string("image")});
    _module.getOrInsertNamedMetadata("air.fragment")
        ->addOperand(_node({llvm::ValueAsMetadata::get(function), _node({output}),
                            _node({position, uv, texture})}));
    _module.getOrInsertNamedMetadata("air.sampler_states")
        ->addOperand(_node({_md_string("air.sampler_state"),
                            llvm::ValueAsMetadata::get(sampler)}));
}

}// namespace

MetalCodegenLLVMResult luisa_compute_metal_codegen_builtin_llvm(
    MetalBuiltinLLVMProgram program,
    const MetalCodegenLLVMConfig &config) noexcept {
    MetalBuiltinLLVMCodegen codegen{config};
    return codegen.generate(program);
}

}// namespace luisa::compute::metal
