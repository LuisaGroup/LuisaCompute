#include "metal_codegen_llvm_impl.h"

namespace luisa::compute::metal::detail {

void MetalCodegenLLVMImpl::_link_native_include() noexcept {
    luisa::vector<luisa::string> external_symbols;
    for (auto &&[xir_function, llvm_function] : _functions) {
        if (xir_function->isa<xir::ExternalFunction>()) {
            LUISA_ASSERT(!llvm_function->use_empty(),
                         "Metal AIR emitted an unused external declaration '{}'.",
                         llvm_function->getName().str());
            external_symbols.emplace_back(llvm_function->getName());
        }
    }
    if (external_symbols.empty()) { return; }
    LUISA_ASSERT(!_config.native_include.empty(),
                 "Metal AIR shader uses ExternalCallable but ShaderOption::native_include is empty. "
                 "Metal4 native includes must contain LLVM IR or bitcode definitions.");

    auto source_name = _config.source_file.empty() ?
                           llvm::StringRef{"metal4_native_include.ll"} :
                           llvm::StringRef{_config.source_file};
    auto buffer = llvm::MemoryBuffer::getMemBuffer(
        llvm::StringRef{_config.native_include.data(),
                        _config.native_include.size()},
        source_name, false);
    llvm::SMDiagnostic error;
    auto native_module = llvm::parseIR(
        buffer->getMemBufferRef(), error, _context);
    if (native_module == nullptr) {
        std::string diagnostic;
        llvm::raw_string_ostream stream{diagnostic};
        error.print("LuisaCompute Metal4 native include", stream);
        stream.flush();
        LUISA_ERROR_WITH_LOCATION(
            "Failed to parse Metal4 native include as LLVM IR/bitcode:\n{}",
            diagnostic);
    }

    auto expected_triple = _module.getTargetTriple();
    auto native_triple = native_module->getTargetTriple();
    LUISA_ASSERT(native_triple.str().empty() ||
                     native_triple == expected_triple,
                 "Metal4 native include targets '{}', expected '{}'.",
                 native_triple.str(), expected_triple.str());
    auto native_layout = native_module->getDataLayoutStr();
    auto expected_layout = _module.getDataLayoutStr();
    LUISA_ASSERT(native_layout.empty() || native_layout == expected_layout,
                 "Metal4 native include has an incompatible LLVM data layout.");
    native_module->setTargetTriple(expected_triple);
    native_module->setDataLayout(_data_layout);

    static constexpr llvm::Attribute::AttrKind abi_attributes[]{
        llvm::Attribute::InReg,
        llvm::Attribute::SExt,
        llvm::Attribute::ZExt,
#if LLVM_VERSION_MAJOR >= 22
        llvm::Attribute::NoExt,
#endif
        llvm::Attribute::ByRef,
        llvm::Attribute::ByVal,
        llvm::Attribute::ElementType,
        llvm::Attribute::InAlloca,
        llvm::Attribute::Preallocated,
        llvm::Attribute::StructRet,
        llvm::Attribute::Nest,
        llvm::Attribute::Returned,
        llvm::Attribute::SwiftAsync,
        llvm::Attribute::SwiftError,
        llvm::Attribute::SwiftSelf,
    };
    auto check_abi_attributes = [](
                                    llvm::StringRef name,
                                    llvm::AttributeSet native_attributes,
                                    llvm::AttributeSet expected_attributes,
                                    llvm::StringRef position) noexcept {
        for (auto kind : abi_attributes) {
            LUISA_ASSERT(native_attributes.getAttribute(kind) ==
                             expected_attributes.getAttribute(kind),
                         "Metal4 native function '{}' has incompatible '{}' ABI "
                         "attribute on {}.",
                         name.str(),
                         llvm::Attribute::getNameFromAttrKind(kind).str(),
                         position.str());
        }
    };

    for (auto &&[xir_function, llvm_value] : _functions) {
        if (!xir_function->isa<xir::ExternalFunction>()) { continue; }
        auto external = llvm_value;
        auto definition = native_module->getFunction(external->getName());
        LUISA_ASSERT(definition != nullptr && !definition->isDeclaration(),
                     "Metal4 native include does not define external function '{}'.",
                     external->getName().str());
        LUISA_ASSERT(!definition->hasLocalLinkage() &&
                         !definition->hasAvailableExternallyLinkage(),
                     "Metal4 native function '{}' must provide an externally linkable definition.",
                     external->getName().str());
        LUISA_ASSERT(definition->getFunctionType() ==
                         external->getFunctionType(),
                     "Metal4 native function '{}' does not match its ExternalCallable ABI.",
                     external->getName().str());
        LUISA_ASSERT(definition->getAddressSpace() == external->getAddressSpace(),
                     "Metal4 native function '{}' has an incompatible function address space.",
                     external->getName().str());
        LUISA_ASSERT(definition->getCallingConv() == external->getCallingConv(),
                     "Metal4 native function '{}' has an incompatible LLVM calling convention.",
                     external->getName().str());
        auto definition_attributes = definition->getAttributes();
        auto external_attributes = external->getAttributes();
        check_abi_attributes(
            external->getName(), definition_attributes.getFnAttrs(),
            external_attributes.getFnAttrs(), "the function");
        check_abi_attributes(
            external->getName(), definition_attributes.getRetAttrs(),
            external_attributes.getRetAttrs(), "the return value");
        auto xir_external = static_cast<const xir::ExternalFunction *>(xir_function);
        auto xir_argument = xir_external->arguments().begin();
        for (auto i = 0u; i < definition->arg_size(); ++i, ++xir_argument) {
            LUISA_ASSERT(xir_argument != xir_external->arguments().end(),
                         "Metal4 ExternalCallable argument count changed during ABI validation.");
            auto native_attributes = definition_attributes.getParamAttrs(i);
            if (auto alignment_attribute =
                    native_attributes.getAttribute(llvm::Attribute::Alignment);
                alignment_attribute.isValid()) {
                auto native_alignment = alignment_attribute.getAlignment();
                auto available_alignment = _type_alignment(xir_argument->type());
                LUISA_ASSERT(
                    xir_argument->is_reference() && native_alignment.has_value() &&
                        native_alignment->value() <= available_alignment,
                    "Metal4 native function '{}' requires alignment {} on parameter {}, "
                    "but its ExternalCallable argument only guarantees alignment {}.",
                    external->getName().str(),
                    native_alignment ? native_alignment->value() : 0u,
                    i, available_alignment);
            }
            auto position = luisa::format("parameter {}", i);
            check_abi_attributes(
                external->getName(), native_attributes,
                external_attributes.getParamAttrs(i),
                llvm::StringRef{position.data(), position.size()});
        }
        LUISA_ASSERT(xir_argument == xir_external->arguments().end(),
                     "Metal4 ExternalCallable argument count changed during ABI validation.");
    }

    if (llvm::Linker::linkModules(
            _module, std::move(native_module),
            llvm::Linker::Flags::LinkOnlyNeeded)) {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to link Metal4 native include into the generated LLVM module.");
    }
    for (auto &&symbol : external_symbols) {
        auto linked = _module.getFunction(
            llvm::StringRef{symbol.data(), symbol.size()});
        LUISA_ASSERT(linked != nullptr && !linked->isDeclaration(),
                     "Metal4 native external function '{}' remained unresolved after linking.",
                     symbol);
    }
}

MetalCodegenLLVMResult MetalCodegenLLVMImpl::generate(const xir::Module &xir_module) noexcept {
    _collect_ray_query_pipelines(xir_module);
    luisa::string reason;
    LUISA_ASSERT(luisa_compute_metal_codegen_llvm_supported(xir_module, _config, &reason),
                 "XIR module is unsupported by Metal AIR LLVM codegen: {}", reason);
    _module.setModuleIdentifier(xir_module.name().value_or("luisa.metal.air"));
    _collect_print_formats(xir_module);
    for (auto function : xir_module.function_list()) {
        if (_config.program == MetalAIRProgram::COMPUTE &&
            function->derived_function_tag() == xir::DerivedFunctionTag::KERNEL) {
            _kernel = static_cast<const xir::KernelFunction *>(function);
            for (auto argument : _kernel->arguments()) {
                _root_arguments.emplace_back(argument);
            }
        } else if (_config.program != MetalAIRProgram::COMPUTE &&
                   function->derived_function_tag() == xir::DerivedFunctionTag::RASTER_STAGE) {
            _raster_stage = static_cast<const xir::RasterStageFunction *>(function);
        }
    }
    if (_config.program == MetalAIRProgram::COMPUTE) {
        LUISA_ASSERT(_kernel != nullptr, "Metal AIR LLVM codegen requires one kernel.");
    } else {
        LUISA_ASSERT(_raster_stage != nullptr,
                     "Metal AIR LLVM codegen requires one raster stage.");
        _root_arguments = _config.raster.root_arguments;
        if (_raster_stage->stage() == xir::RasterStage::FRAGMENT) {
            _raster_stage->traverse_instructions(
                [this](const xir::Instruction *instruction) noexcept {
                    if (!instruction->isa<xir::ThreadGroupInst>()) { return; }
                    auto group = static_cast<const xir::ThreadGroupInst *>(
                        instruction);
                    auto mode = air_raster_depth_mode(group->op());
                    if (mode == AIRRasterDepthMode::NONE) { return; }
                    LUISA_ASSERT(
                        _raster_depth_mode == AIRRasterDepthMode::NONE ||
                            _raster_depth_mode == mode,
                        "Metal AIR fragment stage mixes shader-depth qualifiers.");
                    _raster_depth_mode = mode;
                });
        }
    }
    for (auto function : xir_module.function_list()) {
        if (function->derived_function_tag() == xir::DerivedFunctionTag::CALLABLE) {
            static_cast<void>(_translate_callable(static_cast<const xir::CallableFunction *>(function)));
        }
    }
    _emit_ray_query_intersection_functions();
    if (_config.program == MetalAIRProgram::COMPUTE) {
        static_cast<void>(_translate_kernel(_kernel));
    } else {
        static_cast<void>(_translate_raster_stage(_raster_stage));
    }
    _link_native_include();
    _add_module_metadata();
    if (llvm::verifyModule(_module, &llvm::errs())) {
        _module.print(llvm::errs(), nullptr, false, true);
        LUISA_ERROR_WITH_LOCATION("Metal AIR LLVM module verification failed.");
    }
    return std::move(_result);
}

}// namespace luisa::compute::metal::detail
