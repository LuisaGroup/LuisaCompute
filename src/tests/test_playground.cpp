//
// Created by Mike Smith on 2021/4/6.
//

#include <array>

#include <spirv-tools/optimizer.hpp>
#include <spirv-tools/libspirv.hpp>
#include <SPIRV/GLSL.std.450.h>
#include <SPIRV/SpvBuilder.h>

#include <core/clock.h>

int main() {

    spv::SpvBuildLogger logger;
    spv::Builder builder{spv::Spv_1_5, 0u, &logger};
    builder.addCapability(spv::CapabilityShader);
    auto ext = builder.import("GLSL.std.450");
    builder.setSource(spv::SourceLanguageGLSL, 450);
    auto f = builder.makeEntryPoint("main");
    auto entry = builder.addEntryPoint(spv::ExecutionModelGLCompute, f, "main");
    builder.addExecutionMode(f, spv::ExecutionModeLocalSize, 8, 8, 1);

    auto arg_type = builder.makeStructType({builder.makeIntType(32)}, "Argument");
    builder.addMemberName(arg_type, 0, "a");
    builder.addDecoration(arg_type, spv::DecorationBlock);
    builder.addMemberDecoration(arg_type, 0, spv::DecorationOffset, 0);
    auto pa = builder.createVariable(spv::NoPrecision, spv::StorageClassStorageBuffer, arg_type, "pa");
    builder.addDecoration(pa, spv::DecorationBinding, 0);
    builder.addDecoration(pa, spv::DecorationDescriptorSet, 0);
    entry->addIdOperand(pa);

    auto gid = builder.createVariable(spv::NoPrecision, spv::StorageClassInput,
                                      builder.makeVectorType(builder.makeUintType(32), 3),
                                      "GlobalInvocationId");
    builder.addDecoration(gid, spv::DecorationBuiltIn, spv::BuiltInGlobalInvocationId);

    builder.setAccessChainLValue(pa);
    builder.accessChainPush(builder.makeIntConstant(0), {}, 4u);
    auto a = builder.accessChainLoad(spv::NoPrecision, {}, {}, builder.makeIntType(32));
    builder.clearAccessChain();
    auto fa = builder.createUnaryOp(spv::OpConvertSToF, builder.makeFloatType(32), a);
    auto mat2_type = builder.makeMatrixType(builder.makeFloatType(32), 2, 2);
    auto m = builder.createMatrixConstructor(spv::NoPrecision, {fa}, mat2_type);
    builder.createOp(spv::OpExtInst, mat2_type, {spv::IdImmediate{true, ext}, spv::IdImmediate{false, GLSLstd450MatrixInverse}, spv::IdImmediate{true, m}});
    builder.leaveFunction();
    builder.postProcess();

    std::vector<uint32_t> binary;
    builder.dump(binary);

    auto spv_logger = [](spv_message_level_t level, const char *source,
                     spv_position_t position, const char *message) noexcept {
        std::array levels{"FATAL", "INTERNAL_ERROR", "ERROR", "WARNING", "INFO", "DEBUG"};
        std::cerr << "[" << levels[level] << "] " << message << " [" << source << ":"
                  << position.line << ":" << position.column << ":" << position.index << "]"
                  << std::endl;
    };

    spvtools::SpirvTools tools{SPV_ENV_VULKAN_1_3};
    tools.SetMessageConsumer(spv_logger);
    tools.Validate(binary);

    auto dump = [&tools](auto &&binary) {
        std::string disassembly;
        if (tools.Disassemble(binary, &disassembly,
                              SPV_BINARY_TO_TEXT_OPTION_COLOR |
                                  SPV_BINARY_TO_TEXT_OPTION_INDENT |
                                  SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES |
                                  SPV_BINARY_TO_TEXT_OPTION_COMMENT)) {
            std::cout << disassembly << std::endl;
        }
    };

    std::cout << "Before optimization:" << std::endl;
    dump(binary);

    spvtools::Optimizer optimizer{SPV_ENV_VULKAN_1_3};
    optimizer.SetMessageConsumer(spv_logger);
    optimizer.RegisterPass(spvtools::CreateStripDebugInfoPass());
    optimizer.RegisterPass(spvtools::CreateStripNonSemanticInfoPass());
    optimizer.RegisterLegalizationPasses();
    optimizer.RegisterPerformancePasses();

    std::vector<uint32_t> optimized_binary;
    spvtools::OptimizerOptions options;
    options.set_run_validator(false);
    options.set_preserve_bindings(true);
    luisa::Clock clock;
    optimizer.Run(binary.data(), binary.size(), &optimized_binary, options);

    std::cout << "After optimization (" << clock.toc() << " ms):" << std::endl;
    dump(optimized_binary);
}
