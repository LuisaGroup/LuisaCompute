#include <cstdio>
#include <vector>

#include <glslang/Public/ShaderLang.h>
#include <glslang/Public/ResourceLimits.h>
#include <SPIRV/GlslangToSpv.h>

int main() {
    if (!glslang::InitializeProcess()) {
        fprintf(stderr, "Failed to initialize glslang\n");
        return 1;
    }

    const char* shaderSource = R"(
        #version 450
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
        }
    )";

    glslang::TShader shader(EShLangCompute);
    shader.setStrings(&shaderSource, 1);

    TBuiltInResource resources = *GetDefaultResources();
    EShMessages messages = static_cast<EShMessages>(EShMsgSpvRules | EShMsgVulkanRules);

    if (!shader.parse(&resources, 450, false, messages)) {
        fprintf(stderr, "Parse failed: %s\n", shader.getInfoLog());
        glslang::FinalizeProcess();
        return 1;
    }

    glslang::TProgram program;
    program.addShader(&shader);

    if (!program.link(messages)) {
        fprintf(stderr, "Link failed: %s\n", program.getInfoLog());
        glslang::FinalizeProcess();
        return 1;
    }

    std::vector<unsigned int> spirv;
    glslang::SpvOptions spvOptions;
    spvOptions.generateDebugInfo = false;
    spvOptions.stripDebugInfo = true;
    spvOptions.disableOptimizer = true;

    glslang::GlslangToSpv(*program.getIntermediate(EShLangCompute), spirv, &spvOptions);

    if (spirv.empty()) {
        fprintf(stderr, "SPIRV generation failed\n");
        glslang::FinalizeProcess();
        return 1;
    }

    printf("Generated SPIRV with %zu words\n", spirv.size());
    glslang::FinalizeProcess();
    return 0;
}
