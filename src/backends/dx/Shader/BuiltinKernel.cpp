
#include <Shader/BuiltinKernel.h>
#include <Codegen/DxCodegen.h>
#include <Codegen/ShaderHeader.h>
namespace toolhub::directx {
ComputeShader const *BuiltinKernel::LoadAccelSetKernel(Device *device) {
    CodegenResult code;
    code.bdlsBufferCount = 0;
    code.result = vstd::string(GetAccelProcessCompute());
    code.properties.resize(3);
    code.md5 = vstd::MD5(code.result);
    auto &InstBuffer = code.properties[0];
    InstBuffer.first = "_InstBuffer"sv;
    InstBuffer.second.arrSize = 0;
    InstBuffer.second.registerIndex = 0;
    InstBuffer.second.spaceIndex = 0;
    InstBuffer.second.type = ShaderVariableType::StructuredBuffer;
    auto &Global = code.properties[1];
    Global.first = "_Global"sv;
    Global.second.arrSize = 0;
    Global.second.registerIndex = 0;
    Global.second.spaceIndex = 0;
    Global.second.type = ShaderVariableType::ConstantBuffer;
    auto &SetBuffer = code.properties[2];
    SetBuffer.first = "_SetBuffer"sv;
    SetBuffer.second.arrSize = 0;
    SetBuffer.second.registerIndex = 0;
    SetBuffer.second.spaceIndex = 0;
    SetBuffer.second.type = ShaderVariableType::RWStructuredBuffer;
    return ComputeShader::CompileCompute(
        device,
        code,
        uint3(64, 1, 1),
        60,
        {
            "set_accel_kernel"sv,
        });
}

}// namespace toolhub::directx