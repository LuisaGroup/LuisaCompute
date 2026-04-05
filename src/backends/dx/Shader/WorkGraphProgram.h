#pragma once
#include <Resource/AllocHandle.h>
#include <Shader/Shader.h>

// ??
#include "../../common/hlsl/hlsl_codegen.h"

namespace lc::dx {
class WorkGraphProgram final : public Shader {
public:
    Tag GetTag() const noexcept override { return Tag::WorkGraphProgram; }

    static WorkGraphProgram *CompileWorkGraph(
        Device *device,
        luisa::string_view workGraphName,
        vstd::function<hlsl::CodegenResult()> const &codegen,
        // vstd::vector<luisa::compute::Argument> bindings,
        uint shaderModel,
        bool enableUnsafeMath,
        bool debug
    );

    ~WorkGraphProgram() override;

    WorkGraphProgram(
        Device* device,
        vstd::vector<hlsl::Property> prop,
        // vstd::vector<SavedArgument> args,
        // vstd::vector<luisa::compute::Argument> bindings,
        vstd::vector<std::pair<vstd::string, Type const *>> printers,
        ComPtr<ID3D12StateObject> stateObject,
        AllocHandle backingMemory,
        size_t backingMemorySize,
        D3D12_PROGRAM_IDENTIFIER programId
    );

    [[nodiscard]] void* native_handle() const noexcept { return stateObject.Get(); }

private:
    ComPtr<ID3D12StateObject> stateObject;
    AllocHandle backingMemory;             // GPU scratch buffer
    size_t backingMemorySize;
    D3D12_PROGRAM_IDENTIFIER programId;    // identifies the program for dispatch

};


} // namespace lc::dx