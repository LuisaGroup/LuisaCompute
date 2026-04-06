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
        vstd::vector<luisa::compute::Argument> argBindings,
        vstd::vector<SavedArgument> savedArgs,
        uint shaderModel,
        bool enableUnsafeMath,
        bool debug
    );

    ~WorkGraphProgram() override;

    WorkGraphProgram(
        vstd::vector<hlsl::Property> prop,
        vstd::vector<SavedArgument> savedArgs,
        ComPtr<ID3D12RootSignature> rootSignature,
        vstd::vector<std::pair<vstd::string, Type const *>> printers,
        ComPtr<ID3D12StateObject> stateObject,
        AllocHandle backingMemory,
        size_t backingMemorySize,
        D3D12_PROGRAM_IDENTIFIER programId,
        vstd::vector<luisa::compute::Argument> argBindings
    );

    [[nodiscard]] void* native_handle() const noexcept { return stateObject.Get(); }
    [[nodiscard]] D3D12_PROGRAM_IDENTIFIER const &ProgramId() const noexcept { return programId; }
    [[nodiscard]] ID3D12Resource *BackingMemory() const noexcept { return backingMemory.resource.Get(); }
    [[nodiscard]] size_t BackingMemorySize() const noexcept { return backingMemorySize; }
    [[nodiscard]] vstd::span<luisa::compute::Argument const> ArgBindings() const noexcept { return argBindings; }

private:
    ComPtr<ID3D12StateObject> stateObject;
    AllocHandle backingMemory;             // GPU scratch buffer
    size_t backingMemorySize;
    D3D12_PROGRAM_IDENTIFIER programId;    // identifies the program for dispatch
    vstd::vector<luisa::compute::Argument> argBindings; // merged bound resources across all nodes

};


} // namespace lc::dx