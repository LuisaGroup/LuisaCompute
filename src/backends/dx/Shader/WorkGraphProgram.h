#pragma once
#include <Resource/AllocHandle.h>
#include <Shader/Shader.h>

// ??
#include "../../common/hlsl/hlsl_codegen.h"

namespace lc::dx {
class WorkGraphProgram final : public Shader {
public:
    Tag get_tag() const noexcept override { return Tag::WorkGraphProgram; }

    static WorkGraphProgram *compile_work_graph(
        Device *device,
        luisa::string_view work_graph_name,
        vstd::function<hlsl::CodegenResult()> const &codegen,
        vstd::vector<luisa::compute::Argument> arg_bindings,
        vstd::vector<SavedArgument> saved_args,
        uint shader_model,
        bool enable_unsafe_math,
        bool debug
    );

    ~WorkGraphProgram() override;

    WorkGraphProgram(
        vstd::vector<hlsl::Property> prop,
        vstd::vector<SavedArgument> saved_args,
        ComPtr<ID3D12RootSignature> root_signature,
        vstd::vector<std::pair<vstd::string, Type const *>> printers,
        ComPtr<ID3D12StateObject> state_object,
        AllocHandle backing_memory,
        size_t backing_memory_size,
        D3D12_PROGRAM_IDENTIFIER program_id,
        vstd::vector<luisa::compute::Argument> arg_bindings
    );

    [[nodiscard]] void* native_handle() const noexcept { return _state_object.Get(); }
    [[nodiscard]] D3D12_PROGRAM_IDENTIFIER const &program_id() const noexcept { return _program_id; }
    [[nodiscard]] ID3D12Resource *backing_memory() const noexcept { return _backing_memory.resource.Get(); }
    [[nodiscard]] size_t backing_memory_size() const noexcept { return _backing_memory_size; }
    [[nodiscard]] vstd::span<luisa::compute::Argument const> arg_bindings() const noexcept { return _arg_bindings; }

private:
    ComPtr<ID3D12StateObject> _state_object;
    AllocHandle _backing_memory;             // GPU scratch buffer
    size_t _backing_memory_size;
    D3D12_PROGRAM_IDENTIFIER _program_id;    // identifies the program for dispatch
    vstd::vector<luisa::compute::Argument> _arg_bindings; // merged bound resources across all nodes
};


} // namespace lc::dx
