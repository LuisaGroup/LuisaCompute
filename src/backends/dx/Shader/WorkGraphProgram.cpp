#include "WorkGraphProgram.h"
#include "ShaderSerializer.h"

#include "../../common/hlsl/shader_compiler.h"
#include <Resource/GpuAllocator.h>

#include <utility>

namespace lc::dx {

WorkGraphProgram *WorkGraphProgram::CompileWorkGraph(
    Device *device,
    luisa::string_view workGraphName,
    vstd::function<hlsl::CodegenResult()> const& codegen,
    vstd::vector<luisa::compute::Argument> argBindings,
    vstd::vector<SavedArgument> savedArgs,
    uint shaderModel,
    bool enableUnsafeMath,
    bool debug
) {
    vstd::wstring wide_name;
    wide_name.resize(workGraphName.size());
    std::mbstowcs(wide_name.data(), workGraphName.data(), workGraphName.size());

    hlsl::CodegenResult codegen_result = codegen();

    luisa::string dump_name = luisa::format("work_graph_dump_{}.hlsl", workGraphName);
    auto file = fopen(dump_name.c_str(), "w");
    auto code = codegen_result.result.view();
    fwrite(code.data(), 1, code.size(), file);
    fflush(file);
    fclose(file);

    auto compile_result = Device::Compiler()->compile_work_graph(
        code,
        true,
        shaderModel,
        enableUnsafeMath,
        debug
    );

    bool succeeded = compile_result.multi_visit_or(
        false,
        [&](hlsl::ComUniquePtr<IDxcBlob>&) {
            return true;
        },
        [&](vstd::string& failure) {
            LUISA_ERROR("compilation of work graph ({}) failed: {}", workGraphName, failure);
            return false;
        }
    );

    if (!succeeded) { return nullptr; }
    auto buffer = std::move(compile_result.get<0>());

    CD3DX12_STATE_OBJECT_DESC state_object_desc { D3D12_STATE_OBJECT_TYPE_EXECUTABLE };

    ComPtr<ID3DBlob> root_sig_blob = ShaderSerializer::SerializeRootSig(codegen_result.properties, false);
    ComPtr<ID3D12RootSignature> root_sig;

    ThrowIfFailed(device->device->CreateRootSignature(
        0,
        root_sig_blob->GetBufferPointer(),
        root_sig_blob->GetBufferSize(),
        IID_PPV_ARGS(root_sig.GetAddressOf())
    ));

    auto* global_root_sig_desc = state_object_desc.CreateSubobject<CD3DX12_GLOBAL_ROOT_SIGNATURE_SUBOBJECT>();
    global_root_sig_desc->SetRootSignature(root_sig.Get());

    auto* library_desc = state_object_desc.CreateSubobject<CD3DX12_DXIL_LIBRARY_SUBOBJECT>();
    CD3DX12_SHADER_BYTECODE library_code { buffer->GetBufferPointer(), buffer->GetBufferSize() };
    library_desc->SetDXILLibrary(&library_code);

    auto* work_graph_desc = state_object_desc.CreateSubobject<CD3DX12_WORK_GRAPH_SUBOBJECT>();
    work_graph_desc->IncludeAllAvailableNodes();
    work_graph_desc->SetProgramName(wide_name.c_str());

    ComPtr<ID3D12StateObject> state_object;
    ThrowIfFailed(device->device->CreateStateObject(state_object_desc, IID_PPV_ARGS(&state_object)));

    ComPtr<ID3D12StateObjectProperties> state_object_properties;
    ThrowIfFailed(state_object.As(&state_object_properties));

    ComPtr<ID3D12WorkGraphProperties> work_graph_properties;
    state_object->QueryInterface(IID_PPV_ARGS(&work_graph_properties));

    D3D12_WORK_GRAPH_MEMORY_REQUIREMENTS memory_requirements;
    work_graph_properties->GetWorkGraphMemoryRequirements(0, &memory_requirements);

    AllocHandle backing_memory { nullptr };

    // "Maximum size the driver would be able to make use of for backing memory."
    if (memory_requirements.MaxSizeInBytes != 0) {
        ID3D12Heap* heap;
        uint64_t offset;

        luisa::string alloc_name = luisa::format("work graph ({}) backing memory", workGraphName);
        auto alloc = device->defaultAllocator->AllocateBufferHeap(
            device,
            alloc_name,
            memory_requirements.MaxSizeInBytes,
            D3D12_HEAP_TYPE_DEFAULT,
            &heap,
            &offset,
            D3D12_HEAP_FLAG_NONE
        );

        auto buffer_desc = CD3DX12_RESOURCE_DESC::Buffer(
            memory_requirements.MaxSizeInBytes,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
        );

        backing_memory.allocator = device->defaultAllocator.get();
        backing_memory.allocateHandle = alloc;
        ThrowIfFailed(device->device->CreatePlacedResource(
            heap,
            offset,
            &buffer_desc,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&backing_memory.resource)
        ));
    }

    ComPtr<ID3D12StateObjectProperties1> state_object_properties1;
    state_object->QueryInterface(IID_PPV_ARGS(&state_object_properties1));
    auto program_id = state_object_properties1->GetProgramIdentifier(wide_name.c_str());

    uint bindlessBufferCount = 0;
    if (codegen_result.useBufferBindless) bindlessBufferCount++;
    if (codegen_result.useTex2DBindless) bindlessBufferCount++;
    if (codegen_result.useTex3DBindless) bindlessBufferCount++;

    auto work_graph_program = new WorkGraphProgram(
        std::move(codegen_result.properties),
        std::move(savedArgs),
        std::move(root_sig),
        std::move(codegen_result.printers),
        std::move(state_object),
        std::move(backing_memory),
        memory_requirements.MaxSizeInBytes,
        program_id,
        std::move(argBindings)
    );
    work_graph_program->bindlessCount = bindlessBufferCount;
    return work_graph_program;

}

WorkGraphProgram::WorkGraphProgram(
    vstd::vector<hlsl::Property> prop,
    vstd::vector<SavedArgument> savedArgs,
    ComPtr<ID3D12RootSignature> rootSignature,
    vstd::vector<std::pair<vstd::string, Type const *>> printers,
    ComPtr<ID3D12StateObject> stateObject,
    AllocHandle backingMemory,
    size_t backingMemorySize,
    D3D12_PROGRAM_IDENTIFIER programId,
    vstd::vector<luisa::compute::Argument> argBindings
) : Shader { std::move(prop), std::move(savedArgs), std::move(rootSignature), std::move(printers) },
    stateObject(std::move(stateObject)),
    backingMemory(std::move(backingMemory)),
    backingMemorySize(backingMemorySize),
    programId(programId),
    argBindings(std::move(argBindings)) {}


WorkGraphProgram::~WorkGraphProgram() = default;

} // namespace lc::dx