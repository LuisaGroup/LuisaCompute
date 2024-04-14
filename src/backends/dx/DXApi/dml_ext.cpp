#include "dml_ext.h"
#include "LCCmdBuffer.h"
#include <luisa/runtime/stream.h>
#define _D3D12MA_IUNKNOWN_IMPL_FUNCTIONS
#include "DirectMLX.h"
#include <luisa/backends/ext/dx_custom_cmd.h>
#include <wrl/client.h>
//#include <wil/result_macros.h>
using namespace luisa;
using namespace luisa::compute;
class DMLModule {
    DynamicModule module;
    std::mutex mtx;

public:
    DynamicModule &get() {
        std::lock_guard lck{mtx};
        if (module) return module;
        module = DynamicModule::load("DirectML");
        return module;
    }
};
static DMLModule _dml_module;
class DxDMLGraph : public DMLGraph {
public:
    DeviceInterface *device_interface;
    ComPtr<IDMLDevice> dmlDevice;
    ComPtr<IDMLCompiledOperator> dmlCompiledOperator;

    ComPtr<IDMLBindingTable> dmlBindingTable;
    ComPtr<IDMLCommandRecorder> dmlCommandRecorder;
    ComPtr<ID3D12DescriptorHeap> descriptorHeap;
    const size_t weight_size;
    const size_t output_size;
    const size_t input_size;
    size_t desc_count;
    size_t temp_res_count;
    size_t persist_resource_size;

    uint batch_size;
    uint input;
    uint output;
    luisa::vector<uint> hiddens;
    luisa::vector<FusedActivation> activations;

    bool bind = false;
    bool half;
    BufferCreationInfo temporaryBuffer{BufferCreationInfo::make_invalid()};
    BufferCreationInfo persistentBuffer{BufferCreationInfo::make_invalid()};
    // ComPtr<ID3D12Resource> temporaryBuffer;
    // ComPtr<ID3D12Resource> persistentBuffer;
    unique_ptr<Command> build() noexcept override;
    unique_ptr<Command> forward(Argument::Buffer input_buffer, Argument::Buffer output_buffer, Argument::Buffer weights_buffer) noexcept override;
    DxDMLGraph(
        DeviceInterface *device_interface,
        size_t weight,
        size_t output,
        size_t input,
        size_t batch_size,
        size_t data_size,
        luisa::span<uint const> hiddens,
        luisa::span<const FusedActivation> activations,
        bool half)
        : device_interface(device_interface),
          weight_size(weight * data_size),
          output_size(output * data_size * batch_size),
          input_size(input * data_size * batch_size),
          batch_size(batch_size),
          input(input),
          output(output),
          half(half) {
        if (!hiddens.empty()) {
            vstd::push_back_all(this->hiddens, hiddens);
        }
        vstd::push_back_all(this->activations, activations);
    }
    ~DxDMLGraph() {
        if (temporaryBuffer.valid()) {
            device_interface->destroy_buffer(temporaryBuffer.handle);
        }
        if (persistentBuffer.valid()) {
            device_interface->destroy_buffer(persistentBuffer.handle);
        }
    }
    size_t input_buffer_size_bytes() const noexcept override {
        return input_size;
    }
    size_t output_buffer_size_bytes() const noexcept override {
        return output_size;
    }
    size_t weight_buffer_size_bytes() const noexcept override {
        return weight_size;
    }
};
class DxGraphBuildCommand final : public DXCustomCmd {
public:
    DxGraphBuildCommand(DxDMLGraph *graph) : dmlGraph(graph) {}
    LUISA_MAKE_COMMAND_COMMON(StreamTag::COMPUTE)
    [[nodiscard]] luisa::span<const ResourceUsage> get_resource_usages() const noexcept override {
        return {};
    }

private:
    DxDMLGraph *dmlGraph;
    void execute(
        IDXGIAdapter1 *adapter,
        IDXGIFactory2 *dxgi_factory,
        ID3D12Device *device,
        ID3D12GraphicsCommandList4 *command_list) const noexcept override;
};

static dml::FusedActivation ToDMLActivation(FusedActivation a) {
    dml::FusedActivation r;
    r.param1 = a.param1;
    r.param2 = a.param2;
    r.activation = [&]() {
        switch (a.type) {
            case FusedActivation::Type::ELU: return DML_OPERATOR_ACTIVATION_ELU;
            case FusedActivation::Type::HARD_SIGMOID: return DML_OPERATOR_ACTIVATION_HARD_SIGMOID;
            case FusedActivation::Type::IDENTITY: return DML_OPERATOR_ACTIVATION_IDENTITY;
            case FusedActivation::Type::LEAKY_RELU: return DML_OPERATOR_ACTIVATION_LEAKY_RELU;
            case FusedActivation::Type::LINEAR: return DML_OPERATOR_ACTIVATION_LINEAR;
            case FusedActivation::Type::PARAMETRIC_SOFTPLUS: return DML_OPERATOR_ACTIVATION_PARAMETRIC_SOFTPLUS;
            case FusedActivation::Type::RELU: return DML_OPERATOR_ACTIVATION_RELU;
            case FusedActivation::Type::SCALED_ELU: return DML_OPERATOR_ACTIVATION_SCALED_ELU;
            case FusedActivation::Type::SCALED_TANH: return DML_OPERATOR_ACTIVATION_SCALED_TANH;
            case FusedActivation::Type::SIGMOID: return DML_OPERATOR_ACTIVATION_SIGMOID;
            case FusedActivation::Type::SOFTPLUS: return DML_OPERATOR_ACTIVATION_SOFTPLUS;
            case FusedActivation::Type::SOFTSIGN: return DML_OPERATOR_ACTIVATION_SOFTSIGN;
            case FusedActivation::Type::TANH: return DML_OPERATOR_ACTIVATION_TANH;
            case FusedActivation::Type::THRESHOLDED_RELU: return DML_OPERATOR_ACTIVATION_THRESHOLDED_RELU;
            case FusedActivation::Type::SHRINK: return DML_OPERATOR_ACTIVATION_SHRINK;
            case FusedActivation::Type::CELU: return DML_OPERATOR_ACTIVATION_CELU;
            default: return DML_OPERATOR_INVALID;
        }
    }();
    return r;
}

void DxGraphBuildCommand::execute(IDXGIAdapter1 *adapter, IDXGIFactory2 *dxgi_factory, ID3D12Device *device, ID3D12GraphicsCommandList4 *command_list) const noexcept {
    if (dmlGraph->dmlDevice) [[unlikely]] {
        LUISA_ERROR("DML Graph already been built.");
    }
    const uint input = dmlGraph->input;
    const uint output = dmlGraph->output;
    const uint batch_size = dmlGraph->batch_size;
    DML_TENSOR_DATA_TYPE dataType = dmlGraph->half ? DML_TENSOR_DATA_TYPE_FLOAT16 : DML_TENSOR_DATA_TYPE_FLOAT32;
    DML_CREATE_DEVICE_FLAGS dmlCreateDeviceFlags = DML_CREATE_DEVICE_FLAG_NONE;
    auto &md = _dml_module.get();
    HRESULT(WINAPI * DMLCreateDevice)
    (
        ID3D12Device * d3d12Device,
        DML_CREATE_DEVICE_FLAGS flags,
        REFIID riid,// Expected: IDMLDevice
        _COM_Outptr_opt_ void **ppv);
    DMLCreateDevice = md.function<std::remove_pointer_t<decltype(DMLCreateDevice)>>("DMLCreateDevice");

    ThrowIfFailed(DMLCreateDevice(
        device,
        dmlCreateDeviceFlags,
        IID_PPV_ARGS(dmlGraph->dmlDevice.GetAddressOf())));

    dml::Graph graph(dmlGraph->dmlDevice.Get());
    UINT tensorSizes[4] = {1, 1, UINT(batch_size), UINT(input)};
    dml::TensorDesc::Dimensions inputDimensions(std::begin(tensorSizes), std::end(tensorSizes));
    dml::TensorDesc desc = {dataType, inputDimensions};
    dml::Expression inputLayer = dml::InputTensor(graph, 0, desc);

    vstd::vector<dml::Expression> expressions{};
    expressions.reserve((dmlGraph->hiddens.size() + 1) * 2);
    uint lastDim = input;
    auto &lastOutput = inputLayer;
    for (uint i = 0; i < dmlGraph->hiddens.size(); i++) {
        auto hidden_dim = dmlGraph->hiddens[i];
        UINT matrixSizes[4] = {1, 1, hidden_dim, UINT(lastDim)};
        dml::TensorDesc::Dimensions matrixDimensions = dml::TensorDesc::Dimensions(std::begin(matrixSizes), std::end(matrixSizes));
        auto mdesc = dml::TensorDesc{dataType, matrixDimensions};
        dml::Expression &weights = expressions.emplace_back(dml::InputTensor(graph, i + 1, mdesc));
        dml::Expression &fc = expressions.emplace_back(
            dml::Gemm(lastOutput, weights,
                      dml::NullOpt, DML_MATRIX_TRANSFORM_NONE, DML_MATRIX_TRANSFORM_TRANSPOSE, 1.f, 1.f, ToDMLActivation(dmlGraph->activations[i])));
        lastDim = hidden_dim;
        lastOutput = fc;
    }
    {
        UINT matrixSizes[4] = {1, 1, UINT(output), UINT(lastDim)};
        dml::TensorDesc::Dimensions matrixDimensions = dml::TensorDesc::Dimensions(std::begin(matrixSizes), std::end(matrixSizes));
        auto mdesc = dml::TensorDesc{dataType, matrixDimensions};
        dml::Expression &weights = expressions.emplace_back(dml::InputTensor(graph, dmlGraph->hiddens.size() + 1, mdesc));
        dml::Expression &fc = expressions.emplace_back(dml::Gemm(lastOutput, weights, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE, DML_MATRIX_TRANSFORM_TRANSPOSE, 1.f, 1.f, ToDMLActivation(dmlGraph->activations.back())));
        lastOutput = fc;
    }

    DML_EXECUTION_FLAGS executionFlags = DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION;
    dmlGraph->dmlCompiledOperator.Attach(graph.Compile(executionFlags, {lastOutput}).Detach());

    ComPtr<IDMLOperatorInitializer> dmlOperatorInitializer;
    IDMLCompiledOperator *dmlCompiledOperators[] = {dmlGraph->dmlCompiledOperator.Get()};
    ThrowIfFailed(dmlGraph->dmlDevice->CreateOperatorInitializer(
        vstd::array_count(dmlCompiledOperators),
        dmlCompiledOperators,
        IID_PPV_ARGS(dmlOperatorInitializer.GetAddressOf())));

    // Query the operator for the required size (in descriptors) of its binding table.
    // You need to initialize an operator exactly once before it can be executed, and
    // the two stages require different numbers of descriptors for binding. For simplicity,
    // we create a single descriptor heap that's large enough to satisfy them both.
    DML_BINDING_PROPERTIES initializeBindingProperties = dmlOperatorInitializer->GetBindingProperties();
    DML_BINDING_PROPERTIES executeBindingProperties = dmlGraph->dmlCompiledOperator->GetBindingProperties();
    dmlGraph->desc_count = std::max(
        initializeBindingProperties.RequiredDescriptorCount,
        executeBindingProperties.RequiredDescriptorCount);

    // Create descriptor heaps.

    D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc{};
    descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    descriptorHeapDesc.NumDescriptors = dmlGraph->desc_count;
    descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ThrowIfFailed(device->CreateDescriptorHeap(
        &descriptorHeapDesc,
        IID_PPV_ARGS(dmlGraph->descriptorHeap.GetAddressOf())));

    // Set the descriptor heap(s).
    ID3D12DescriptorHeap *d3D12DescriptorHeaps[] = {dmlGraph->descriptorHeap.Get()};
    command_list->SetDescriptorHeaps(vstd::array_count(d3D12DescriptorHeaps), d3D12DescriptorHeaps);

    // Create a binding table over the descriptor heap we just created.
    DML_BINDING_TABLE_DESC dmlBindingTableDesc{};
    dmlBindingTableDesc.Dispatchable = dmlOperatorInitializer.Get();
    dmlBindingTableDesc.CPUDescriptorHandle = dmlGraph->descriptorHeap->GetCPUDescriptorHandleForHeapStart();
    dmlBindingTableDesc.GPUDescriptorHandle = dmlGraph->descriptorHeap->GetGPUDescriptorHandleForHeapStart();
    dmlBindingTableDesc.SizeInDescriptors = dmlGraph->desc_count;

    ComPtr<IDMLBindingTable> initBindingTable;
    ThrowIfFailed(dmlGraph->dmlDevice->CreateBindingTable(
        &dmlBindingTableDesc,
        IID_PPV_ARGS(initBindingTable.GetAddressOf())));

    // Create the temporary and persistent resources that are necessary for executing an operator.

    // The temporary resource is scratch memory (used internally by DirectML), whose contents you don't need to define.
    // The persistent resource is long-lived, and you need to initialize it using the IDMLOperatorInitializer.

    dmlGraph->temp_res_count = std::max(
        initializeBindingProperties.TemporaryResourceSize,
        executeBindingProperties.TemporaryResourceSize);
    dmlGraph->persist_resource_size = executeBindingProperties.PersistentResourceSize;

    // Bind and initialize the operator on the GPU.

    if (dmlGraph->temp_res_count != 0) {
        dmlGraph->temporaryBuffer = dmlGraph->device_interface->create_buffer(
            Type::of<void>() /*nullptr*/,
            dmlGraph->temp_res_count,
            nullptr);
        if (initializeBindingProperties.TemporaryResourceSize != 0) {
            DML_BUFFER_BINDING bufferBinding{
                reinterpret_cast<ID3D12Resource *>(dmlGraph->temporaryBuffer.native_handle),
                0, dmlGraph->temp_res_count};
            DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
            initBindingTable->BindTemporaryResource(&bindingDesc);
        }
    }

    if (dmlGraph->persist_resource_size != 0) {
        dmlGraph->persistentBuffer = dmlGraph->device_interface->create_buffer(
            Type::of<void>() /*nullptr*/,
            dmlGraph->persist_resource_size,
            nullptr);
        // The persistent resource should be bound as the output to the IDMLOperatorInitializer.
        DML_BUFFER_BINDING bufferBinding{
            reinterpret_cast<ID3D12Resource *>(dmlGraph->persistentBuffer.native_handle),
            0, dmlGraph->persist_resource_size};
        DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
        initBindingTable->BindOutputs(1, &bindingDesc);
    }

    // The command recorder is a stateless object that records Dispatches into an existing Direct3D 12 command list.
    ThrowIfFailed(dmlGraph->dmlDevice->CreateCommandRecorder(
        IID_PPV_ARGS(dmlGraph->dmlCommandRecorder.GetAddressOf())));

    dmlGraph->dmlCommandRecorder->RecordDispatch(
        command_list,
        dmlOperatorInitializer.Get(),
        initBindingTable.Get());

    ThrowIfFailed(dmlGraph->dmlDevice->CreateBindingTable(
        &dmlBindingTableDesc,
        IID_PPV_ARGS(dmlGraph->dmlBindingTable.GetAddressOf())));
}

class DxGraphForwardCommand final : public DXCustomCmd {
    luisa::vector<ResourceUsage> resource_usages;
    luisa::span<const ResourceUsage> get_resource_usages() const noexcept override {
        return resource_usages;
    }
public:
    DxGraphForwardCommand(DxDMLGraph *graph, Argument::Buffer const &ipt, Argument::Buffer const &opt, Argument::Buffer const &w)
        : dmlGraph(graph),
          input(reinterpret_cast<lc::dx::DefaultBuffer *>(ipt.handle)->GetResource()),
          output(reinterpret_cast<lc::dx::DefaultBuffer *>(opt.handle)->GetResource()),
          weight(reinterpret_cast<lc::dx::DefaultBuffer *>(w.handle)->GetResource()) {
        if (ipt.size != graph->input_size) [[unlikely]] {
            LUISA_ERROR("Input buffer size {} mismatch. required {}", ipt.size, graph->input_size);
        }
        if (opt.size != graph->output_size) [[unlikely]] {
            LUISA_ERROR("Output buffer size {} mismatch. required {}", opt.size, graph->output_size);
        }
        if (w.size != graph->weight_size) [[unlikely]] {
            LUISA_ERROR("Weight buffer size {} mismatch. required {}", w.size, graph->weight_size);
        }
        resource_usages.emplace_back(
            ipt,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        resource_usages.emplace_back(
            opt,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        resource_usages.emplace_back(
            w,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        if (dmlGraph->temporaryBuffer.valid()) {
            resource_usages.emplace_back(
                Argument::Buffer{
                    .handle = dmlGraph->temporaryBuffer.handle,
                    .offset = 0,
                    .size = dmlGraph->temporaryBuffer.total_size_bytes},
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        if (dmlGraph->persistentBuffer.valid()) {
            resource_usages.emplace_back(
                Argument::Buffer{
                    .handle = dmlGraph->persistentBuffer.handle,
                    .offset = 0,
                    .size = dmlGraph->persistentBuffer.total_size_bytes},
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
    }
    [[nodiscard]] StreamTag stream_tag() const noexcept override {
        return StreamTag::COMPUTE;
    }

private:
    DxDMLGraph *dmlGraph;
    ID3D12Resource *input;
    ID3D12Resource *output;
    ID3D12Resource *weight;

    void execute(
        IDXGIAdapter1 *adapter,
        IDXGIFactory2 *dxgi_factory,
        ID3D12Device *device,
        ID3D12GraphicsCommandList4 *command_list) const noexcept override;
};

void DxGraphForwardCommand::execute(IDXGIAdapter1 *adapter, IDXGIFactory2 *dxgi_factory, ID3D12Device *device, ID3D12GraphicsCommandList4 *command_list) const noexcept {
    // const uint layer = dmlGraph->hiddens.size() + 1;

    uint data_size = dmlGraph->half ? 2 : 4;
    if (!dmlGraph->bind) {
        dmlGraph->bind = true;
        DML_BINDING_TABLE_DESC dmlBindingTableDesc{};
        dmlBindingTableDesc.Dispatchable = dmlGraph->dmlCompiledOperator.Get();
        dmlBindingTableDesc.CPUDescriptorHandle = dmlGraph->descriptorHeap->GetCPUDescriptorHandleForHeapStart();
        dmlBindingTableDesc.GPUDescriptorHandle = dmlGraph->descriptorHeap->GetGPUDescriptorHandleForHeapStart();
        dmlBindingTableDesc.SizeInDescriptors = dmlGraph->desc_count;
        dmlBindingTableDesc.Dispatchable = dmlGraph->dmlCompiledOperator.Get();
        ThrowIfFailed(dmlGraph->dmlBindingTable->Reset(&dmlBindingTableDesc));

        if (dmlGraph->temp_res_count != 0) {
            DML_BUFFER_BINDING bufferBinding{reinterpret_cast<ID3D12Resource *>(dmlGraph->temporaryBuffer.native_handle), 0, dmlGraph->temp_res_count};
            DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
            dmlGraph->dmlBindingTable->BindTemporaryResource(&bindingDesc);
        }
        if (dmlGraph->persist_resource_size != 0) {
            DML_BUFFER_BINDING bufferBinding{reinterpret_cast<ID3D12Resource *>(dmlGraph->persistentBuffer.native_handle), 0, dmlGraph->persist_resource_size};
            DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
            dmlGraph->dmlBindingTable->BindPersistentResource(&bindingDesc);
        }
        {

            vstd::vector<DML_BINDING_DESC> inputBindingDescs{};
            vstd::vector<DML_BUFFER_BINDING> inputBufferBindings{};
            inputBufferBindings.resize(dmlGraph->hiddens.size() + 2);
            inputBufferBindings[0] = DML_BUFFER_BINDING{input, 0, dmlGraph->input_size};
            inputBindingDescs.emplace_back(DML_BINDING_DESC{DML_BINDING_TYPE_BUFFER, &inputBufferBindings[0]});
            uint lastDim = dmlGraph->input;
            size_t offset = 0;
            for (uint i = 0; i < dmlGraph->hiddens.size(); i++) {
                auto hidden_dim = dmlGraph->hiddens[i];
                inputBufferBindings[i + 1] = DML_BUFFER_BINDING{weight, offset, size_t(lastDim) * hidden_dim * data_size};
                inputBindingDescs.emplace_back(DML_BINDING_DESC{DML_BINDING_TYPE_BUFFER, &inputBufferBindings[i + 1]});
                offset += inputBufferBindings[i + 1].SizeInBytes;
                lastDim = hidden_dim;
            }
            {
                inputBufferBindings[dmlGraph->hiddens.size() + 1] = DML_BUFFER_BINDING{weight, offset, size_t(lastDim) * dmlGraph->output * data_size};
                inputBindingDescs.emplace_back(DML_BINDING_DESC{DML_BINDING_TYPE_BUFFER, &inputBufferBindings[dmlGraph->hiddens.size() + 1]});
            }

            dmlGraph->dmlBindingTable->BindInputs(inputBindingDescs.size(), inputBindingDescs.data());
        }
        {
            DML_BUFFER_BINDING outputBufferBinding{output, 0, dmlGraph->output_size};
            DML_BINDING_DESC outputBindingDesc{DML_BINDING_TYPE_BUFFER, &outputBufferBinding};
            dmlGraph->dmlBindingTable->BindOutputs(1, &outputBindingDesc);
        }
    }
    ID3D12DescriptorHeap *d3D12DescriptorHeaps[] = {dmlGraph->descriptorHeap.Get()};
    command_list->SetDescriptorHeaps(vstd::array_count(d3D12DescriptorHeaps), d3D12DescriptorHeaps);

    //Dispatch the operator
    dmlGraph->dmlCommandRecorder->RecordDispatch(command_list, dmlGraph->dmlCompiledOperator.Get(), dmlGraph->dmlBindingTable.Get());
}

lc::dx::DxDirectMLExt::DxDirectMLExt(DeviceInterface *device) : device(device) {
}

luisa::unique_ptr<DMLGraph> lc::dx::DxDirectMLExt::create_graph(
    uint32_t batch_size,
    uint32_t input_elements,
    uint32_t out_elements,
    luisa::span<const uint32_t> hidden_layer_elements,
    luisa::span<const FusedActivation> activations,
    bool half_precision) noexcept {
    uint data_size = half_precision ? 2 : 4;
    size_t weight_size = 0;
    auto last_size = input_elements;
    for (auto &&i : hidden_layer_elements) {
        weight_size += last_size * i;
        last_size = i;
    }
    weight_size += last_size * out_elements;
    if (activations.size() != (hidden_layer_elements.size() + 1)) [[unlikely]] {
        LUISA_ERROR("Hidden layers' and activations' size mismatch.");
    }
    auto graph = luisa::make_unique<DxDMLGraph>(
        device,
        weight_size,
        out_elements,
        input_elements,
        batch_size,
        data_size,
        hidden_layer_elements,
        activations,
        half_precision);
    return graph;
}

unique_ptr<Command> DxDMLGraph::build() noexcept {
    return luisa::make_unique<DxGraphBuildCommand>(this);
}
unique_ptr<Command> DxDMLGraph::forward(Argument::Buffer input_buffer, Argument::Buffer output_buffer, Argument::Buffer weights_buffer) noexcept {
    return luisa::make_unique<DxGraphForwardCommand>(this, input_buffer, output_buffer, weights_buffer);
}
