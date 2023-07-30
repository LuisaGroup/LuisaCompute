#include "dml_ext.h"
#include "LCCmdBuffer.h"
#include <luisa/runtime/stream.h>
#define _D3D12MA_IUNKNOWN_IMPL_FUNCTIONS
#include "DirectMLX.h"
#include <luisa/backends/ext/dx_custom_cmd.h>
#include <wrl/client.h>
#include <Resource/DefaultBuffer.h>
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
    ComPtr<IDMLDevice> dmlDevice;
    ComPtr<IDMLCompiledOperator> dmlCompiledOperator;

    ComPtr<IDMLBindingTable> dmlBindingTable;
    ComPtr<IDMLCommandRecorder> dmlCommandRecorder;
    ComPtr<ID3D12DescriptorHeap> descriptorHeap;
    size_t weightSize;
    size_t outputSize;
    size_t inputSize;
    size_t descriptorCount;
    size_t temporaryResourceSize;
    size_t persistentResourceSize;

    int layer;
    int input;
    int output;
    int hiddenDim;

    bool bind = false;
    bool half;

    ComPtr<ID3D12Resource> temporaryBuffer;
    ComPtr<ID3D12Resource> persistentBuffer;
    unique_ptr<Command> build(int batchSize, int input, int layer, int hiddenDim, int output) noexcept override;
    unique_ptr<Command> forward(Argument::Buffer input_buffer, Argument::Buffer output_buffer, Argument::Buffer weights_buffer) noexcept override;
};
class DxGraphBuildCommand final : public DXCustomCmd {
public:
    DxGraphBuildCommand(DxDMLGraph *graph, int batchSize, int input, int layer, int hiddenDim, int output) : dmlGraph(graph), batchSize(batchSize), input(input), layer(layer), hiddenDim(hiddenDim), output(output) {
        graph->layer = layer;
        graph->input = input;
        graph->output = output;
        graph->hiddenDim = hiddenDim;
    }
    [[nodiscard]] StreamTag stream_tag() const noexcept override {
        return StreamTag::COMPUTE;
    }

private:
    DxDMLGraph *dmlGraph;
    int batchSize;
    int input;
    int layer;
    int hiddenDim;
    int output;

    void execute(
        IDXGIAdapter1 *adapter,
        IDXGIFactory2 *dxgi_factory,
        ID3D12Device *device,
        ID3D12GraphicsCommandList4 *command_list) const noexcept override;
};

void DxGraphBuildCommand::execute(IDXGIAdapter1 *adapter, IDXGIFactory2 *dxgi_factory, ID3D12Device *device, ID3D12GraphicsCommandList4 *command_list) const noexcept {
    unsigned int dataSize = dmlGraph->half ? 2 : 4;
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
    UINT tensorSizes[4] = {1, 1, UINT(batchSize), UINT(input)};
    dml::TensorDesc::Dimensions inputDimensions(std::begin(tensorSizes), std::end(tensorSizes));
    dml::TensorDesc desc = {dataType, inputDimensions};
    dml::Expression inputLayer = dml::InputTensor(graph, 0, desc);

    vstd::vector<dml::Expression> expressions{};
    int lastDim = input;
    auto &lastOutput = inputLayer;
    for (int i = 0; i < layer; i++) {
        UINT matrixSizes[4] = {1, 1, UINT(hiddenDim), UINT(lastDim)};
        dml::TensorDesc::Dimensions matrixDimensions = dml::TensorDesc::Dimensions(std::begin(matrixSizes), std::end(matrixSizes));
        auto mdesc = dml::TensorDesc{dataType, matrixDimensions};
        dml::Expression &weights = expressions.emplace_back(dml::InputTensor(graph, i + 1, mdesc));
        dml::Expression &fc = expressions.emplace_back(
            dml::Gemm(lastOutput, weights,
                      dml::NullOpt, DML_MATRIX_TRANSFORM_NONE, DML_MATRIX_TRANSFORM_TRANSPOSE, 1.f, 1.f, dml::FusedActivation::Relu()));
        lastDim = hiddenDim;
        lastOutput = fc;
    }
    {
        UINT matrixSizes[4] = {1, 1, UINT(output), UINT(lastDim)};
        dml::TensorDesc::Dimensions matrixDimensions = dml::TensorDesc::Dimensions(std::begin(matrixSizes), std::end(matrixSizes));
        auto mdesc = dml::TensorDesc{dataType, matrixDimensions};
        dml::Expression &weights = expressions.emplace_back(dml::InputTensor(graph, layer + 1, mdesc));
        dml::Expression &fc = expressions.emplace_back(dml::Gemm(lastOutput, weights, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE, DML_MATRIX_TRANSFORM_TRANSPOSE));
        lastDim = hiddenDim;
        lastOutput = fc;
    }
    int numWeights = input * hiddenDim + hiddenDim * hiddenDim * (layer) + hiddenDim * output;
    if (layer == 0) {
        numWeights = input * output;
    }
    dmlGraph->weightSize = numWeights * dataSize;
    dmlGraph->outputSize = output * batchSize * dataSize;
    dmlGraph->inputSize = input * batchSize * dataSize;

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
    dmlGraph->descriptorCount = std::max(
        initializeBindingProperties.RequiredDescriptorCount,
        executeBindingProperties.RequiredDescriptorCount);

    // Create descriptor heaps.

    D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc{};
    descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    descriptorHeapDesc.NumDescriptors = dmlGraph->descriptorCount;
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
    dmlBindingTableDesc.SizeInDescriptors = dmlGraph->descriptorCount;

    ComPtr<IDMLBindingTable> initBindingTable;
    ThrowIfFailed(dmlGraph->dmlDevice->CreateBindingTable(
        &dmlBindingTableDesc,
        IID_PPV_ARGS(initBindingTable.GetAddressOf())));

    // Create the temporary and persistent resources that are necessary for executing an operator.

    // The temporary resource is scratch memory (used internally by DirectML), whose contents you don't need to define.
    // The persistent resource is long-lived, and you need to initialize it using the IDMLOperatorInitializer.

    dmlGraph->temporaryResourceSize = std::max(
        initializeBindingProperties.TemporaryResourceSize,
        executeBindingProperties.TemporaryResourceSize);
    dmlGraph->persistentResourceSize = executeBindingProperties.PersistentResourceSize;

    // Bind and initialize the operator on the GPU.

    auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    if (dmlGraph->temporaryResourceSize != 0) {
        auto bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(dmlGraph->temporaryResourceSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        ThrowIfFailed(device->CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &bufferDesc,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(dmlGraph->temporaryBuffer.GetAddressOf())));

        if (initializeBindingProperties.TemporaryResourceSize != 0) {
            DML_BUFFER_BINDING bufferBinding{dmlGraph->temporaryBuffer.Get(), 0, dmlGraph->temporaryResourceSize};
            DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
            initBindingTable->BindTemporaryResource(&bindingDesc);
        }
    }

    if (dmlGraph->persistentResourceSize != 0) {
        auto bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(dmlGraph->persistentResourceSize);
        ThrowIfFailed(device->CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &bufferDesc,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(dmlGraph->persistentBuffer.GetAddressOf())));

        // The persistent resource should be bound as the output to the IDMLOperatorInitializer.
        DML_BUFFER_BINDING bufferBinding{dmlGraph->persistentBuffer.Get(), 0, dmlGraph->persistentResourceSize};
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
public:
    DxGraphForwardCommand(DxDMLGraph *graph, Argument::Buffer const &ipt, Argument::Buffer const &opt, Argument::Buffer const &w)
        : dmlGraph(graph),
          input(reinterpret_cast<lc::dx::DefaultBuffer *>(ipt.handle)->GetResource()),
          output(reinterpret_cast<lc::dx::DefaultBuffer *>(opt.handle)->GetResource()),
          weight(reinterpret_cast<lc::dx::DefaultBuffer *>(w.handle)->GetResource()) {
        resource_usages.emplace_back(
            ipt,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        resource_usages.emplace_back(
            opt,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        resource_usages.emplace_back(
            w,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
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
    unsigned int dataSize = dmlGraph->half ? 2 : 4;
    if (!dmlGraph->bind) {
        dmlGraph->bind = true;
        DML_BINDING_TABLE_DESC dmlBindingTableDesc{};
        dmlBindingTableDesc.Dispatchable = dmlGraph->dmlCompiledOperator.Get();
        dmlBindingTableDesc.CPUDescriptorHandle = dmlGraph->descriptorHeap->GetCPUDescriptorHandleForHeapStart();
        dmlBindingTableDesc.GPUDescriptorHandle = dmlGraph->descriptorHeap->GetGPUDescriptorHandleForHeapStart();
        dmlBindingTableDesc.SizeInDescriptors = dmlGraph->descriptorCount;
        dmlBindingTableDesc.Dispatchable = dmlGraph->dmlCompiledOperator.Get();
        ThrowIfFailed(dmlGraph->dmlBindingTable->Reset(&dmlBindingTableDesc));

        if (dmlGraph->temporaryResourceSize != 0) {
            DML_BUFFER_BINDING bufferBinding{dmlGraph->temporaryBuffer.Get(), 0, dmlGraph->temporaryResourceSize};
            DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
            dmlGraph->dmlBindingTable->BindTemporaryResource(&bindingDesc);
        }
        if (dmlGraph->persistentResourceSize != 0) {
            DML_BUFFER_BINDING bufferBinding{dmlGraph->persistentBuffer.Get(), 0, dmlGraph->persistentResourceSize};
            DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
            dmlGraph->dmlBindingTable->BindPersistentResource(&bindingDesc);
        }
        {

            vstd::vector<DML_BINDING_DESC> inputBindingDescs{};
            vstd::vector<DML_BUFFER_BINDING> inputBufferBindings{};
            inputBufferBindings.resize(dmlGraph->layer + 2);
            inputBufferBindings[0] = DML_BUFFER_BINDING{input, 0, dmlGraph->inputSize};
            inputBindingDescs.emplace_back(DML_BINDING_DESC{DML_BINDING_TYPE_BUFFER, &inputBufferBindings[0]});
            int lastDim = dmlGraph->input;
            size_t offset = 0;
            for (int i = 0; i < dmlGraph->layer; i++) {
                inputBufferBindings[i + 1] = DML_BUFFER_BINDING{weight, offset, size_t(lastDim) * dmlGraph->hiddenDim * dataSize};
                inputBindingDescs.emplace_back(DML_BINDING_DESC{DML_BINDING_TYPE_BUFFER, &inputBufferBindings[i + 1]});
                offset += inputBufferBindings[i + 1].SizeInBytes;
                lastDim = dmlGraph->hiddenDim;
            }
            {
                inputBufferBindings[dmlGraph->layer + 1] = DML_BUFFER_BINDING{weight, offset, size_t(lastDim) * dmlGraph->output * dataSize};
                inputBindingDescs.emplace_back(DML_BINDING_DESC{DML_BINDING_TYPE_BUFFER, &inputBufferBindings[dmlGraph->layer + 1]});
            }

            dmlGraph->dmlBindingTable->BindInputs(inputBindingDescs.size(), inputBindingDescs.data());
        }
        {
            auto bt = CD3DX12_RESOURCE_BARRIER::UAV(input);
            command_list->ResourceBarrier(
                1,
                &bt);
        }
        {
            DML_BUFFER_BINDING outputBufferBinding{output, 0, dmlGraph->outputSize};
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

luisa::unique_ptr<DMLGraph> lc::dx::DxDirectMLExt::create_graph(bool half) noexcept {
    auto graph = luisa::make_unique<DxDMLGraph>();
    graph->half = half;
    return graph;
}

unique_ptr<Command> DxDMLGraph::build(int batchSize, int input, int layer, int hiddenDim, int output) noexcept {
    return luisa::make_unique<DxGraphBuildCommand>(this, batchSize, input, layer, hiddenDim, output);
}
unique_ptr<Command> DxDMLGraph::forward(Argument::Buffer input_buffer, Argument::Buffer output_buffer, Argument::Buffer weights_buffer) noexcept {
    return luisa::make_unique<DxGraphForwardCommand>(this, input_buffer, output_buffer, weights_buffer);
}
