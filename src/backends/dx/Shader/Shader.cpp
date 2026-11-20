#include <luisa/core/magic_enum.h>
#include <Shader/Shader.h>
#include <d3dcompiler.h>
#include <DXRuntime/CommandBuffer.h>
#include <DXRuntime/CommandAllocator.h>
#include <DXRuntime/Device.h>
#include <DXRuntime/GlobalSamplers.h>
#include <Resource/TopAccel.h>
#include <Resource/DefaultBuffer.h>
#include <Shader/ShaderSerializer.h>
#include <luisa/core/logging.h>

namespace lc::dx {

void Shader::_init_root_sig_layout(bool append_object_id) {
    auto layout = build_root_sig_layout(
        _properties, _kernel_arguments, !_printers.empty(), append_object_id);
    _spilled_tables = layout.spilled_tables;
    _spilled_srv_root_param = layout.spilled_srv_root_param;
    _spilled_uav_root_param = layout.spilled_uav_root_param;
    _spilled_srv_count = layout.spilled_srv_count;
    _spilled_uav_count = layout.spilled_uav_count;
    _root_sig_bindings = std::move(layout.bindings);
    LUISA_ASSERT(
        layout.mapping_complete,
        "DX descriptor-table spill failed to map all shader properties "
        "(properties={}, kernel_args={}, prop_offset={}, has_printers={}).",
        _properties.size(), _kernel_arguments.size(),
        layout.kernel_arg_prop_offset, !_printers.empty());
}

Shader::Shader(
    vstd::vector<hlsl::Property> &&prop,
    vstd::vector<SavedArgument> &&args,
    ComPtr<ID3D12RootSignature> &&root_sig,
    vstd::vector<std::pair<vstd::string, Type const *>> &&printers,
    uint validation_count)
    : _root_sig(std::move(root_sig)),
      _properties(std::move(prop)),
      _kernel_arguments(std::move(args)),
      _printers(std::move(printers)),
      _validation_count(validation_count) {
    _init_root_sig_layout(false);
}

Shader::Shader(
    vstd::vector<hlsl::Property> &&prop,
    vstd::vector<SavedArgument> &&args,
    ID3D12Device *device,
    vstd::vector<std::pair<vstd::string, Type const *>> &&printers,
    uint validation_count,
    bool isRaster)
    : _properties(std::move(prop)),
      _kernel_arguments(std::move(args)),
      _printers(std::move(printers)),
      _validation_count(validation_count) {
    _init_root_sig_layout(isRaster);
    auto serializedRootSig = ShaderSerializer::SerializeRootSig(
        _properties,
        isRaster);
    ThrowIfFailed(device->CreateRootSignature(
        0,
        serializedRootSig->GetBufferPointer(),
        serializedRootSig->GetBufferSize(),
        IID_PPV_ARGS(_root_sig.GetAddressOf())));
}

void Shader::set_compute_resource(
    uint property_name,
    CommandBufferBuilder *cb,
    BufferView buffer) const {
    auto cmd_list = cb->get_cb()->cmd_list();
    auto &&var = _properties[property_name];
    switch (var.type) {
        case hlsl::ShaderVariableType::ConstantBuffer: {
            cmd_list->SetComputeRootConstantBufferView(
                property_name,
                buffer.buffer->GetAddress() + buffer.offset);
        } break;
        case hlsl::ShaderVariableType::StructuredBuffer: {
            cmd_list->SetComputeRootShaderResourceView(
                property_name,
                buffer.buffer->GetAddress() + buffer.offset);
        } break;
        case hlsl::ShaderVariableType::RWStructuredBuffer: {
            cmd_list->SetComputeRootUnorderedAccessView(
                property_name,
                buffer.buffer->GetAddress() + buffer.offset);
        } break;
        default:
            LUISA_ERROR("Invalid shader resource type {}.\n\n"
                        "This might be due to the change of shader cache.\n"
                        "Please try delete the cache folder (default: build_dir/bin/.cache) "
                        "and re-run the program.\n"
                        "If the problem persists, please report this issue to the developers.\n",
                        to_string(var.type));
    }
}
void Shader::set_compute_resource(
    uint property_name,
    CommandBufferBuilder *cb,
    DescriptorHeapView view) const {
    auto cmd_list = cb->get_cb()->cmd_list();
    auto &&var = _properties[property_name];
    switch (var.type) {
        case hlsl::ShaderVariableType::UAVBufferHeap:
        case hlsl::ShaderVariableType::UAVTextureHeap:
        case hlsl::ShaderVariableType::CBVBufferHeap:
        case hlsl::ShaderVariableType::SamplerHeap:
        case hlsl::ShaderVariableType::SRVBufferHeap:
        case hlsl::ShaderVariableType::SRVTextureHeap: {
            cmd_list->SetComputeRootDescriptorTable(
                property_name,
                view.heap->hGPU(view.index));
        } break;
        default: LUISA_ASSUME(false); break;
    }
}
void Shader::set_compute_resource(
    uint property_name,
    CommandBufferBuilder *cb,
    std::pair<uint, uint4> const &const_value) const {
    auto cmd_list = cb->get_cb()->cmd_list();
    LUISA_ASSUME(_properties[property_name].type == hlsl::ShaderVariableType::ConstantValue);
    cmd_list->SetComputeRoot32BitConstants(property_name, const_value.first, &const_value.second, 0);
}
void Shader::set_compute_resource(
    uint property_name,
    CommandBufferBuilder *cmd_list,
    TopAccel const *bAccel) const {
    return set_compute_resource(
        property_name,
        cmd_list,
        BufferView(bAccel->GetAccelBuffer()));
}
void Shader::set_raster_resource(
    uint property_name,
    CommandBufferBuilder *cb,
    BufferView buffer) const {
    auto cmd_list = cb->get_cb()->cmd_list();
    auto &&var = _properties[property_name];
    switch (var.type) {
        case hlsl::ShaderVariableType::ConstantBuffer: {
            cmd_list->SetGraphicsRootConstantBufferView(
                property_name,
                buffer.buffer->GetAddress() + buffer.offset);
        } break;
        case hlsl::ShaderVariableType::StructuredBuffer: {
            cmd_list->SetGraphicsRootShaderResourceView(
                property_name,
                buffer.buffer->GetAddress() + buffer.offset);
        } break;
        case hlsl::ShaderVariableType::RWStructuredBuffer: {
            cmd_list->SetGraphicsRootUnorderedAccessView(
                property_name,
                buffer.buffer->GetAddress() + buffer.offset);
        } break;
        default: LUISA_ASSUME(false); break;
    }
}
void Shader::set_raster_resource(
    uint property_name,
    CommandBufferBuilder *cb,
    DescriptorHeapView view) const {
    auto cmd_list = cb->get_cb()->cmd_list();
    auto &&var = _properties[property_name];
    switch (var.type) {
        case hlsl::ShaderVariableType::UAVBufferHeap:
        case hlsl::ShaderVariableType::UAVTextureHeap:
        case hlsl::ShaderVariableType::CBVBufferHeap:
        case hlsl::ShaderVariableType::SamplerHeap:
        case hlsl::ShaderVariableType::SRVBufferHeap:
        case hlsl::ShaderVariableType::SRVTextureHeap: {
            cmd_list->SetGraphicsRootDescriptorTable(
                property_name,
                view.heap->hGPU(view.index));
        } break;
        default: LUISA_ASSUME(false); break;
    }
}
void Shader::set_raster_resource(
    uint property_name,
    CommandBufferBuilder *cmd_list,
    TopAccel const *bAccel) const {
    return set_raster_resource(
        property_name,
        cmd_list,
        BufferView(bAccel->GetAccelBuffer()));
}
void Shader::set_raster_resource(
    uint property_name,
    CommandBufferBuilder *cb,
    std::pair<uint, uint4> const &const_value) const {
    auto cmd_list = cb->get_cb()->cmd_list();
    LUISA_ASSUME(_properties[property_name].type == hlsl::ShaderVariableType::ConstantValue);
    cmd_list->SetGraphicsRoot32BitConstants(property_name, const_value.first, &const_value.second, 0);
}
namespace {

void write_spilled_srv_descriptor(
    DescriptorHeap *heap,
    uint slot,
    Shader::RootSigBinding const &binding,
    BindProperty const &prop,
    bool sm69) {
    prop.visit([&]<typename T>(T const &val) {
        if constexpr (std::is_same_v<T, TopAccel const *>) {
            D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
            desc.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
            desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            desc.RaytracingAccelerationStructure.Location = val->GetAccelBuffer()->GetAddress();
            heap->CreateSRV(nullptr, desc, slot);
        } else if constexpr (std::is_same_v<T, DescriptorHeapView>) {
            LUISA_ERROR("Unexpected DescriptorHeapView in spilled SRV table slot {}.", slot);
        } else if constexpr (std::is_same_v<T, BufferView>) {
            const uint64 total_bytes = val.buffer->GetByteSize();
            const uint64 avail_bytes = val.offset < total_bytes ? (total_bytes - val.offset) : 0ull;
            const uint64 view_bytes = val.byteSize > 0ull ? std::min<uint64>(val.byteSize, avail_bytes) : avail_bytes;
            const bool is_raw = binding.force_raw || binding.element_stride == 0u ||
                                (binding.element_stride == 1u && !sm69);
            D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
            desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            if (is_raw) {
                desc.Format = DXGI_FORMAT_R32_TYPELESS;
                desc.Buffer.FirstElement = val.offset / 4ull;
                const uint64 max_elems = avail_bytes / 4ull;
                const uint64 req_elems = (view_bytes + 3ull) / 4ull;
                desc.Buffer.NumElements = static_cast<UINT>(std::min<uint64>(
                    std::max<uint64>(1ull, std::min<uint64>(req_elems, max_elems)),
                    1073741536ull - desc.Buffer.FirstElement));
                desc.Buffer.StructureByteStride = 0u;
                desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
            } else {
                const uint64 stride = binding.element_stride;
                desc.Format = DXGI_FORMAT_UNKNOWN;
                desc.Buffer.FirstElement = val.offset / stride;
                const uint64 elems = std::max<uint64>(1ull, view_bytes / stride);
                desc.Buffer.NumElements = static_cast<UINT>(elems);
                desc.Buffer.StructureByteStride = static_cast<UINT>(stride);
                desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
            }
            heap->CreateSRV(val.buffer->GetResource(), desc, slot);
        }
    });
}

void write_spilled_uav_descriptor(
    DescriptorHeap *heap,
    uint slot,
    Shader::RootSigBinding const &binding,
    BindProperty const &prop,
    bool sm69) {
    prop.visit([&]<typename T>(T const &val) {
        if constexpr (std::is_same_v<T, DescriptorHeapView>) {
            LUISA_ERROR("Unexpected DescriptorHeapView in spilled UAV table slot {}.", slot);
        } else if constexpr (std::is_same_v<T, BufferView>) {
            const uint64 total_bytes = val.buffer->GetByteSize();
            const uint64 avail_bytes = val.offset < total_bytes ? (total_bytes - val.offset) : 0ull;
            const uint64 view_bytes = val.byteSize > 0ull ? std::min<uint64>(val.byteSize, avail_bytes) : avail_bytes;
            const bool is_raw = binding.force_raw || binding.element_stride == 0u ||
                                (binding.element_stride == 1u && !sm69);
            D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
            desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            desc.Buffer.CounterOffsetInBytes = 0ull;
            if (is_raw) {
                desc.Format = DXGI_FORMAT_R32_TYPELESS;
                desc.Buffer.FirstElement = val.offset / 4ull;
                const uint64 max_elems = avail_bytes / 4ull;
                const uint64 req_elems = (view_bytes + 3ull) / 4ull;
                desc.Buffer.NumElements = static_cast<UINT>(std::min<uint64>(
                    std::max<uint64>(1ull, std::min<uint64>(req_elems, max_elems)),
                    1073741536ull - desc.Buffer.FirstElement));
                desc.Buffer.StructureByteStride = 0u;
                desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
            } else {
                const uint64 stride = binding.element_stride;
                desc.Format = DXGI_FORMAT_UNKNOWN;
                desc.Buffer.FirstElement = val.offset / stride;
                const uint64 elems = std::max<uint64>(1ull, view_bytes / stride);
                desc.Buffer.NumElements = static_cast<UINT>(elems);
                desc.Buffer.StructureByteStride = static_cast<UINT>(stride);
                desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
            }
            heap->CreateUAV(val.buffer->GetResource(), desc, slot);
        }
    });
}

}// namespace

void Shader::bind_compute_resources(
    CommandBufferBuilder *cb,
    vstd::span<const BindProperty> resources) const {
    LUISA_ASSUME(resources.size() == _properties.size());
    if (!_spilled_tables) {
        for (auto i : vstd::range(static_cast<int64>(resources.size()))) {
            resources[i].visit([&](auto &&b) {
                set_compute_resource(i, cb, b);
            });
        }
        return;
    }
    auto *alloc = cb->get_cb()->get_alloc();
    auto *device = alloc->device();
    auto *heap = device->global_heap.get();
    auto *cmd_list = cb->get_cb()->cmd_list();
    const bool sm69 = device->feature_check.shader_model_6_9_supported();
    const uint total_spilled = _spilled_srv_count + _spilled_uav_count;
    uint base = 0u;
    if (total_spilled > 0u) {
        auto *node = heap->SubAllocate(total_spilled);
        base = heap->GetSubAllocOffset(node);
        alloc->execute_after_complete([heap, node]() {
            heap->DeAllocate(node);
        });
    }
    for (size_t i = 0; i < resources.size(); ++i) {
        auto const &binding = _root_sig_bindings[i];
        switch (binding.kind) {
            case RootSigBinding::Kind::Direct: {
                const uint root_idx = binding.index;
                auto const &var = _properties[i];
                resources[i].visit([&]<typename T>(T const &b) {
                    if constexpr (std::is_same_v<T, BufferView>) {
                        switch (var.type) {
                            case hlsl::ShaderVariableType::ConstantBuffer:
                                cmd_list->SetComputeRootConstantBufferView(root_idx, b.buffer->GetAddress() + b.offset);
                                break;
                            case hlsl::ShaderVariableType::StructuredBuffer:
                                cmd_list->SetComputeRootShaderResourceView(root_idx, b.buffer->GetAddress() + b.offset);
                                break;
                            case hlsl::ShaderVariableType::RWStructuredBuffer:
                                cmd_list->SetComputeRootUnorderedAccessView(root_idx, b.buffer->GetAddress() + b.offset);
                                break;
                            default: LUISA_ASSUME(false); break;
                        }
                    } else if constexpr (std::is_same_v<T, DescriptorHeapView>) {
                        cmd_list->SetComputeRootDescriptorTable(root_idx, b.heap->hGPU(b.index));
                    } else if constexpr (std::is_same_v<T, TopAccel const *>) {
                        cmd_list->SetComputeRootShaderResourceView(root_idx, b->GetAccelBuffer()->GetAddress());
                    } else if constexpr (std::is_same_v<T, std::pair<uint, uint4>>) {
                        cmd_list->SetComputeRoot32BitConstants(root_idx, b.first, &b.second, 0);
                    }
                });
            } break;
            case RootSigBinding::Kind::SpilledSRV:
                write_spilled_srv_descriptor(heap, base + binding.index, binding, resources[i], sm69);
                break;
            case RootSigBinding::Kind::SpilledUAV:
                write_spilled_uav_descriptor(heap, base + _spilled_srv_count + binding.index, binding, resources[i], sm69);
                break;
        }
    }
    if (_spilled_srv_count > 0u) {
        cmd_list->SetComputeRootDescriptorTable(_spilled_srv_root_param, heap->hGPU(base));
    }
    if (_spilled_uav_count > 0u) {
        cmd_list->SetComputeRootDescriptorTable(_spilled_uav_root_param, heap->hGPU(base + _spilled_srv_count));
    }
}

void Shader::bind_raster_resources(
    CommandBufferBuilder *cb,
    vstd::span<const BindProperty> resources) const {
    LUISA_ASSUME(resources.size() == _properties.size());
    if (!_spilled_tables) {
        for (auto i : vstd::range(static_cast<int64>(resources.size()))) {
            resources[i].visit([&](auto &&b) {
                set_raster_resource(i, cb, b);
            });
        }
        return;
    }
    auto *alloc = cb->get_cb()->get_alloc();
    auto *device = alloc->device();
    auto *heap = device->global_heap.get();
    auto *cmd_list = cb->get_cb()->cmd_list();
    const bool sm69 = device->feature_check.shader_model_6_9_supported();
    const uint total_spilled = _spilled_srv_count + _spilled_uav_count;
    uint base = 0u;
    if (total_spilled > 0u) {
        auto *node = heap->SubAllocate(total_spilled);
        base = heap->GetSubAllocOffset(node);
        alloc->execute_after_complete([heap, node]() {
            heap->DeAllocate(node);
        });
    }
    for (size_t i = 0; i < resources.size(); ++i) {
        auto const &binding = _root_sig_bindings[i];
        switch (binding.kind) {
            case RootSigBinding::Kind::Direct: {
                const uint root_idx = binding.index;
                auto const &var = _properties[i];
                resources[i].visit([&]<typename T>(T const &b) {
                    if constexpr (std::is_same_v<T, BufferView>) {
                        switch (var.type) {
                            case hlsl::ShaderVariableType::ConstantBuffer:
                                cmd_list->SetGraphicsRootConstantBufferView(root_idx, b.buffer->GetAddress() + b.offset);
                                break;
                            case hlsl::ShaderVariableType::StructuredBuffer:
                                cmd_list->SetGraphicsRootShaderResourceView(root_idx, b.buffer->GetAddress() + b.offset);
                                break;
                            case hlsl::ShaderVariableType::RWStructuredBuffer:
                                cmd_list->SetGraphicsRootUnorderedAccessView(root_idx, b.buffer->GetAddress() + b.offset);
                                break;
                            default: LUISA_ASSUME(false); break;
                        }
                    } else if constexpr (std::is_same_v<T, DescriptorHeapView>) {
                        cmd_list->SetGraphicsRootDescriptorTable(root_idx, b.heap->hGPU(b.index));
                    } else if constexpr (std::is_same_v<T, TopAccel const *>) {
                        cmd_list->SetGraphicsRootShaderResourceView(root_idx, b->GetAccelBuffer()->GetAddress());
                    } else if constexpr (std::is_same_v<T, std::pair<uint, uint4>>) {
                        cmd_list->SetGraphicsRoot32BitConstants(root_idx, b.first, &b.second, 0);
                    }
                });
            } break;
            case RootSigBinding::Kind::SpilledSRV:
                write_spilled_srv_descriptor(heap, base + binding.index, binding, resources[i], sm69);
                break;
            case RootSigBinding::Kind::SpilledUAV:
                write_spilled_uav_descriptor(heap, base + _spilled_srv_count + binding.index, binding, resources[i], sm69);
                break;
        }
    }
    if (_spilled_srv_count > 0u) {
        cmd_list->SetGraphicsRootDescriptorTable(_spilled_srv_root_param, heap->hGPU(base));
    }
    if (_spilled_uav_count > 0u) {
        cmd_list->SetGraphicsRootDescriptorTable(_spilled_uav_root_param, heap->hGPU(base + _spilled_srv_count));
    }
}

void Shader::_save_pso(ID3D12PipelineState *pso, vstd::string_view pso_name, luisa::BinaryIO const *file_stream, Device const *device) const {
    LUISA_VERBOSE("Write Pipeline cache to {}.", pso_name);
    ComPtr<ID3DBlob> psoCache;
    pso->GetCachedBlob(&psoCache);
    static_cast<void>(file_stream->write_shader_cache(
        pso_name,
        {reinterpret_cast<std::byte const *>(psoCache->GetBufferPointer()),
         psoCache->GetBufferSize()}));
};
vstd::string Shader::pso_name(Device const *device, vstd::string_view file_name) {
    vstd::fixed_vector<uint8_t, 64> data;
    luisa::enlarge_by(data, 16 + file_name.size());
    std::memcpy(data.data(), &device->adapter_id, 16);
    std::memcpy(data.data() + 16, file_name.data(), file_name.size());
    vstd::MD5 hash{data};
    return hash.to_string(false) + ".dx";
}

}// namespace lc::dx
