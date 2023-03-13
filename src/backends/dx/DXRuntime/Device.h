#pragma once
#include <d3dx12.h>
#include <Resource/BufferView.h>
#include <vstl/v_guid.h>
#include <vstl/md5.h>
#include <dxgi1_4.h>
#include <core/binary_io.h>
#include <vstl/binary_reader.h>
#include <runtime/device.h>
#include <DxRuntime/DxPtr.h>
#include <ext_settings.h>
namespace luisa::compute {
class BinaryIO;
class Context;
}// namespace luisa::compute
class ElementAllocator;
using Microsoft::WRL::ComPtr;
namespace toolhub::directx {
struct ShaderPaths;
class GpuAllocator;
class DescriptorHeap;
class ComputeShader;
class PipelineLibrary;
class ShaderCompiler;
class BinaryStream : public luisa::compute::IBinaryStream, public vstd::IOperatorNewBase {
public:
    BinaryReader reader;
    BinaryStream(vstd::string const &path);
    size_t length() const override;
    size_t pos() const override;
    void read(luisa::span<std::byte> dst) override;
    ~BinaryStream();
};
struct SerializeVisitor : public luisa::compute::BinaryIO {
    ShaderPaths const &path;
    SerializeVisitor(
        ShaderPaths const &path) noexcept;
    luisa::unique_ptr<luisa::compute::IBinaryStream> Read(vstd::string const &filePath) noexcept;
    void Write(vstd::string const &filePath, luisa::span<std::byte const> data) noexcept;
    //	static vstd::string_view FileNameFilter(vstd::string_view path);
    luisa::unique_ptr<luisa::compute::IBinaryStream> read_shader_bytecode(luisa::string_view name) noexcept override;
    luisa::unique_ptr<luisa::compute::IBinaryStream> read_shader_cache(luisa::string_view name) noexcept override;
    luisa::unique_ptr<luisa::compute::IBinaryStream> read_internal_shader(luisa::string_view name) noexcept override;
    void write_shader_bytecode(luisa::string_view name, luisa::span<std::byte const> data) noexcept override;
    void write_shader_cache(luisa::string_view name, luisa::span<std::byte const> data) noexcept override;
    void write_internal_shader(luisa::string_view name, luisa::span<std::byte const> data) noexcept override;
};
class Device {
public:
    size_t maxAllocatorCount = 2;
    ShaderPaths const &path;
    std::atomic<luisa::compute::BinaryIO *> fileIo = nullptr;
    mutable SerializeVisitor serVisitor;
    struct LazyLoadShader {
    public:
        using LoadFunc = vstd::func_ptr_t<ComputeShader *(Device *, luisa::compute::BinaryIO *)>;

    private:
        vstd::unique_ptr<ComputeShader> shader;
        LoadFunc loadFunc;

    public:
        LazyLoadShader(LoadFunc loadFunc);
        ComputeShader *Get(Device *self);
        bool Check(Device *self);
        ~LazyLoadShader();
    };
    bool SupportMeshShader() const;
    vstd::MD5 adapterID;
    DxPtr<IDXGIAdapter1> adapter;
    DxPtr<ID3D12Device5> device;
    DxPtr<IDXGIFactory4> dxgiFactory;
    vstd::unique_ptr<GpuAllocator> defaultAllocator;
    vstd::unique_ptr<DescriptorHeap> globalHeap;
    vstd::unique_ptr<DescriptorHeap> samplerHeap;
    vstd::unique_ptr<luisa::compute::DirectXDeviceConfigExt> deviceSettings;
    LazyLoadShader setAccelKernel;

    LazyLoadShader bc6TryModeG10;
    LazyLoadShader bc6TryModeLE10;
    LazyLoadShader bc6EncodeBlock;

    LazyLoadShader bc7TryMode456;
    LazyLoadShader bc7TryMode137;
    LazyLoadShader bc7TryMode02;
    LazyLoadShader bc7EncodeBlock;

    /*vstd::unique_ptr<ComputeShader> bc6_0;
    vstd::unique_ptr<ComputeShader> bc6_1;
    vstd::unique_ptr<ComputeShader> bc6_2;
    vstd::unique_ptr<ComputeShader> bc7_0;
    vstd::unique_ptr<ComputeShader> bc7_1;
    vstd::unique_ptr<ComputeShader> bc7_2;
    vstd::unique_ptr<ComputeShader> bc7_3;*/
    Device(luisa::compute::Context &ctx, ShaderPaths const &path, luisa::compute::DeviceConfig const *settings);
    Device(Device const &) = delete;
    Device(Device &&) = delete;
    ~Device();
    void WaitFence(ID3D12Fence *fence, uint64 fenceIndex);
    static ShaderCompiler *Compiler();
};
}// namespace toolhub::directx