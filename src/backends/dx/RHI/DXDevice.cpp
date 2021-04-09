#include <codecvt>
#include <Common/GFXUtil.h>
#include <runtime/device.h>
#include <RenderComponent/RenderComponentInclude.h>
#include "DXStream.hpp"
namespace luisa::compute {
using namespace Microsoft::WRL;

static GFXFormat LCFormatToVEngineFormat(PixelFormat format) {
	switch (format) {
		//TODO:
		default:
			return GFXFormat_R8G8B8A8_SNorm;
	}
}
class DXDevice final : public Device {
public:
	DXDevice(const Context& ctx, uint32_t index) : Device(ctx) {// TODO: support device selection?
		InitD3D(index);
		dxDevice = md3dDevice.Get();
	}
	uint64 create_buffer(size_t size_bytes) noexcept override {
		return reinterpret_cast<uint64>(
			new StructuredBuffer(
				dxDevice,
				{StructuredBufferElement::Get(1, size_bytes)},
				GPUResourceState_Common,
				nullptr//TODO: allocator
				));
	}
	void dispose_buffer(uint64 handle) noexcept override {
		delete reinterpret_cast<StructuredBuffer*>(handle);
	}

	// texture
	uint64 create_texture(
		PixelFormat format, uint dimension, uint width, uint height, uint depth,
		uint mipmap_levels, bool is_bindless) override {
		return reinterpret_cast<uint64>(
			new RenderTexture(
				dxDevice,
				nullptr,//TODO: allocator
				width,
				height,
				RenderTextureFormat::GetColorFormat(LCFormatToVEngineFormat(format)),
				dimension > 2 ? TextureDimension::Tex3D : TextureDimension::Tex2D,
				depth,
				mipmap_levels,
				RenderTextureState::Common,
				0));
	}
	void dispose_texture(uint64 handle) noexcept override {
		delete reinterpret_cast<RenderTexture*>(handle);
	}

	// stream
	uint64 create_stream() noexcept override {
		return reinterpret_cast<uint64>(
			new DXStream(dxDevice, GFXCommandListType_Compute)//TODO: need support copy
		);
	}
	void dispose_stream(uint64 handle) noexcept override {
		synchronize_stream(handle);
		delete reinterpret_cast<DXStream*>(handle);
	}
	void synchronize_stream(uint64 stream_handle) noexcept override {
		DXStream* stream = reinterpret_cast<DXStream*>(stream_handle);
		stream->Sync(cpuFence.Get());
	}
	void dispatch(uint64 stream_handle, CommandBuffer cmd_buffer) noexcept override {

	}
	// kernel
	void prepare_kernel(uint32_t uid) noexcept override {
		// do async compile here...
	}
	uint64_t create_event() noexcept override {
		return 0;
	}
	void dispose_event(uint64_t handle) noexcept override {}
	void synchronize_event(uint64_t handle) noexcept override {}

	~DXDevice() {
	}
	//////////// Variables
	GFXDevice* dxDevice{nullptr};
	DECLARE_VENGINE_OVERRIDE_OPERATOR_NEW
private:
	///////////// D3D
	void InitD3D(uint32_t index) {
#if defined(DEBUG) || defined(_DEBUG)
		// Enable the D3D12 debug layer.
		{
			ComPtr<ID3D12Debug> debugController;
			ThrowIfFailed(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)));
			debugController->EnableDebugLayer();
		}
#endif
		ThrowIfFailed(CreateDXGIFactory1(IID_PPV_ARGS(&mdxgiFactory)));
		auto suitableIndex = 0u;
		int32_t adapterIndex = 0; // we'll start looking for directx 12  compatible graphics devices starting at index 0
		bool adapterFound = false;// set this to true when a good one was found
		while (mdxgiFactory->EnumAdapters1(adapterIndex, &adapter) != DXGI_ERROR_NOT_FOUND) {
			DXGI_ADAPTER_DESC1 desc;
			adapter->GetDesc1(&desc);
			if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0) {
				HRESULT hr = D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_1,
											   IID_PPV_ARGS(&md3dDevice));
				if (SUCCEEDED(hr) && suitableIndex++ == index) {
					adapterFound = true;
					std::wstring description{desc.Description};
					std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
					LUISA_VERBOSE(
						"Create DirectX device #{}: {}.",
						index, converter.to_bytes(description));
					break;
				}
			}
			adapter->Release();
			adapterIndex++;
		}
		// Check 4X MSAA quality support for our back buffer format.
		// All Direct3D 11 capable devices support 4X MSAA for all render
		// target formats, so we only need to check quality support.
		if (!adapterFound) {
			LUISA_ERROR_WITH_LOCATION(
				"Failed to create DirectX device with index {}.", index);
		}
		ThrowIfFailed(md3dDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&cpuFence)));
		CreateCommandQueue();
	}
	void CreateCommandQueue() {
		ID3D12Device* device = md3dDevice.Get();
		D3D12_COMMAND_QUEUE_DESC queueDesc = {};
		queueDesc.Type = (D3D12_COMMAND_LIST_TYPE)GFXCommandListType_Compute;
		queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_DISABLE_GPU_TIMEOUT;
		ThrowIfFailed(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&mComputeCommandQueue)));
		queueDesc.Type = (D3D12_COMMAND_LIST_TYPE)GFXCommandListType_Copy;
		ThrowIfFailed(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&mCopyCommandQueue)));
	}
	ComPtr<IDXGIFactory4> mdxgiFactory;
	ComPtr<ID3D12Device> md3dDevice;
	IDXGIAdapter1* adapter{nullptr};
	ComPtr<ID3D12Fence> cpuFence;
	uint64 signalCount = 1;

	ComPtr<GFXCommandQueue> mComputeCommandQueue;
	ComPtr<GFXCommandQueue> mCopyCommandQueue;
};
}// namespace luisa::compute

LUISA_EXPORT luisa::compute::Device* create(const luisa::compute::Context& ctx, uint32_t id) noexcept {
	//TODO: device not finished;
	return new luisa::compute::DXDevice{ctx, id};
}

LUISA_EXPORT void destroy(luisa::compute::Device* device) noexcept {
	delete device;
}
