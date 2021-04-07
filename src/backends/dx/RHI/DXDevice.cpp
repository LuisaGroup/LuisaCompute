#include <codecvt>
#include <Common/GFXUtil.h>
#include <runtime/device.h>
#include <RenderComponent/RenderComponentInclude.h>
#include <PipelineComponent/ThreadCommand.h>
#include <PipelineComponent/CommandAllocator.h>
namespace luisa::compute {

static GFXFormat LCFormatToVEngineFormat(PixelFormat format) {
	switch (format) {
		//TODO:
		default:
			return GFXFormat_R8G8B8A8_SNorm;
	}
}

class DXDevice final : public Device {

private:
	///////////// D3D
	Microsoft::WRL::ComPtr<IDXGIFactory4> mdxgiFactory;
	Microsoft::WRL::ComPtr<ID3D12Device> md3dDevice;
	IDXGIAdapter1* adapter{nullptr};
	void InitD3D(uint32_t index) {
#if defined(DEBUG) || defined(_DEBUG)
		// Enable the D3D12 debug layer.
		{
			Microsoft::WRL::ComPtr<ID3D12Debug> debugController;
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
	}
	//////////// GFX
	GFXDevice* dxDevice{nullptr};

public:
	DXDevice(const Context& ctx, uint32_t index) : Device(ctx) {// TODO: support device selection?
		InitD3D(index);
		dxDevice = md3dDevice.Get();
	}
	uint64 create_buffer(size_t size_bytes) noexcept {
		return reinterpret_cast<uint64>(
			new StructuredBuffer(
				dxDevice,
				{StructuredBufferElement::Get(1, size_bytes)},
				GPUResourceState_Common,
				nullptr//TODO: allocator
				));
	}
	void dispose_buffer(uint64 handle) noexcept {
		delete reinterpret_cast<StructuredBuffer*>(handle);
	}

	// texture
	uint64 create_texture(
		PixelFormat format, uint dimension, uint width, uint height, uint depth,
		uint mipmap_levels, bool is_bindless) {
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
	void dispose_texture(uint64 handle) noexcept {
		delete reinterpret_cast<RenderTexture*>(handle);
	}

	// stream
	uint64 create_stream() noexcept {
		return 0u;
		//return reinterpret_cast<uint64>(new ThreadCommand)
	}
	void dispose_stream(uint64 handle) noexcept {}
	void synchronize_stream(uint64 stream_handle) noexcept {}
	void dispatch(uint64 stream_handle, CommandBuffer cmd_buffer, std::function<void()> callback) noexcept {}

	// kernel
	void prepare_kernel(uint32_t uid) noexcept {
		// do async compile here...
	}
	~DXDevice() {
	}
};
}// namespace luisa::compute

LUISA_EXPORT luisa::compute::Device* create(const luisa::compute::Context& ctx, uint32_t id) noexcept {
	return new luisa::compute::DXDevice{ctx, id};
}

LUISA_EXPORT void destroy(luisa::compute::Device* device) noexcept {
	delete device;
}
