#include <codecvt>
#include <Common/GFXUtil.h>
#include <Common/LockFreeArrayQueue.h>
#include <runtime/device.h>
#include <RenderComponent/RenderComponentInclude.h>
#include <RHI/DXStream.hpp>
#include <RHI/ShaderCompiler.h>
#include <RHI/DXEvent.h>
#include <RHI/InternalShaders.h>
#include <RHI/RenderTexturePackage.h>
#include <ShaderCompile/HLSLCompiler.h>
#include <PipelineComponent/DXAllocator.h>
#include <Singleton/Graphics.h>
#include <Singleton/ShaderLoader.h>
#include <RenderComponent/ComputeShader.h>

#include <Singleton/ShaderID.h>
#include <RenderComponent/CBufferAllocator.h>

namespace luisa::compute {
using namespace Microsoft::WRL;

static GFXFormat LCFormatToVEngineFormat(PixelFormat format) {
	switch (format) {
		case PixelFormat::R8SInt:
			return GFXFormat_R8_SInt;
		case PixelFormat::R8UInt:
			return GFXFormat_R8_UInt;
		case PixelFormat::R8UNorm:
			return GFXFormat_R8_UNorm;
		case PixelFormat::RG8SInt:
			return GFXFormat_R8G8_SInt;
		case PixelFormat::RG8UInt:
			return GFXFormat_R8G8_UInt;
		case PixelFormat::RG8UNorm:
			return GFXFormat_R8G8B8A8_UNorm;
		case PixelFormat::RGBA8SInt:
			return GFXFormat_R8G8B8A8_SInt;
		case PixelFormat::RGBA8UInt:
			return GFXFormat_R8G8B8A8_UInt;
		case PixelFormat::RGBA8UNorm:
			return GFXFormat_R8G8B8A8_UNorm;

		case PixelFormat::R16SInt:
			return GFXFormat_R16_SInt;
		case PixelFormat::R16UInt:
			return GFXFormat_R16_UInt;
		case PixelFormat::R16UNorm:
			return GFXFormat_R16_UNorm;
		case PixelFormat::RG16SInt:
			return GFXFormat_R16G16_SInt;
		case PixelFormat::RG16UInt:
			return GFXFormat_R16G16_UInt;
		case PixelFormat::RG16UNorm:
			return GFXFormat_R16G16B16A16_UNorm;
		case PixelFormat::RGBA16SInt:
			return GFXFormat_R16G16B16A16_SInt;
		case PixelFormat::RGBA16UInt:
			return GFXFormat_R16G16B16A16_UInt;
		case PixelFormat::RGBA16UNorm:
			return GFXFormat_R16G16B16A16_UNorm;

		case PixelFormat::R32SInt:
			return GFXFormat_R32_SInt;
		case PixelFormat::R32UInt:
			return GFXFormat_R32_UInt;
		case PixelFormat::RG32SInt:
			return GFXFormat_R32G32_SInt;
		case PixelFormat::RG32UInt:
			return GFXFormat_R32G32_UInt;
		case PixelFormat::RGBA32SInt:
			return GFXFormat_R32G32B32A32_SInt;
		case PixelFormat::RGBA32UInt:
			return GFXFormat_R32G32B32A32_UInt;

		case PixelFormat::R16F:
			return GFXFormat_R16_Float;
		case PixelFormat::RG16F:
			return GFXFormat_R16G16_Float;
		case PixelFormat::RGBA16F:
			return GFXFormat_R16G16B16A16_Float;

		case PixelFormat::R32F:
			return GFXFormat_R32_Float;
		case PixelFormat::RG32F:
			return GFXFormat_R32G32_Float;
		case PixelFormat::RGBA32F:
			return GFXFormat_R32G32B32A32_Float;
	}
}
class FrameResource;
class DXDevice final : public Device::Interface  {
public:
	DXDevice(const Context& ctx, uint32_t index) : Device::Interface(ctx) {// TODO: support device selection?
		EnableThreadLocal();
		InitD3D(index);
		dxDevice.New(md3dDevice.Get());

		SCompile::HLSLCompiler::InitRegisterData();

		graphicsInstance.New(dxDevice);
		shaderGlobal = ShaderLoader::Init(dxDevice);
		cbAlloc.New(dxDevice, false);
		ShaderID::Init();

		internalShaders.New();

	}
	uint64 create_buffer(size_t size_bytes) noexcept override {
		Graphics::current = graphicsInstance;
		return reinterpret_cast<uint64>(
			new StructuredBuffer(
				dxDevice,
				{StructuredBufferElement::Get(1, size_bytes)},
				GPUResourceState_Common,
				DXAllocator::GetBufferAllocator()));
	}
	void dispose_buffer(uint64 handle) noexcept override {
		delete reinterpret_cast<StructuredBuffer*>(handle);
	}

	// texture
	uint64 create_texture(
		PixelFormat format, uint dimension, uint width, uint height, uint depth,
		uint mipmap_levels, bool is_bindless) override {
		Graphics::current = graphicsInstance;
		RenderTexturePackage* pack = new RenderTexturePackage();
		pack->bindless = is_bindless;
		pack->format = format;
		pack->rt.New(
			dxDevice,
			DXAllocator::GetTextureAllocator(),
			width,
			height,
			RenderTextureFormat::GetColorFormat(LCFormatToVEngineFormat(format)),
			dimension > 2 ? TextureDimension::Tex3D : TextureDimension::Tex2D,
			depth,
			mipmap_levels,
			RenderTextureState::Common,
			0);
		return reinterpret_cast<uint64>(pack);
	}
	void dispose_texture(uint64 handle) noexcept override {
		delete reinterpret_cast<RenderTexture*>(handle);
	}

	// stream
	uint64 create_stream() noexcept override {
		return reinterpret_cast<uint64>(
			new DXStream(dxDevice, mComputeCommandQueue.Get(), GFXCommandListType_Compute)//TODO: need support copy
		);
	}
	void dispose_stream(uint64 handle) noexcept override {
		synchronize_stream(handle);
		delete reinterpret_cast<DXStream*>(handle);
	}
	void synchronize_stream(uint64 stream_handle) noexcept override {
		DXStream* stream = reinterpret_cast<DXStream*>(stream_handle);
		stream->Sync(cpuFence.Get(), mtx);
		FreeFrameResource(stream->GetSignal());
	}
	void dispatch(uint64 stream_handle, CommandBuffer cmd_buffer) noexcept override {
		EnableThreadLocal();
		DXStream* stream = reinterpret_cast<DXStream*>(stream_handle);
		stream->Execute(
			dxDevice,
			std::move(cmd_buffer),
			cpuFence.Get(),
			[&](GFXCommandListType type) {
				return GetFrameResource(type);
			},
			internalShaders,
			usingQueue[GetQueueIndex(stream->GetType())],
			mtx,
			signalCount);
	}
	// kernel
	void compile_kernel(uint32_t uid) noexcept override {
		ShaderCompiler::TryCompileCompute(uid);
	}
	uint64 create_event() noexcept override {
		return reinterpret_cast<uint64>(new DXEvent());
	}
	void dispose_event(uint64 handle) noexcept override {
		delete reinterpret_cast<DXEvent*>(handle);
	}
	void signal_event(uint64 handle, uint64 stream_handle) noexcept override {
		DXStream* stream = reinterpret_cast<DXStream*>(stream_handle);
		DXEvent* evt = reinterpret_cast<DXEvent*>(handle);
		evt->AddSignal(
			reinterpret_cast<uint64>(stream->GetQueue()),
			stream->GetSignal());
	}
	void wait_event(uint64 handle, uint64 stream_handle) noexcept override {
		DXEvent* evt = reinterpret_cast<DXEvent*>(handle);
		DXStream* stream = reinterpret_cast<DXStream*>(stream_handle);
		std::lock_guard lck(mtx);
		evt->GPUWaitEvent(
			stream->GetQueue(),
			cpuFence.Get());
	}
	/*
	uint64 signal_event(uint64 handle, uint64 stream_handle);
	void wait_event(uint64 signal, uint64 stream_handle)
	*/
	void synchronize_event(uint64 handle) noexcept override {
		DXEvent* evt = reinterpret_cast<DXEvent*>(handle);
		evt->Sync(std::move(Runnable<void(uint64)>([&](uint64 signal) {
			DXStream::WaitFence(
				cpuFence.Get(),
				signal);
		})));
	}

	~DXDevice() {
		ShaderLoader::Dispose(shaderGlobal);
	}
	//////////// Variables
	StackObject<GFXDevice, true> dxDevice;
	DECLARE_VENGINE_OVERRIDE_OPERATOR_NEW
private:
	///////////// D3D
	ComPtr<IDXGIFactory4> mdxgiFactory;
	ComPtr<ID3D12Device> md3dDevice;
	IDXGIAdapter1* adapter{nullptr};
	ComPtr<ID3D12Fence> cpuFence;
	uint64 signalCount = 1;

	ComPtr<GFXCommandQueue> mComputeCommandQueue;
	ComPtr<GFXCommandQueue> mCopyCommandQueue;
	static constexpr uint QUEUE_COUNT = 1;
	LockFreeArrayQueue<FrameResource*> waitingRes[QUEUE_COUNT];
	SingleThreadArrayQueue<FrameResource*> usingQueue[QUEUE_COUNT];
	std::mutex mtx;
	StackObject<InternalShaders> internalShaders;
	StackObject<Graphics, true> graphicsInstance;
	ShaderLoaderGlobal* shaderGlobal;
	ComputeShader* copyShader;
	StackObject<CBufferAllocator, true> cbAlloc;
	HashMap<uint, IShader*> loadShaders;
	void InitD3D(uint32_t index) {
#if defined(DEBUG)
		// Enable the D3D12 debug layer.
		{
			ComPtr<ID3D12Debug> debugController;
			ThrowIfFailed(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)));
			debugController->EnableDebugLayer();
		}
#endif
		ThrowIfFailed(CreateDXGIFactory1(IID_PPV_ARGS(&mdxgiFactory)));
		int32_t adapterIndex = 0; // we'll start looking for directx 12  compatible graphics devices starting at index 0
		bool adapterFound = false;// set this to true when a good one was found
		while (mdxgiFactory->EnumAdapters1(adapterIndex, &adapter) != DXGI_ERROR_NOT_FOUND) {
			DXGI_ADAPTER_DESC1 desc;
			adapter->GetDesc1(&desc);
			if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0) {
				HRESULT hr = D3D12CreateDevice(
					adapter, D3D_FEATURE_LEVEL_12_1,
					IID_PPV_ARGS(&md3dDevice));
				if (SUCCEEDED(hr)) {
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
		if (!adapterFound) [[unlikely]] {
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
	void EnableThreadLocal() {
		Graphics::current = graphicsInstance;
		ShaderLoader::current = shaderGlobal;
	}
	void InitInternal() {
		EnableThreadLocal();
		internalShaders->copyShader = ShaderLoader::GetComputeShader(
			"DXCompiledShader/Internal/Copy.compute.cso"_sv);
	}

	void FreeFrameResource(uint64 lastSignal = 0) {
		std::lock_guard lck(mtx);
		if (lastSignal == 0)
			lastSignal = cpuFence->GetCompletedValue();
		for (uint index = 0; index < QUEUE_COUNT; ++index) {
			for (uint i = 0; i < QUEUE_COUNT; ++i) {
				auto& queue = usingQueue[index];
				auto& waitingQueue = waitingRes[index];
				FrameResource* res;
				while (queue.GetLast(&res)) {
					if (lastSignal >= res->signalIndex) {
						queue.Pop();
						res->ReleaseTemp();
						waitingQueue.Push(res);
					}
				}
			}
		}
	}

	FrameResource* GetFrameResource(GFXCommandListType type) {
		FreeFrameResource();
		FrameResource* result = nullptr;
		size_t index = GetQueueIndex(type);
		if (waitingRes[index].Pop(&result))
			return result;
		return new FrameResource(dxDevice, type, cbAlloc);
	}
	size_t GetQueueIndex(GFXCommandListType lstType) {
		switch (lstType) {
			case GFXCommandListType_Compute:
				return 0;
		}
		//TODO
		return 0;
	}
};
}// namespace luisa::compute

LUISA_EXPORT luisa::compute::Device::Interface* create(const luisa::compute::Context& ctx, uint32_t id) noexcept {
	return new luisa::compute::DXDevice{ctx, id};
}

LUISA_EXPORT void destroy(luisa::compute::Device::Interface* device) noexcept {
	delete device;
}
/*
void CreateDevice_Test() {
	using namespace luisa::compute;
	auto dev = new DXDevice(0);
	auto stream = dev->create_stream();
	dev->synchronize_stream(stream);
	dev->dispose_stream(stream);
	auto buffer = dev->create_buffer(
		64);
	dev->dispose_buffer(buffer);
	auto tex = dev->create_texture(
		PixelFormat::RGBA32F,
		2,
		1024,
		1024,
		1,
		0,
		true
	);
	dev->dispose_texture(
		tex
	);
	delete dev;
	std::cout << "Finish\n";
}
*/
