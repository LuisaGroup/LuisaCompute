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
		//TODO:
		default:
			return GFXFormat_R8G8B8A8_SNorm;
	}
}
class FrameResource;
class DXDevice final : public Device::Interface {
public:
	DXDevice(const Context& ctx, uint32_t index)
		: Device::Interface(ctx) {// TODO: support device selection?
		InitD3D(index);
		dxDevice.New(md3dDevice.Get());
		SCompile::HLSLCompiler::InitRegisterData(ctx);
		graphicsInstance.New(dxDevice);
		shaderGlobal = ShaderLoader::Init(dxDevice);
		cbAlloc.New(dxDevice, false);
		ShaderID::Init();
	}
	uint64 create_buffer(size_t size_bytes) noexcept override {
		Graphics::current = graphicsInstance;
		return reinterpret_cast<uint64>(
			new StructuredBuffer(
				dxDevice,
				{StructuredBufferElement::Get(1, size_bytes)},
				GPUResourceState_Common,
				DXAllocator::GetBufferAllocator()//TODO: allocator
				));
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
			DXAllocator::GetTextureAllocator(),//TODO: allocator
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
		evt->Sync([&](uint64 signal) {
			DXStream::WaitFence(
				cpuFence.Get(),
				signal);
		});
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
	InternalShaders internalShaders;
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
		auto suitableIndex = 0u;
		int32_t adapterIndex = 0; // we'll start looking for directx 12  compatible graphics devices starting at index 0
		bool adapterFound = false;// set this to true when a good one was found
		while (mdxgiFactory->EnumAdapters1(adapterIndex, &adapter) != DXGI_ERROR_NOT_FOUND) {
			DXGI_ADAPTER_DESC1 desc;
			adapter->GetDesc1(&desc);
			if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0) {
				HRESULT hr = D3D12CreateDevice(
					adapter, D3D_FEATURE_LEVEL_12_1,
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
	void EnableThreadLocal() {
		Graphics::current = graphicsInstance;
		ShaderLoader::current = shaderGlobal;
	}
	void InitInternal() {
		EnableThreadLocal();
		internalShaders.copyShader = ShaderLoader::GetComputeShader(
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
