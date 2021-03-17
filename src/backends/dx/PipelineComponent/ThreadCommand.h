#pragma once
#include "../Common/GFXUtil.h"
#include <mutex>
#include <atomic>
#include "CommandAllocator.h"
#include "../Common/LockFreeArrayQueue.h"
#include "../Singleton/Graphics.h"
#include "../RenderComponent/PSOContainer.h"
struct StateTransformBuffer {
	GFXResource* targetResource;
	GFXResourceState beforeState;
	GFXResourceState afterState;
};
class PipelineComponent;
class StructuredBuffer;
class RenderTexture;
class IShader;
class DescriptorHeapRoot;
class ThreadCommand final {
	friend class PipelineComponent;
	friend class Graphics;

private:
	struct RenderTargetRegist {
		uint64 rtInstanceID;
		uint mipLevel;
		uint slice;
	};
	//Datas
	uint64 shaderRootInstanceID;
	uint64 descHeapInstanceID;
	void const* pso;
	vengine::vector<D3D12_CPU_DESCRIPTOR_HANDLE> colorHandles;
	D3D12_CPU_DESCRIPTOR_HANDLE depthHandle;
	PSOContainer psoContainer;
	ObjectPtr<CommandAllocator> cmdAllocator;
	Microsoft::WRL::ComPtr<GFXCommandList> cmdList;
	HashMap<RenderTexture*, ResourceReadWriteState> rtStateMap;
	HashMap<StructuredBuffer*, ResourceReadWriteState> sbufferStateMap;
	uint64 frameCount;
	bool managingAllocator;
	bool containedResources = false;
	vengine::vector<D3D12_RESOURCE_BARRIER> resourceBarrierCommands;
	struct ResourceBarrierCommand {
		GPUResourceBase const* resource;
		GFXResourceState targetState;
		int32_t index;
		ResourceBarrierCommand() {
		}
		ResourceBarrierCommand(
			GPUResourceBase const* resource,
			GFXResourceState targetState,
			int32_t index)
			: resource(resource),
			  targetState(targetState),
			  index(index) {}
	};
	HashMap<uint64, ResourceBarrierCommand> barrierRecorder;
	HashMap<GFXResource*, bool> uavBarriersDict;
	HashMap<std::pair<GFXResource*, GFXResource*>, bool> aliasBarriersDict;
	vengine::vector<GFXResource*> uavBarriers;
	vengine::vector<std::pair<GFXResource*, GFXResource*>> aliasBarriers;
	void KillSame();
	void Clear();
	bool UpdateResStateLocal(RenderTexture* rt, ResourceReadWriteState state);
	bool UpdateResStateLocal(StructuredBuffer* rt, ResourceReadWriteState state);

public:
	inline GFXCommandAllocator* GetAllocator() const { return cmdAllocator->GetAllocator().Get(); }
	inline GFXCommandList* GetCmdList() const { return cmdList.Get(); }
	ThreadCommand(GFXDevice* device, GFXCommandListType type, ObjectPtr<CommandAllocator> const& allocator = nullptr);
	~ThreadCommand();
	void SetResourceReadWriteState(RenderTexture* rt, ResourceReadWriteState state);
	void SetResourceReadWriteState(StructuredBuffer* rt, ResourceReadWriteState state);
	void ResetCommand();
	void CloseCommand();
	bool UpdateRegisterShader(IShader const* shader);
	bool UpdateDescriptorHeap(DescriptorHeapRoot const* descHeap);
	bool UpdatePSO(void* psoObj);
	bool UpdateRenderTarget(
		uint NumRenderTargetDescriptors,
		const D3D12_CPU_DESCRIPTOR_HANDLE* pRenderTargetDescriptors,
		const D3D12_CPU_DESCRIPTOR_HANDLE* pDepthStencilDescriptor);
	void RegistInitState(GFXResourceState initState, GPUResourceBase const* resource);
	void UpdateResState(GFXResourceState newState, GPUResourceBase const* resource);
	void UpdateResState(GFXResourceState beforeState, GFXResourceState afterState, GPUResourceBase const* resource);
	void ExecuteResBarrier();
	void UAVBarrier(GPUResourceBase const*);
	void UAVBarriers(const std::initializer_list<GPUResourceBase const*>&);
	void AliasBarrier(GPUResourceBase const* before, GPUResourceBase const* after);
	void AliasBarriers(std::initializer_list<std::pair<GPUResourceBase const*, GPUResourceBase const*>> const&);
	void SetRenderTarget(
		RenderTexture const* const* renderTargets,
		uint rtCount,
		RenderTexture const* depthTex = nullptr) {
		psoContainer.SetRenderTarget(
			this,
			renderTargets,
			rtCount,
			depthTex);
	}
	void SetRenderTarget(
		const std::initializer_list<RenderTexture const*>& renderTargets,
		RenderTexture const* depthTex = nullptr) {
		psoContainer.SetRenderTarget(
			this,
			renderTargets,
			depthTex);
	}
	void SetRenderTarget(
		const RenderTarget* renderTargets,
		uint rtCount,
		const RenderTarget& depth) {
		psoContainer.SetRenderTarget(
			this,
			renderTargets,
			rtCount,
			depth);
	}
	void SetRenderTarget(
		const std::initializer_list<RenderTarget>& init,
		const RenderTarget& depth) {
		psoContainer.SetRenderTarget(
			this,
			init,
			depth);
	}
	void SetRenderTarget(
		const RenderTarget* renderTargets,
		uint rtCount) {
		psoContainer.SetRenderTarget(
			this,
			renderTargets,
			rtCount);
	}
	void SetRenderTarget(
		const std::initializer_list<RenderTarget>& init) {
		psoContainer.SetRenderTarget(
			this,
			init);
	}

	GFXPipelineState* GetPSOState(PSODescriptor const& desc, GFXDevice* device) {
		return psoContainer.GetPSOState(desc, device);
	}
	GFXPipelineState* TryGetPSOStateAsync(
		PSODescriptor const& desc, GFXDevice* device) {
		return psoContainer.TryGetPSOStateAsync(
			desc, device);
	}
	KILL_COPY_CONSTRUCT(ThreadCommand)
	DECLARE_VENGINE_OVERRIDE_OPERATOR_NEW
};

class ThreadCommandFollower {
private:
	ThreadCommand* cmd;

public:
	ThreadCommandFollower(ThreadCommand* cmd) : cmd(cmd) {
		cmd->ResetCommand();
	}
	ThreadCommandFollower(const ThreadCommandFollower&) = delete;
	ThreadCommandFollower(ThreadCommandFollower&&) = delete;
	void operator=(const ThreadCommandFollower&) = delete;
	void operator=(ThreadCommandFollower&&) = delete;
	~ThreadCommandFollower() {
		cmd->CloseCommand();
	}
};