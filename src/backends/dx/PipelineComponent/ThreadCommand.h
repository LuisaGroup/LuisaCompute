#pragma once
#include <Common/GFXUtil.h>
#include <mutex>
#include <atomic>
#include <PipelineComponent/CommandAllocator.h>
#include <Common/LockFreeArrayQueue.h>
#include <Singleton/Graphics.h>
#include <RenderComponent/PSOContainer.h>
class PipelineComponent;
class StructuredBuffer;
class RenderTexture;
class IShader;
class DescriptorHeap;
class DescriptorHeapRoot;
class VENGINE_DLL_RENDERER ThreadCommand final {
	friend class PipelineComponent;
	friend class Graphics;

private:
	struct RenderTargetRegist {
		uint64 rtInstanceID;
		uint mipLevel;
		uint slice;
	};
	IShader const* bindedShader;
	Type bindedShaderType;
	//Datas
	uint64 shaderRootInstanceID;
	uint64 descHeapInstanceID;
	DescriptorHeap const* descHeap;
	void const* pso;
	vengine::vector<D3D12_CPU_DESCRIPTOR_HANDLE> colorHandles;
	D3D12_CPU_DESCRIPTOR_HANDLE depthHandle;
	PSOContainer psoContainer;
	ObjectPtr<CommandAllocator> cmdAllocator;
	Microsoft::WRL::ComPtr<GFXCommandList> cmdList;
	uint64 frameCount;
	bool managingAllocator;
	bool containedResources = false;
	GFXCommandListType commandListType;
	vengine::vector<D3D12_RESOURCE_BARRIER> resourceBarrierCommands;
	struct ResourceBarrierCommand {
		GPUResourceBase const* resource;
		GPUResourceState targetState;
		int32_t index;
		ResourceBarrierCommand() {
		}
		ResourceBarrierCommand(
			GPUResourceBase const* resource,
			GPUResourceState targetState,
			int32_t index)
			: resource(resource),
			  targetState(targetState),
			  index(index) {}
	};
	HashMap<uint64, ResourceBarrierCommand> barrierRecorder;
	HashMap<GFXResource*, bool> uavBarriersDict;
	HashMap<std::pair<GFXResource*, GFXResource*>, bool> aliasBarriersDict;
	HashMap<GPUResourceBase const*, GPUResourceState> backToInitState;
	vengine::vector<GFXResource*> uavBarriers;
	vengine::vector<std::pair<GFXResource*, GFXResource*>> aliasBarriers;
	void KillSame();
	void Clear();

public:
	DescriptorHeap const* GetBindedHeap() const {return descHeap;}
	IShader const* GetBindedShader() const { return bindedShader; }
	Type GetBindedShaderType() const { return bindedShaderType; }
	inline GFXCommandAllocator* GetAllocator() const { return cmdAllocator->GetAllocator().Get(); }
	inline GFXCommandList* GetCmdList() const { return cmdList.Get(); }
	ThreadCommand(GFXDevice* device, GFXCommandListType type, ObjectPtr<CommandAllocator> const& allocator = nullptr);
	~ThreadCommand();
	void ResetCommand();
	void CloseCommand();
	bool UpdateRegisterShader(IShader const* shader);
	bool UpdateDescriptorHeap(DescriptorHeap const* descHeap, DescriptorHeapRoot const* descHeapRoot);
	bool UpdatePSO(void* psoObj);
	bool UpdateRenderTarget(
		uint NumRenderTargetDescriptors,
		const D3D12_CPU_DESCRIPTOR_HANDLE* pRenderTargetDescriptors,
		const D3D12_CPU_DESCRIPTOR_HANDLE* pDepthStencilDescriptor);
	void RegistInitState(GPUResourceState initState, GPUResourceBase const* resource, bool backToInitAfterRender = false);
	void UpdateResState(GPUResourceState newState, GPUResourceBase const* resource);
	void UpdateResState(GPUResourceState beforeState, GPUResourceState afterState, GPUResourceBase const* resource);
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