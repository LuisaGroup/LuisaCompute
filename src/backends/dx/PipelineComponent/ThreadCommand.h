#pragma once
#include <Common/GFXUtil.h>
#include <Common/HashPicker.h>
#include <mutex>
#include <atomic>
#include <PipelineComponent/CommandAllocator.h>
#include <util/LockFreeArrayQueue.h>
#include <Singleton/Graphics.h>
class PipelineComponent;
class StructuredBuffer;
class RenderTexture;
class IShader;
class DescriptorHeap;
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
	vstd::vector<D3D12_CPU_DESCRIPTOR_HANDLE> colorHandles;
	D3D12_CPU_DESCRIPTOR_HANDLE depthHandle;
	ObjectPtr<CommandAllocator> cmdAllocator;
	Microsoft::WRL::ComPtr<GFXCommandList> cmdList;
	uint64 frameCount;
	bool managingAllocator;
	bool containedResources = false;
	GFXCommandListType commandListType;
	vstd::vector<D3D12_RESOURCE_BARRIER> resourceBarrierCommands;
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
	HashPicker<GFXResource*> uavBarriersDict;
	HashMap<std::pair<GFXResource*, GFXResource*>, bool> aliasBarriersDict;
	HashMap<GPUResourceBase const*, GPUResourceState> backToInitState;
	vstd::vector<std::pair<GFXResource*, GFXResource*>> aliasBarriers;
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
	bool UpdateDescriptorHeap(DescriptorHeap const* descHeap);
	bool UpdatePSO(void* psoObj);
	bool UpdateRenderTarget(
		uint NumRenderTargetDescriptors,
		const D3D12_CPU_DESCRIPTOR_HANDLE* pRenderTargetDescriptors,
		const D3D12_CPU_DESCRIPTOR_HANDLE* pDepthStencilDescriptor);
	void RegistInitState(GPUResourceState initState, GPUResourceBase const* resource, bool backToInitAfterRender = false);
	void UpdateResState(GPUResourceState newState, GPUResourceBase const* resource);
	void UpdateResState(GPUResourceState beforeState, GPUResourceState afterState, GPUResourceBase const* resource, bool backToInitAfterRender = false);
	void ExecuteResBarrier();
	void UAVBarrier(GPUResourceBase const*);
	void UAVBarriers(const std::initializer_list<GPUResourceBase const*>&);
	void AliasBarrier(GPUResourceBase const* before, GPUResourceBase const* after);
	void AliasBarriers(std::initializer_list<std::pair<GPUResourceBase const*, GPUResourceBase const*>> const&);
	
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
