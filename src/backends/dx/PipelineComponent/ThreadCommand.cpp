#include <PipelineComponent/ThreadCommand.h>
#include <PipelineComponent/ThreadCommand.h>
//#endif
#include <PipelineComponent/ThreadCommand.h>
#include <RenderComponent/RenderTexture.h>
#include <RenderComponent/StructuredBuffer.h>
#include <Singleton/Graphics.h>
#include <RenderComponent/IShader.h>
void ThreadCommand::ResetCommand() {
	shaderRootInstanceID = 0;
	descHeapInstanceID = 0;
	pso = nullptr;
	colorHandles.clear();
	depthHandle.ptr = 0;
	auto alloc = cmdAllocator->GetAllocator().Get();
	if (managingAllocator) {
		ThrowIfFailed(alloc->Reset());
	} else {
		cmdAllocator->Reset(frameCount);
	}
	ThrowIfFailed(cmdList->Reset(alloc, nullptr));
}
void ThreadCommand::CloseCommand() {
	Clear();
	cmdList->Close();
}
bool ThreadCommand::UpdateRegisterShader(IShader const* shader) {
	uint64 id = shader->GetInstanceID();
	if (id == shaderRootInstanceID)
		return false;
	shaderRootInstanceID = id;
	bindedShader = shader;
	bindedShaderType = typeid(*shader);
	return true;
}
bool ThreadCommand::UpdateDescriptorHeap(DescriptorHeap const* descHeap) {
	this->descHeap = descHeap;
	uint64 id = descHeap->GetInstanceID();
	if (id == descHeapInstanceID)
		return false;
	descHeapInstanceID = id;
	ID3D12DescriptorHeap* heap = descHeap->pDH.Get();
	cmdList->SetDescriptorHeaps(1, &heap);
	return true;
}
bool ThreadCommand::UpdatePSO(void* psoObj) {
	if (pso == psoObj)
		return false;
	pso = psoObj;
	cmdList->SetPipelineState(static_cast<ID3D12PipelineState*>(psoObj));
	return true;
}
bool ThreadCommand::UpdateRenderTarget(
	uint NumRenderTargetDescriptors,
	const D3D12_CPU_DESCRIPTOR_HANDLE* pRenderTargetDescriptors,
	const D3D12_CPU_DESCRIPTOR_HANDLE* pDepthStencilDescriptor) {
	D3D12_CPU_DESCRIPTOR_HANDLE curDepthHandle = {0};
	auto originDepth = pDepthStencilDescriptor;
	if (!pDepthStencilDescriptor) {
		pDepthStencilDescriptor = &curDepthHandle;
	}
	auto Disposer = [&]() -> void {
		colorHandles.clear();
		colorHandles.resize(NumRenderTargetDescriptors);
		memcpy(colorHandles.data(), pRenderTargetDescriptors, sizeof(D3D12_CPU_DESCRIPTOR_HANDLE) * NumRenderTargetDescriptors);
		depthHandle = *pDepthStencilDescriptor;
		cmdList->OMSetRenderTargets(
			NumRenderTargetDescriptors,
			pRenderTargetDescriptors,
			false,
			originDepth);
	};
	if (colorHandles.size() != NumRenderTargetDescriptors) {
		Disposer();
		return true;
	}
	if (pDepthStencilDescriptor->ptr != depthHandle.ptr) {
		Disposer();
		return true;
	}
	for (uint i = 0; i < NumRenderTargetDescriptors; ++i) {
		if (colorHandles[i].ptr != pRenderTargetDescriptors[i].ptr) {
			Disposer();
			return true;
		}
	}

	return false;
}

void ThreadCommand::RegistInitState(GPUResourceState initState, GPUResourceBase const* resource, bool toInit) {
	auto ite = barrierRecorder.Find(resource->GetInstanceID());
	if (!ite) {
		barrierRecorder.Insert(resource->GetInstanceID(), ResourceBarrierCommand(resource, initState, -1));
	}
	if (toInit) {
		backToInitState.Insert(resource, initState);
	}
}

void ThreadCommand::UpdateResState(GPUResourceState newState, GPUResourceBase const* resource) {
	containedResources = true;
	auto ite = barrierRecorder.Find(resource->GetInstanceID());
	if (!ite) {
		return;
	}
	auto&& cmd = ite.Value();
	if (cmd.index < 0) {
		if (cmd.targetState != newState) {
			cmd.index = resourceBarrierCommands.size();
			resourceBarrierCommands.push_back(CD3DX12_RESOURCE_BARRIER::Transition(
				resource->GetResource(),
				resource->GetGFXResourceState(cmd.targetState),
				resource->GetGFXResourceState(newState)));
			cmd.targetState = newState;
		}
	} else {
		if (resourceBarrierCommands.empty()) return;
		resourceBarrierCommands[cmd.index].Transition.StateAfter = resource->GetGFXResourceState(newState);
		cmd.targetState = newState;
	}
}
void ThreadCommand::UAVBarrier(GPUResourceBase const* res) {
	containedResources = true;
	auto ite = uavBarriersDict.Find(res->GetResource());
	if (!ite) {
		uavBarriersDict.Insert(res->GetResource());
	}
}
void ThreadCommand::UAVBarriers(const std::initializer_list<GPUResourceBase const*>& args) {
	containedResources = true;
	for (auto i = args.begin(); i != args.end(); ++i) {
		UAVBarrier(*i);
	}
}
void ThreadCommand::AliasBarrier(GPUResourceBase const* before, GPUResourceBase const* after) {
	containedResources = true;
	auto ite = aliasBarriersDict.Find({before->GetResource(), after->GetResource()});
	if (!ite) {
		aliasBarriersDict.Insert({before->GetResource(), after->GetResource()}, true);
		aliasBarriers.push_back({before->GetResource(), after->GetResource()});
	}
}
void ThreadCommand::AliasBarriers(std::initializer_list<std::pair<GPUResourceBase const*, GPUResourceBase const*>> const& lst) {
	containedResources = true;
	for (auto& i : lst) {
		AliasBarrier(i.first, i.second);
	}
}

void ThreadCommand::KillSame() {
	bool isCopyQueue = (commandListType == GFXCommandListType_Copy);
	for (size_t i = 0; i < resourceBarrierCommands.size(); ++i) {
		auto& a = resourceBarrierCommands[i];
		bool toCommon = a.Transition.StateAfter == D3D12_RESOURCE_STATE_COMMON;
		if (a.Transition.StateBefore == a.Transition.StateAfter
			|| (toCommon && (isCopyQueue || a.Transition.pResource->GetDesc().Dimension == D3D12_RESOURCE_DIMENSION_BUFFER))) {
			auto last = resourceBarrierCommands.end() - 1;
			if (i != (resourceBarrierCommands.size() - 1)) {
				a = *last;
			}
			resourceBarrierCommands.erase(last);
			i--;
		} else if (!toCommon) {
			auto uavIte = uavBarriersDict.Find(a.Transition.pResource);
			if (uavIte) {
				uavBarriersDict.Remove(uavIte);
			}
		}
	}

	{
		D3D12_RESOURCE_BARRIER uavBar;
		uavBar.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
		uavBar.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
		for (auto ite = uavBarriersDict.begin(); ite != uavBarriersDict.end(); ++ite) {
			uavBar.UAV.pResource = *ite;
			resourceBarrierCommands.push_back(uavBar);
		}
	}
	D3D12_RESOURCE_BARRIER aliasBarrier;
	aliasBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
	aliasBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
	for (auto& i : aliasBarriers) {
		aliasBarrier.Aliasing.pResourceBefore = i.first;
		aliasBarrier.Aliasing.pResourceAfter = i.second;
		resourceBarrierCommands.push_back(aliasBarrier);
	}
}
void ThreadCommand::ExecuteResBarrier() {
	if (!containedResources) return;
	containedResources = false;
	KillSame();
	if (!resourceBarrierCommands.empty()) {
		cmdList->ResourceBarrier(resourceBarrierCommands.size(), resourceBarrierCommands.data());
		resourceBarrierCommands.clear();
	}
	uavBarriersDict.Clear();
	aliasBarriers.clear();
	aliasBarriersDict.Clear();
	barrierRecorder.IterateAll([](ResourceBarrierCommand& cmd) -> void {
		cmd.index = -1;
	});
}
void ThreadCommand::Clear() {
	if (backToInitState.size() > 0) {
		backToInitState.IterateAll([&](GPUResourceBase const* key, GPUResourceState value) -> void {
			UpdateResState(value, key);
		});
	} else if (!containedResources)
		return;
	containedResources = false;
	KillSame();
	if (!resourceBarrierCommands.empty()) {
		cmdList->ResourceBarrier(resourceBarrierCommands.size(), resourceBarrierCommands.data());
		resourceBarrierCommands.clear();
	}
	uavBarriersDict.Clear();
	barrierRecorder.Clear();
	aliasBarriers.clear();
	aliasBarriersDict.Clear();
	backToInitState.Clear();
}
void ThreadCommand::UpdateResState(GPUResourceState beforeState, GPUResourceState afterState, GPUResourceBase const* resource, bool toInit) {
	containedResources = true;
	auto ite = barrierRecorder.Find(resource->GetInstanceID());
	if (!ite) {
		barrierRecorder.Insert(resource->GetInstanceID(), ResourceBarrierCommand(resource, afterState, resourceBarrierCommands.size()));
		if (toInit) {
			backToInitState.Insert(resource, beforeState);
		}
		resourceBarrierCommands.push_back(CD3DX12_RESOURCE_BARRIER::Transition(
			resource->GetResource(),
			resource->GetGFXResourceState(beforeState),
			resource->GetGFXResourceState(afterState)));
	} else if (ite.Value().index < 0) {
		auto&& cmd = ite.Value();
		cmd.index = resourceBarrierCommands.size();
		if (cmd.targetState != afterState) {
			resourceBarrierCommands.push_back(CD3DX12_RESOURCE_BARRIER::Transition(
				resource->GetResource(),
				resource->GetGFXResourceState(cmd.targetState),
				resource->GetGFXResourceState(afterState)));
			cmd.targetState = afterState;
		}
	} else {
		auto&& cmd = ite.Value();
		resourceBarrierCommands[cmd.index].Transition.StateAfter = resource->GetGFXResourceState(afterState);
		cmd.targetState = afterState;
	}
}

ThreadCommand::ThreadCommand(GFXDevice* device, GFXCommandListType type, ObjectPtr<CommandAllocator> const& allocator)
	: barrierRecorder(32),
	  uavBarriersDict(32),
	  aliasBarriersDict(32),
	  commandListType(type) {

	resourceBarrierCommands.reserve(32);
	if (allocator) {
		cmdAllocator = allocator;
		managingAllocator = false;
	} else {
		managingAllocator = true;
		cmdAllocator = MakeObjectPtr(
			new CommandAllocator(device, type));
	}
	ThrowIfFailed(device->device()->CreateCommandList(
		0,
		(D3D12_COMMAND_LIST_TYPE)type,
		cmdAllocator->GetAllocator().Get(),// Associated command allocator
		nullptr,						   // Initial PipelineStateObject
		IID_PPV_ARGS(&cmdList)));
	cmdList->Close();
}
ThreadCommand::~ThreadCommand() {
}
