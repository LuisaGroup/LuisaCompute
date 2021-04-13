#pragma once
#include <Common/GFXUtil.h>
#include <PipelineComponent/CommandAllocator.h>
#include <PipelineComponent/ThreadCommand.h>
#include <Common/LockFreeArrayQueue.h>
#include <runtime/command_buffer.h>
#include <PipelineComponent/FrameResource.h>
#include <Common/Runnable.h>
namespace luisa::compute {
class DXStream {
public:
	using GetFrameResourceFunc = Runnable<FrameResource*(GFXDevice*, GFXCommandListType)>;
	DXStream(
		GFXDevice* device,
		GFXCommandListType listType)
		: listType(listType) {
	}
	GFXCommandListType GetType() const {
		return listType;
	}
	void Sync(ID3D12Fence* fence, std::mutex& mtx) {
		std::lock_guard lck(mtx);
		if (dispatchedRes.empty()) return;
		auto lastRes = *(dispatchedRes.end() - 1);
		if (fence->GetCompletedValue() < lastRes->signalIndex) {
#ifdef UNICODE
			LPCWSTR falseValue = (LPCWSTR) false;
#else
			LPCSTR falseValue = (LPCSTR) false;
#endif
			HANDLE eventHandle = CreateEventEx(nullptr, falseValue, false, EVENT_ALL_ACCESS);
			// Fire event when GPU hits current fence.
			ThrowIfFailed(fence->SetEventOnCompletion(lastRes->signalIndex, eventHandle));
			// Wait until the GPU hits current fence event is fired.
			WaitForSingleObject(eventHandle, INFINITE);
			CloseHandle(eventHandle);
		}
		for (auto& i : dispatchedRes) {
			i->ReleaseTemp();
		}
		dispatchedRes.clear();
	}

	void Execute(
		GFXDevice* device,
		CommandBuffer&& buffer,
		GFXCommandQueue* queue,
		ID3D12Fence* fence,
		GetFrameResourceFunc const& getResource,
		SingleThreadArrayQueue<FrameResource*>& res,
		std::mutex& mtx,
		uint64& cpuSignalIndex) {
		///////////// Local-Thread
		FrameResource* tempRes = getResource(device, listType);
		tempRes->tCmd.ResetCommand();
		//TODO: execute buffer
		tempRes->tCmd.CloseCommand();
		///////////// Global-Sync
		std::lock_guard lck(mtx);
		std::initializer_list<ID3D12CommandList*> cmd = {tempRes->tCmd.GetCmdList()};
		queue->ExecuteCommandLists(cmd.size(), cmd.begin());
		queue->Signal(fence, cpuSignalIndex);
		tempRes->signalIndex = cpuSignalIndex;
		dispatchedRes.push_back(tempRes);
		cpuSignalIndex++;
		res.Push(tempRes);
	}
	DECLARE_VENGINE_OVERRIDE_OPERATOR_NEW

private:
	vengine::vector<FrameResource*> dispatchedRes;
	GFXCommandListType listType;
};
}// namespace luisa::compute