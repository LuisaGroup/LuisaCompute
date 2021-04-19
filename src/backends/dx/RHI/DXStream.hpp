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
		GFXCommandQueue* queue,
		GFXCommandListType listType)
		: listType(listType),
		  queue(queue)

	{
	}
	GFXCommandListType GetType() const {
		return listType;
	}

	static void WaitFence(
		ID3D12Fence* fence,
		uint64 signalIndex) {
		if (signalIndex > 0 && (fence->GetCompletedValue() < signalIndex)) {
#ifdef UNICODE
			LPCWSTR falseValue = (LPCWSTR) false;
#else
			LPCSTR falseValue = (LPCSTR) false;
#endif
			HANDLE eventHandle = CreateEventEx(nullptr, falseValue, false, EVENT_ALL_ACCESS);
			// Fire event when GPU hits current fence.
			ThrowIfFailed(fence->SetEventOnCompletion(signalIndex, eventHandle));
			// Wait until the GPU hits current fence event is fired.
			WaitForSingleObject(eventHandle, INFINITE);
			CloseHandle(eventHandle);
		}
	}
	void Sync(ID3D12Fence* fence, std::mutex& mtx) {
		{
			std::lock_guard lck(mtx);
			if (dispatchedRes.empty()) return;
			auto lastRes = *(dispatchedRes.end() - 1);
			WaitFence(fence, lastRes->signalIndex);
		}
		for (auto& i : dispatchedRes) {
			i->ReleaseTemp();
		}
		dispatchedRes.clear();
	}

	void WaitToFence(
		std::mutex& mtx,
		ID3D12Fence* fence,
		uint64 signal) {
		std::lock_guard lck(mtx);
		if (signal > 0)
			queue->Wait(
				fence, signal);
	}

	void Execute(
		GFXDevice* device,
		CommandBuffer&& buffer,
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
		dispatchedRes.push_back(tempRes);
		std::initializer_list<ID3D12CommandList*> cmd = {tempRes->tCmd.GetCmdList()};
		queue->ExecuteCommandLists(cmd.size(), cmd.begin());
		queue->Signal(fence, cpuSignalIndex);
		tempRes->signalIndex = cpuSignalIndex;
		lastSignal = cpuSignalIndex;
		cpuSignalIndex++;
		res.Push(tempRes);
	}
	DECLARE_VENGINE_OVERRIDE_OPERATOR_NEW
	uint64 GetSignal() const {
		return lastSignal;
	}

private:
	vengine::vector<FrameResource*> dispatchedRes;
	GFXCommandListType listType;
	GFXCommandQueue* queue;
	uint64 lastSignal = 0;
};
}// namespace luisa::compute
