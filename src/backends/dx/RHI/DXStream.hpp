#pragma once
#include <Common/GFXUtil.h>
#include <PipelineComponent/CommandAllocator.h>
#include <PipelineComponent/ThreadCommand.h>
#include <runtime/command_buffer.h>
namespace luisa::compute {
class DXStream {
public:
	DXStream(
		GFXDevice* device,
		GFXCommandListType type)
		: tCmd(device, type) {
	}

	void Sync(ID3D12Fence* fence) {
		if (signalIndex == 0) return;
		if (fence->GetCompletedValue() < signalIndex) {
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

	void Execute(CommandBuffer&& buffer, GFXCommandQueue* queue, ID3D12Fence* fence, uint64& cpuSignalIndex) {
		tCmd.ResetCommand();
		//TODO: execute buffer
		tCmd.CloseCommand();
		std::initializer_list<ID3D12CommandList*> cmd = {tCmd.GetCmdList()};
		queue->ExecuteCommandLists(cmd.size(), cmd.begin());
		queue->Signal(fence, cpuSignalIndex);
		signalIndex = cpuSignalIndex;
		cpuSignalIndex++;
	}
	DECLARE_VENGINE_OVERRIDE_OPERATOR_NEW

private:
	ThreadCommand tCmd;
	uint64 signalIndex = 0;
};
}// namespace luisa::compute