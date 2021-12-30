#pragma vengine_package vengine_directx
#include <DXRuntime/CommandQueue.h>
#include <DXRuntime/CommandBuffer.h>
#include <DXRuntime/CommandAllocator.h>
#include <Resource/IGpuAllocator.h>
namespace toolhub::directx {
CommandQueue::CommandQueue(
	Device* device,
	IGpuAllocator* resourceAllocator,
	D3D12_COMMAND_LIST_TYPE type)
	: device(device),
	  type(type),
	  resourceAllocator(resourceAllocator), thd([this] { ExecuteThread(); }) {
	D3D12_COMMAND_QUEUE_DESC queueDesc = {};
	queueDesc.Type = type;
	queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_DISABLE_GPU_TIMEOUT;
	ThrowIfFailed(device->device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(queue.GetAddressOf())));
	ThrowIfFailed(device->device->CreateFence(
		0,
		D3D12_FENCE_FLAG_NONE,
		IID_PPV_ARGS(&cmdFence)));
}
CommandQueue::AllocatorPtr CommandQueue::CreateAllocator() {
	auto newPtr = allocatorPool.Pop();
	if (newPtr) {
		return std::move(*newPtr);
	}
	AllocatorPtr p(new CommandAllocator(device, resourceAllocator, type));
	return p;
}
void CommandQueue::ExecuteThread() {
	while (enabled) {
		while (auto b = executedAllocators.Pop()) {
			(*b)->Complete(cmdFence.Get(), executedFrame + 1);
			(*b)->Reset();
			allocatorPool.Push(std::move(*b));
			executedFrame++;
			{
				std::unique_lock lck(mtx);
				mainCv.notify_all();
			}
		}
		{
			std::unique_lock lck(mtx);
			while (executedFrame >= lastFrame) {
				waitCv.wait(lck);
			}
		}
	}
}

CommandQueue::~CommandQueue() {
	{
		std::unique_lock lck(mtx);
		enabled = false;
		executedFrame = 0;
		lastFrame = 1;
		waitCv.notify_one();
	}
	thd.join();
}
uint64 CommandQueue::Execute(AllocatorPtr&& alloc) {
	alloc->Execute(queue.Get(), cmdFence.Get(), lastFrame + 1);
	executedAllocators.Push(std::move(alloc));
	{
		std::lock_guard lck(mtx);
		lastFrame++;
		waitCv.notify_one();
	}
	//TODO: notify thread;
	return lastFrame;
}
void CommandQueue::Complete(uint64 fence) {
	std::unique_lock lck(mtx);
	while (executedFrame < fence) {
		mainCv.wait(lck);
	}
}
}// namespace toolhub::directx