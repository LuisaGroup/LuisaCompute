#include "RenderCommand.h"
LockFreeArrayQueue<RenderCommandExecutable, false> RenderCommand::queue(100);
void RenderCommand::UpdateResState(
	RenderCommand* ptr) {
	queue.Push<RenderCommandExecutable>({ptr});
}
bool RenderCommand::ExecuteResBarrier(
	GFXDevice* device,
	ThreadCommand* directCommandList,
	ThreadCommand* copyCommandList) {
	RenderCommandExecutable ptr;
	bool v = false;
	v = queue.Pop(&ptr);
	if (!v) return false;
	ptr.ptr->Execute(device, directCommandList, copyCommandList);
	{
		delete (ptr.ptr);
	}
	return true;
}
