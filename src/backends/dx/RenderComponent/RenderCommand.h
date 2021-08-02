#pragma once
#include <Common/GFXUtil.h>
#include <core/vstl/Pool.h>
#include <mutex>
#include <core/vstl/LockFreeArrayQueue.h>
class RenderCommand;
class ThreadCommand;
struct RenderCommandExecutable {
	RenderCommand* ptr = nullptr;
};
class VENGINE_DLL_RENDERER RenderCommand {
private:
	static LockFreeArrayQueue<RenderCommandExecutable, VEngine_AllocType::Default> queue;

public:
	DECLARE_VENGINE_OVERRIDE_OPERATOR_NEW
	virtual ~RenderCommand() {}

	virtual void Execute(
		GFXDevice* device,
		ThreadCommand* directCommandList,
		ThreadCommand* copyCommandList) = 0;
	static void UpdateResState(
		RenderCommand* ptr);
	static bool ExecuteResBarrier(
		GFXDevice* device,
		ThreadCommand* directCommandList,
		ThreadCommand* copyCommandList);
};