#pragma once
#include <Common/GFXUtil.h>
#include <util/LockFreeArrayQueue.h>
#include <util/Runnable.h>
class SeparableRenderer;
class VENGINE_DLL_RENDERER SeparableRendererManager {
public:
	SeparableRendererManager();

	~SeparableRendererManager();
	void AddRenderer(SeparableRenderer* renderer, uint customSettings);
	void DeleteRenderer(SeparableRenderer* renderer, uint customSettings, bool deleteSelf);
	void UpdateRenderer(SeparableRenderer* renderer, uint customSettings);
	void Execute(
		GFXDevice* device,
		Runnable<void(GFXDevice*, SeparableRenderer*, uint)> const& lastFrameUpdateFunction,
		Runnable<bool(GFXDevice*, SeparableRenderer*, uint)> const& addFunction,
		Runnable<void(GFXDevice*, SeparableRenderer*, SeparableRenderer*, uint, bool)> const& removeFunction,// device, current, last, custom, isLast
		Runnable<bool(GFXDevice*, SeparableRenderer*, uint)> const& updateFunction,
		Runnable<void(SeparableRenderer*)> const& rendDisposer);
	uint64 GetElementCount() const { return elements.size(); }
	VSTL_OVERRIDE_OPERATOR_NEW
private:
	struct CallCommand {
		SeparableRenderer* renderer;
		uint updateOpe;
		bool deleteSelf;
	};
	vstd::vector<SeparableRenderer*> elements;
	struct CallCommandList {
		LockFreeArrayQueue<CallCommand> addCallCmds;
		LockFreeArrayQueue<CallCommand> removeCallCmds;
		LockFreeArrayQueue<CallCommand> updateCallCmds;
	};
	ArrayList<CallCommand> callCmdsCache;
	ArrayList<CallCommand> lastUpdateQueue;
	CallCommandList callCmds;
	VSTL_DELETE_COPY_CONSTRUCT(SeparableRendererManager)
	VSTL_DELETE_MOVE_CONSTRUCT(SeparableRendererManager)
};
