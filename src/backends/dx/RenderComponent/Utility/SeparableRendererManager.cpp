#include "SeparableRendererManager.h"
#include "SeparableRenderer.h"
#include <Common/GameTimer.h>
SeparableRendererManager::SeparableRendererManager() {}

SeparableRendererManager ::~SeparableRendererManager() {
}

void SeparableRendererManager::AddRenderer(SeparableRenderer* renderer, uint customSettings) {
	CallCommand cmd;
	cmd.renderer = renderer;
	cmd.updateOpe = customSettings;
	callCmds.addCallCmds.Push(cmd);
}

void SeparableRendererManager::DeleteRenderer(SeparableRenderer* renderer, uint customSettings, bool deleteSelf) {
	CallCommand cmd;
	cmd.renderer = renderer;
	cmd.updateOpe = customSettings;
	cmd.deleteSelf = deleteSelf;
	callCmds.removeCallCmds.Push(cmd);
}

void SeparableRendererManager::UpdateRenderer(SeparableRenderer* renderer, uint customSettings) {
	CallCommand cmd;
	cmd.renderer = renderer;
	cmd.updateOpe = customSettings;
	callCmds.updateCallCmds.Push(cmd);
}

void SeparableRendererManager::Execute(
	GFXDevice* device,
	Runnable<void(GFXDevice*, SeparableRenderer*, uint)> const& lastFrameUpdateFunction,
	Runnable<bool(GFXDevice*, SeparableRenderer*, uint)> const& addFunction,
	Runnable<void(GFXDevice*, SeparableRenderer*, SeparableRenderer*, uint, bool)> const& removeFunction,// device, current, last, custom, isLast
	Runnable<bool(GFXDevice*, SeparableRenderer*, uint)> const& updateFunction,
	Runnable<void(SeparableRenderer*)> const& rendDisposer) {
	uint64 const frameCount = GameTimer::GetFrameCount();
	uint64 addStart = 0;
	uint64 updateStart = 0;
	uint64 removeStart = 0;
	auto MarkFunc = [&]<bool filter>(
						LockFreeArrayQueue<CallCommand>& vec,
						uint64& sz,
						bool markAsRemoved) -> void {
		sz = callCmdsCache.size();
		CallCommand cmd;
		while (vec.Pop(&cmd)) {
			cmd.renderer->lastUpdatedFrame = frameCount;
			if constexpr (filter) {
				if (cmd.renderer->rendererRemoved) {
					continue;
				}
			} else {
				cmd.renderer->rendererRemoved = markAsRemoved;
			}
			callCmdsCache.push_back(cmd);
		}
	};

	MarkFunc.operator()<false>(callCmds.addCallCmds, addStart, true);
	MarkFunc.operator()<false>(callCmds.removeCallCmds, removeStart, false);
	MarkFunc.operator()<true>(callCmds.updateCallCmds, updateStart, true);
	for (auto&& i : lastUpdateQueue) {
		if (i.renderer->lastUpdatedFrame >= frameCount) continue;
		lastFrameUpdateFunction(device, i.renderer, i.updateOpe);
	}
	lastUpdateQueue.clear();
	if (callCmdsCache.empty()) return;
	//std::lock_guard lck(grpRenderDataMtx);
	for (uint64 i = addStart; i < removeStart; ++i) {
		auto&& v = callCmdsCache[i];
		if (v.renderer->listIndex != -1) continue;
		uint size = elements.size();
		v.renderer->listIndex = size;
		elements.push_back(v.renderer);
		if (addFunction(device, v.renderer, v.updateOpe)) {
			lastUpdateQueue.push_back(v);
		}
	}
	uint64 afterRemoveSize = elements.size() - (updateStart - removeStart);
	for (uint64 i = removeStart; i < updateStart; ++i) {
		auto&& v = callCmdsCache[i];
		auto arrEnd = elements.end() - 1;
		uint index = v.renderer->listIndex;
		uint last = elements.size() - 1;
		bool isLast = index == last;
		removeFunction(device, v.renderer, *arrEnd, v.updateOpe, isLast);
		if (!isLast) {
			(*arrEnd)->listIndex = index;
			elements[index] = *arrEnd;
		}
		elements.erase(arrEnd);
		if (v.deleteSelf) {
			rendDisposer(v.renderer);
		} else {
			v.renderer->listIndex = -1;
		}
	}
	for (uint64 i = updateStart; i < callCmdsCache.size(); ++i) {
		auto&& v = callCmdsCache[i];
		if (updateFunction(device, v.renderer, v.updateOpe)) {
			lastUpdateQueue.push_back(v);
		}
	}
	callCmdsCache.clear();
}
