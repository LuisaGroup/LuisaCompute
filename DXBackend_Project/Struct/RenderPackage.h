#pragma once
#include "../Common/GFXUtil.h"
class FrameResource;
class TransitionBarrierBuffer;
class PSOContainer;
struct RenderPackage
{
	GFXDevice* device;
	GFXCommandList*  commandList;
	FrameResource*  frameRes;
	TransitionBarrierBuffer* transitionBarrier;
	PSOContainer* psoContainer;
	constexpr RenderPackage(
		GFXDevice* device,
		GFXCommandList* commandList,
		FrameResource* frameRes,
		TransitionBarrierBuffer* transitionBarrier,
		PSOContainer* psoContainer) :
		device(device),
		commandList(commandList),
		frameRes(frameRes),
		transitionBarrier(transitionBarrier),
		psoContainer(psoContainer) {}

	constexpr RenderPackage() : device(nullptr),
		commandList(nullptr),
		frameRes(nullptr),
		transitionBarrier(nullptr),
		psoContainer(nullptr)
	{
		
	}
	constexpr bool operator==(const RenderPackage& p) const
	{
		return
			device == p.device &&
			commandList == p.commandList &&
			frameRes == p.frameRes &&
			transitionBarrier == p.transitionBarrier &&
			psoContainer == p.psoContainer;
	}
	constexpr bool operator!=(const RenderPackage& p) const
	{
		return !operator==(p);
	}
};