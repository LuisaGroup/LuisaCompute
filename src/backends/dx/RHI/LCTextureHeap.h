#pragma once
#include <RenderComponent/RenderComponentInclude.h>
#include <PipelineComponent/DXAllocator.h>
#include <PipelineComponent/ThreadCommand.h>
namespace luisa::compute {

class LCTextureHeap {
private:
	vstd::optional<UploadBuffer> uploadBuffer;
	vstd::optional<StructuredBuffer> defaultBuffer;
	struct TextureData {
		uint heapID;
		uint samplerID;
	};

public:
	LCTextureHeap(GFXDevice* device, uint size) {
		uploadBuffer.New(device, size, false, sizeof(TextureData), DXAllocator::GetBufferAllocator());
		auto eles = {StructuredBufferElement::Get(sizeof(TextureData), size)};
		defaultBuffer.New(
			device,
			eles,
			GPUResourceState_NonPixelShaderRes,
			DXAllocator::GetBufferAllocator());
	}

	void UploadID(
		uint index,
		uint texTargetID,
		uint samplerTargetID,
		LockFreeArrayQueue<Runnable<void(ThreadCommand*)>>& beforeStreamTasks) {
		TextureData t{texTargetID, samplerTargetID};
		uploadBuffer->CopyData(index, &t);
		beforeStreamTasks.Push([=](ThreadCommand* tcmd) {
			tcmd->RegistInitState(
				defaultBuffer->GetInitState(),
				defaultBuffer);
			tcmd->UpdateResState(
				GPUResourceState_CopyDest,
				defaultBuffer);
			Graphics::CopyBufferRegion(
				tcmd,
				defaultBuffer,
				index * sizeof(TextureData),
				uploadBuffer,
				index * sizeof(TextureData),
				sizeof(TextureData));
			tcmd->UpdateResState(
				defaultBuffer->GetInitState(),
				defaultBuffer);
		});
	}

	~LCTextureHeap() {}
};
}// namespace luisa::compute