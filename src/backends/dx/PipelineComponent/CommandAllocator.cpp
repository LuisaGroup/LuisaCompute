#include <PipelineComponent/CommandAllocator.h>
CommandAllocator::CommandAllocator(GFXDevice* device, GFXCommandListType type) {
	ThrowIfFailed(
		device->device()->CreateCommandAllocator((D3D12_COMMAND_LIST_TYPE)type, IID_PPV_ARGS(allocator.GetAddressOf())));
}
void CommandAllocator::Reset(uint64 frameIndex) {
	if (updatedFrame >= frameIndex) return;
	updatedFrame = frameIndex;
	ThrowIfFailed(
		allocator->Reset());
}
