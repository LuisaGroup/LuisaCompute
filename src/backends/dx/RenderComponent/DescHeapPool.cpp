//#endif
#include <RenderComponent/DescHeapPool.h>
DescHeapPool::DescHeapPool(
	uint size, uint initCapacity, D3D12_DESCRIPTOR_HEAP_TYPE type) : size(size), capacity(initCapacity), type(type) {
	poolValue.reserve(initCapacity);
	arr.reserve(10);
}
void DescHeapPool::Add(GFXDevice* device) {
	auto& newDesc = arr.emplace_back(
		new DescriptorHeap(device, type, size * capacity, type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV));
	DescriptorHeap* heap = newDesc.get();
	for (uint i = 0; i < capacity; ++i) {
		poolValue.emplace_back(heap, i * size);
	}
}
DescHeapElement DescHeapPool::Get(GFXDevice* device) {
	if (poolValue.empty())
		Add(device);
	return poolValue.erase_last();
}
void DescHeapPool::Return(const DescHeapElement& target) {
	poolValue.push_back(target);
}
