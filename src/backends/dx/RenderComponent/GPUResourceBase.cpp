#include <RenderComponent/GPUResourceBase.h>
#include <PipelineComponent/FrameResource.h>
GPUResourceBase::GPUResourceBase(GFXDevice* device, GPUResourceType resourceType, IGPUAllocator* allocator)
	: resourceType(resourceType),
	  allocator(allocator),
	  device(device) {
}
GPUResourceBase::~GPUResourceBase() {
	if (allocator) {
		allocator->Release(GetInstanceID());
	}
}
void GPUResourceBase::ReleaseAfterFrame(luisa::compute::FrameResource* resource) {
	resource->deferredDeleteResource.emplace_back(std::move(Resource));
	if (allocator) {
		uint64 instanceID = GetInstanceID();
		IGPUAllocator* allocator = this->allocator;
		resource->afterSyncTask.emplace_back([instanceID, allocator]() {
			allocator->Release(instanceID);
		});
		this->allocator = nullptr;
	}
}
