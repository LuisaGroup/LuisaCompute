#pragma once
#include <util/MetaLib.h>
#include <util/Memory.h>
#include <util/vector.h>
#include <util/Pool.h>
template<typename T>
class PoolAllocator {
private:
	StackObject<Pool<Storage<T, 1>, VEngine_AllocType::Default>> globalTransformPool;
	std::mutex transPoolLock;
	luisa::spin_mutex transAllocLock;
	std::atomic_bool poolInited = false;

public:
	PoolAllocator() {}
	PoolAllocator(PoolAllocator&& o) {
		poolInited = o.poolInited;
		o.poolInited = false;
		globalTransformPool.New(std::move(*o.globalTransformPool));
	}
	void* Allocate(size_t checkSize) {
		if (checkSize != sizeof(T)) {
			VEngine_Log(
				{"Failed Allocate Type ",
				 typeid(T).name(),
				 "\n"});
			VSTL_ABORT();
		}
		if (!poolInited) {
			std::lock_guard lck(transPoolLock);
			globalTransformPool.New(512);
			poolInited = true;
		}
		std::lock_guard lck(transAllocLock);
		return globalTransformPool->New();
	}
	void Free(void* ptr, size_t checkSize) {
		if (checkSize != sizeof(T)) {
			VEngine_Log(
				{"Failed Deallocate Type ",
				 typeid(T).name(),
				 "\n"});
			VSTL_ABORT();
		}
		std::lock_guard lck(transAllocLock);
		globalTransformPool->Delete(ptr);
	}
	~PoolAllocator() {
		if (poolInited) {
			globalTransformPool.Delete();
		}
	}
};
