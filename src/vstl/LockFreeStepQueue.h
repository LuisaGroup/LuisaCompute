#pragma once
#include <vstl/LockFreeArrayQueue.h>
namespace vstd {
template<typename T, size_t QUEUE_SET_POW = 2, VEngine_AllocType allocType = VEngine_AllocType::VEngine>
class LockFreeStepQueue {
	static_assert(QUEUE_SET_POW < 8, "size too large!");
	static constexpr uint8_t QUEUE_SET = 1 << QUEUE_SET_POW;
	StackObject<LockFreeArrayQueue<T, allocType>> queues[QUEUE_SET];
	std::atomic_size_t len = 0;
	std::atomic_uint8_t count = 0;
	std::atomic_uint8_t readCount = 0;

public:
	size_t Length() const { return len; }
	LockFreeStepQueue() {
		for (auto&& i : queues) {
			i.New();
		}
	}
	LockFreeStepQueue(size_t capacity) {
		for (auto&& i : queues) {
			i.New(capacity);
		}
	}
	~LockFreeStepQueue() {
		for (auto&& i : queues) {
			i.Delete();
		}
	}
	template<typename... Args>
	void Push(Args&&... args) {
		while (true) {
			if (queues[(++count) & (QUEUE_SET - 1)]->TryPush(std::forward<Args>(args)...)) {
				len.fetch_add(1, std::memory_order_relaxed);
				return;
			}
		}
	}
	optional<T> Pop() {
		while (len.load(std::memory_order_acquire) > 0) {
			auto opt = queues[(++readCount) & (QUEUE_SET - 1)]->TryPop();
			if (opt) {
				len.fetch_sub(1, std::memory_order_release);
				return opt;
			}
		}
		return optional<T>();
	}
};
}// namespace vstd