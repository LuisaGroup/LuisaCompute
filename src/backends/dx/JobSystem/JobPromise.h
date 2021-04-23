#pragma once
#include <Common/Common.h>
#include <JobSystem/JobBucket.h>
template<typename T>
class JobPromise {
	template<typename F>
	friend class JobPromise;

public:
	template<typename Func, typename... Args>
	JobPromise(JobBucket* bucket, Func&& f, Args&&... promiseArgs) {
		std::initializer_list<JobHandle> handles = {
			(promiseArgs.handle)...};
		handle = bucket->GetTask(
			handles,
			std::move(Runnable<void()>([&, f]() {
				std::lock_guard<spin_mutex> lck(finishLock);
				result.New(f((*promiseArgs.result)...));
			})));
	}
	JobPromise(JobPromise const& value) = delete;
	JobPromise(JobPromise&& value) = delete;

	T* operator->() {
		return result;
	}

	T const* operator->() const {
		return result;
	}

	T& operator*() {
		return *operator->();
	}
	T const& operator*() const {
		return *operator->();
	}

	~JobPromise() {
		if (finishLock.isLocked()) {
			VEngine_Log("Try to kill a busy JobPromise!\n");
			VENGINE_EXIT;
		}
	}

private:
	StackObject<T> result;
	spin_mutex finishLock;
	JobHandle handle;
};