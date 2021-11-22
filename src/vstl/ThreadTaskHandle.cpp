#pragma vengine_package vengine_dll
#include <vstl/ThreadTaskHandle.h>
#include <vstl/ThreadPool.h>
ThreadTaskHandle::TaskData::TaskData(
	vstd::function<void()>&& func, bool waitable)
	: func(std::move(func)) {
	isWaitable = waitable;
	if (waitable)
		mainThreadLocker.New();
	state.store(static_cast<uint8_t>(TaskState::Waiting), std::memory_order_release);
}
ThreadTaskHandle::TaskData::TaskData(bool waitable) {
	isWaitable = waitable;
	if (waitable)
		mainThreadLocker.New();
	state.store(static_cast<uint8_t>(TaskState::Waiting), std::memory_order_release);
}
ThreadTaskHandle::TaskData::~TaskData() {
	if (isWaitable && refCount > 0) {
		mainThreadLocker.Delete();
	}
}

ThreadTaskHandle::TaskData::Locker* ThreadTaskHandle::TaskData::GetThreadLocker() {
	std::lock_guard lck(lockerMtx);
	if (refCount == 0 || !mainThreadLocker) return nullptr;
	refCount++;
	return mainThreadLocker;
}
void ThreadTaskHandle::TaskData::ReleaseThreadLocker() {
	std::unique_lock lck(lockerMtx);
	if ((--refCount) == 0) {
		lck.unlock();
		mainThreadLocker.Delete();
	}
}

ThreadTaskHandle::ThreadTaskHandle(
	ThreadPool* pool, bool waitable) : pool(pool) {
	isArray = false;
	auto ptr = vstd::MakeObjectPtr(
		new TaskData(waitable));
	taskFlag.New(std::move(ptr));
}
ThreadTaskHandle::ThreadTaskHandle(
	ThreadPool* pool,
	vstd::function<void()>&& func, bool waitable) : pool(pool) {
	isArray = false;
	auto ptr = vstd::MakeObjectPtr(new TaskData(std::move(func), waitable));
	taskFlag.New(std::move(ptr));
}

ThreadTaskHandle::ThreadTaskHandle(
	ThreadPool* tPool,
	vstd::function<void(size_t)>&& func,
	size_t parallelCount,
	size_t threadCount, bool waitable) : pool(tPool) {
    auto Min = [](auto &&a, auto &&b) {
        if (a < b) return a;
        return b;
    };
	threadCount = Min(threadCount, tPool->workerThreadCount);
	threadCount = Min(threadCount, parallelCount);
	isArray = true;
	taskFlags.New();
	auto&& tasks = *taskFlags;
	tasks.reserve(threadCount + 1);
	
	auto taskCounter = tPool->counters.New_Lock(tPool->counterMtx, threadCount);
	for (size_t v = 0; v < threadCount - 1; ++v) {
		tasks.emplace_back(vstd::MakeObjectPtr(
			new TaskData(
				[=]() {
					auto i = taskCounter->counter.fetch_add(1, std::memory_order_acq_rel);
					while (i < parallelCount) {
						func(i);
						i = taskCounter->counter.fetch_add(1, std::memory_order_acq_rel);
					}
					if (taskCounter->finishedCount.fetch_sub(1, std::memory_order_relaxed) == 1) {
						tPool->counters.Delete_Lock(tPool->counterMtx, taskCounter);
					}
				},
				waitable)));
	}
	tasks.emplace_back(vstd::MakeObjectPtr(
		new TaskData(
			[=, func = std::move(func)]() {
				auto i = taskCounter->counter.fetch_add(1, std::memory_order_acq_rel);
				while (i < parallelCount) {
					func(i);
					i = taskCounter->counter.fetch_add(1, std::memory_order_acq_rel);
				}
				if (taskCounter->finishedCount.fetch_sub(1, std::memory_order_relaxed) == 1) {
					tPool->counters.Delete_Lock(tPool->counterMtx, taskCounter);
				}
			},
			waitable)));
}

ThreadTaskHandle::ThreadTaskHandle(
	ThreadPool* tPool,
	vstd::function<void(size_t, size_t)>&& func,
	size_t parallelCount,
	size_t threadCount,
	bool waitable) : pool(tPool) {
    auto Min = [](auto &&a, auto &&b) {
        if (a < b) return a;
        return b;
    };
	threadCount = Min(threadCount, tPool->workerThreadCount);
	threadCount = Min(threadCount, parallelCount);
	isArray = true;
	taskFlags.New();
	auto&& tasks = *taskFlags;
	size_t eachJobCount = parallelCount / threadCount;
	tasks.reserve(threadCount + 1);
	auto AddTask = [&](size_t beg, size_t ed) {
		tasks.emplace_back(vstd::MakeObjectPtr(
			new TaskData(
				[=]() {
					func(beg, ed);
				},
				waitable)));
	};
	for (size_t i = 0; i < threadCount; ++i) {
		AddTask(i * eachJobCount, (i + 1) * eachJobCount);
	}
	size_t full = eachJobCount * threadCount;
	size_t lefted = parallelCount - full;
	if (lefted > 0) {
		AddTask(full, parallelCount);
	}
}

void ThreadTaskHandle::Complete() const {
	struct TPoolCounter {
		ThreadPool* t;
		TPoolCounter(
			ThreadPool* t) : t(t) {
			t->pausedWorkingThread++;
		}
		~TPoolCounter() {
			t->pausedWorkingThread--;
		}
	};
	TPoolCounter tcounter(pool);
	int64 needEnableState = 0;
	auto checkExecuteFunc = [&](vstd::ObjectPtr<TaskData> const& p) {
		pool->ExecuteTask(p, needEnableState);
	};
	auto func = [&](TaskData* p) {
		if (!p->isWaitable) {
			VEngine_Log("Try to wait non-waitable job!");
			VENGINE_EXIT;
		}
		auto state = static_cast<TaskState>(p->state.load(std::memory_order_acquire));
		if (state == TaskState::Finished) return;
		auto mtxPtr = p->GetThreadLocker();
		if (mtxPtr) {
			auto disp = vstd::create_disposer([p]() {
				p->ReleaseThreadLocker();
			});
			{
				std::unique_lock lck(mtxPtr->first);
				while (p->state.load(std::memory_order_acquire) != static_cast<uint8_t>(TaskState::Finished)) {
					mtxPtr->second.wait(lck);
				}
			}
		}
	};
	if (isArray) {
		for (auto& taskFlag : *taskFlags)
			checkExecuteFunc(taskFlag);
		pool->EnableThread(needEnableState);
		pool->ActiveOneBackupThread();
		for (auto& taskFlag : *taskFlags) {
			func(taskFlag);
		}
	} else {
		checkExecuteFunc(*taskFlag);
		pool->EnableThread(needEnableState);
		pool->ActiveOneBackupThread();
		func(*taskFlag);
	}
}

bool ThreadTaskHandle::IsComplete() const {
	if (!pool) return true;
	auto func = [&](TaskData* p) {
		return static_cast<TaskState>(p->state.load(std::memory_order_relaxed)) == TaskState::Finished;
	};
	if (isArray) {
		for (auto& taskFlag : *taskFlags) {
			if (!func(taskFlag)) return false;
		}
		return true;
	} else {
		return func(*taskFlag);
	}
}
template<typename H>
void ThreadTaskHandle::TAddDepend(H&& handle) const {
	auto func = [&](vstd::ObjectPtr<TaskData> const& selfPtr, auto&& dep, uint64& dependAdd) {
		TaskData* p = dep;
		TaskData* self = selfPtr;

		TaskState state = static_cast<TaskState>(p->state.load(std::memory_order_acquire));
		if ((uint8_t)state < (uint8_t)TaskState::Executed) {
			p->dependedJobs.push_back(selfPtr);
			if constexpr (std::is_lvalue_reference_v<H>) {
				self->dependingJob.push_back(dep);
			} else {
				self->dependingJob.push_back(std::move(dep));
			}
			dependAdd++;
		} else {
			VEngine_Log("Try depend on executing task!");
			VENGINE_EXIT;
		}
	};
	auto executeSelf = [&](vstd::ObjectPtr<TaskData> const& self, auto&& handle) {
		uint64 v = 0;
		if (handle.isArray) {
			for (auto& i : *handle.taskFlags) {
				func(self, i, v);
			}
		} else {
			func(self, *handle.taskFlag, v);
		}
		self->dependCount.fetch_add(v, std::memory_order_relaxed);
	};
	if (isArray) {
		for (auto& i : *taskFlags) {
			executeSelf(i, handle);
		}
	} else {
		executeSelf(*taskFlag, handle);
	}
}
void ThreadTaskHandle::AddDepend(ThreadTaskHandle const& handle) const {
	TAddDepend<ThreadTaskHandle const&>(handle);
}

void ThreadTaskHandle::AddDepend(std::span<ThreadTaskHandle const> handles) const {

	for (auto& handle : handles) {
		AddDepend(handle);
	}
}

void ThreadTaskHandle::Execute() const {
	int64 needEnableState = 0;
	if (isArray) {
		for (auto& i : *taskFlags) {
			pool->ExecuteTask(i, needEnableState);
		}
	} else {
		pool->ExecuteTask(*taskFlag, needEnableState);
	}
	pool->EnableThread(needEnableState);
}
ThreadTaskHandle::ThreadTaskHandle(ThreadTaskHandle const& v)
	: pool(v.pool),
	  isArray(v.isArray) {
	if (isArray) {
		taskFlags.New(*v.taskFlags);
	} else {
		taskFlag.New(*v.taskFlag);
	}
}
ThreadTaskHandle::ThreadTaskHandle(ThreadTaskHandle&& v)
	: pool(v.pool),
	  isArray(v.isArray) {
	if (isArray) {
		taskFlags.New(std::move(*v.taskFlags));
	} else {
		taskFlag.New(std::move(*v.taskFlag));
	}
}
ThreadTaskHandle::~ThreadTaskHandle() {
	if (isArray) {
		taskFlags.Delete();
	} else {
		taskFlag.Delete();
	}
}