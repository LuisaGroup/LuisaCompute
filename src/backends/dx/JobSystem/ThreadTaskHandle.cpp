#pragma vengine_package vengine_dll
#include <JobSystem/ThreadTaskHandle.h>
#include <JobSystem/ThreadPool.h>
ThreadTaskHandle::TaskData::TaskData(
	ObjectPtr<PoolType> const& p,
	Runnable<void()>&& func)
	: func(std::move(func)),
	  poolPtr(p) {
	state.store(static_cast<uint8_t>(TaskState::Waiting), std::memory_order_release);
}
ThreadTaskHandle::PoolType::PoolType() : pool(256) {}

ThreadTaskHandle::PoolType::~PoolType() {}
ThreadTaskHandle::TaskData::~TaskData() {}

ThreadTaskHandle::ThreadTaskHandle(
	ThreadPool* pool,
	ObjectPtr<PoolType> const& tPool,
	Runnable<void()>&& func) : pool(pool) {
	isArray = false;
	auto ptr = MakeObjectPtr(
		tPool->pool.New_Lock(tPool->mtx, tPool, std::move(func)),
		[](void* ptr) {
			TaskData* pp = reinterpret_cast<TaskData*>(ptr);
			ObjectPtr<PoolType> pt = std::move(pp->poolPtr);
			pt->pool.Delete_Lock(pt->mtx, ptr);
		});
	taskFlag.New(std::move(ptr));
}

ThreadTaskHandle::ThreadTaskHandle(
	ThreadPool* tPool,
	ObjectPtr<PoolType> const& pool,
	Runnable<void(size_t)>&& func,
	size_t parallelCount,
	size_t threadCount) : pool(tPool) {
	threadCount = Min(threadCount, tPool->workerThreadCount);
	isArray = true;
	taskFlags.New();
	auto&& tasks = *taskFlags;
	size_t eachJobCount = parallelCount / threadCount;
	tasks.reserve(eachJobCount + 1);
	auto&& pp = pool->pool;
	auto&& pm = pool->mtx;
	auto AddTask = [&](size_t beg, size_t ed) {
		tasks.emplace_back(MakeObjectPtr(
			pp.New_Lock(pm, pool, [=]() {
				for (auto c : vstd::range(beg, ed)) {
					func(c);
				}
			}),
			[](void* ptr) {
				auto pp = reinterpret_cast<ThreadTaskHandle::TaskData*>(ptr);
				ObjectPtr<PoolType> pt = std::move(pp->poolPtr);
				pt->pool.Delete_Lock(pt->mtx, ptr);
			}));
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

ThreadTaskHandle::ThreadTaskHandle(
	ThreadPool* tPool,
	ObjectPtr<PoolType> const& pool,
	Runnable<void(size_t, size_t)>&& func,
	size_t parallelCount,
	size_t threadCount) : pool(tPool) {
	threadCount = Min(threadCount, tPool->workerThreadCount);
	isArray = true;
	taskFlags.New();
	auto&& tasks = *taskFlags;
	size_t eachJobCount = parallelCount / threadCount;
	tasks.reserve(eachJobCount + 1);
	auto&& pp = pool->pool;
	auto&& pm = pool->mtx;
	auto AddTask = [&](size_t beg, size_t ed) {
		tasks.emplace_back(MakeObjectPtr(
			pp.New_Lock(pm, pool, [=]() {
				func(beg, ed);
			}),
			[](void* ptr) {
				auto pp = reinterpret_cast<ThreadTaskHandle::TaskData*>(ptr);
				ObjectPtr<PoolType> pt = std::move(pp->poolPtr);
				pt->pool.Delete_Lock(pt->mtx, ptr);
			}));
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
	auto checkExecuteFunc = [&](ObjectPtr<TaskData> const& p) {
		auto state = static_cast<TaskState>(p->state.load(std::memory_order_acquire));
		if (state != TaskState::Waiting) return;
		pool->ExecuteTask(p);
	};
	auto func = [&](TaskData* p) {
		auto state = static_cast<TaskState>(p->state.load(std::memory_order_acquire));
		if (state == TaskState::Finished) return;
		std::unique_lock lck(p->mtx);
		while (p->state.load(std::memory_order_acquire) != static_cast<uint8_t>(TaskState::Finished)) {
			p->cv.wait(lck);
		}
	};
	if (isArray) {
		for (auto& taskFlag : *taskFlags)
			checkExecuteFunc(taskFlag);
		pool->ActiveOneBackupThread();
		for (auto& taskFlag : *taskFlags) {
			func(taskFlag);
		}
	} else {
		checkExecuteFunc(*taskFlag);
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

void ThreadTaskHandle::AddDepend(ThreadTaskHandle const& handle) const {
	AddDepend(std::span<ThreadTaskHandle const>(&handle, 1));
}

void ThreadTaskHandle::AddDepend(std::span<ThreadTaskHandle const> handles) const {

	auto func = [&](ObjectPtr<TaskData> const& selfPtr, ObjectPtr<TaskData> const& dep) {
		TaskData* p = dep;
		TaskState state = static_cast<TaskState>(p->state.load(std::memory_order_acquire));
		if (state != TaskState::Waiting) {
			vstl_log("Try To depend on a executed job!\n");
			VSTL_ABORT();
		}
		TaskData* self = selfPtr;
		self->dependingJob.push_back(dep);
		p->dependedJobs.push_back(selfPtr);
	};
	auto executeSelf = [&](ObjectPtr<TaskData> const& self, ThreadTaskHandle const& handle) {
		if (handle.isArray) {
			for (auto& i : *handle.taskFlags) {
				func(self, i);
			}
			self->dependCount.fetch_add(handle.taskFlags->size(), std::memory_order_relaxed);

		} else {
			func(self, *handle.taskFlag);
			self->dependCount.fetch_add(1, std::memory_order_relaxed);
		}
	};
	for (auto& handle : handles) {
		if (isArray) {
			for (auto& i : *taskFlags) {
				executeSelf(i, handle);
			}
		} else {
			executeSelf(*taskFlag, handle);
		}
	}
}

void ThreadTaskHandle::Execute() const {

	if (isArray) {
		for (auto& i : *taskFlags) {
			pool->ExecuteTask(i);
		}
	} else {
		pool->ExecuteTask(*taskFlag);
	}
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