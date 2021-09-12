#pragma vengine_package vengine_dll
#include <JobSystem/ThreadPool.h>
static thread_local bool vengineTpool_isWorkerThread = false;
bool ThreadPool::IsWorkerThread() {
	return vengineTpool_isWorkerThread;
}

ThreadPool::ThreadPool(size_t workerThreadCount)
	: pool(MakeObjectPtr(
		[&]() {
			void* ptr = vstl_malloc(sizeof(PoolType));
			new (ptr) PoolType();
			return reinterpret_cast<PoolType*>(ptr);
		}(),
		[](void* ptr) {
			reinterpret_cast<PoolType*>(ptr)->~PoolType();
			vstl_free(ptr);
		})) {
	workerThreadCount = Max<size_t>(workerThreadCount, 1);
	this->workerThreadCount = workerThreadCount;
	enabled.test_and_set(std::memory_order_release);
	threads.reserve(workerThreadCount);
	auto LockThread = [this](std::mutex& mtx, std::condition_variable& cv, auto&& beforeLock, auto&& afterLock) {
		std::unique_lock lck(mtx);
		beforeLock();
		while (enabled.test(std::memory_order_acquire) && taskList.Length() == 0) {
			cv.wait(lck);
		}
		afterLock();
	};
	Runnable<void()> runThread = [this, LockThread]() {
		vengineTpool_isWorkerThread = true;
		while (enabled.test(std::memory_order_acquire)) {
			ObjectPtr<ThreadTaskHandle::TaskData> func;
			while (taskList.Pop(&func)) {
				ThreadTaskHandle::TaskData* ptr = func;
				ThreadExecute(ptr);
			}
			LockThread(
				threadLock, cv, []() {}, []() {});
		}
	};
	runBackupThread = [this, LockThread](size_t threadIndex) {
		vengineTpool_isWorkerThread = true;
		while (enabled.test(std::memory_order_acquire)) {
			ObjectPtr<ThreadTaskHandle::TaskData> func;
			while (taskList.Pop(&func)) {
				ThreadExecute(func);
				if (threadIndex >= pausedWorkingThread) break;
			}
			LockThread(
				backupThreadLock, backupCV, [&]() { waitingBackupThread++; },
				[&]() { waitingBackupThread--; });
		}
	};
	for (auto i : vstd::range(workerThreadCount - 1)) {
		threads.emplace_back(runThread);
	}
	threads.emplace_back(std::move(runThread));
}

void ThreadPool::ActiveOneBackupThread() {
	if (!vengineTpool_isWorkerThread) return;
	{
		std::lock_guard lck(backupThreadLock);
		if (waitingBackupThread > 0) {
			backupCV.notify_one();
			return;
		}
	}
	auto sz = backupThreads.size();
	std::thread t(runBackupThread, sz);
	std::lock_guard ll(threadVectorLock);
	backupThreads.emplace_back(std::move(t));
}

void ThreadPool::ThreadExecute(ThreadTaskHandle::TaskData* ptr) {

	uint8_t expected = static_cast<uint8_t>(ThreadTaskHandle::TaskState::Executed);

	if (!ptr->state.compare_exchange_strong(
			expected, static_cast<uint8_t>(ThreadTaskHandle::TaskState::Working), std::memory_order_acq_rel)) {
		return;
	}
	ptr->func();
	{
		ptr->state.store(static_cast<uint8_t>(ThreadTaskHandle::TaskState::Finished), std::memory_order_release);
		std::lock_guard lck(ptr->mtx);
		ptr->cv.notify_one();
	}
	for (auto& i : ptr->dependedJobs) {
		ThreadTaskHandle::TaskData* d = i;
		if (d->dependCount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
			if (d->state.load(std::memory_order_acquire) != static_cast<uint8_t>(ThreadTaskHandle::TaskState::Executed)) continue;
			taskList.Push(i);
			std::lock_guard lck(threadLock);
			cv.notify_one();
		}
	}
	ptr->dependedJobs.clear();
}

ThreadPool::~ThreadPool() {
	{
		std::lock_guard lck(threadLock);
		enabled.clear(std::memory_order_release);
		cv.notify_all();
	}
	{
		std::lock_guard lck(backupThreadLock);
		backupCV.notify_all();
	}
	for (auto& i : backupThreads) {
		i.join();
	}
	for (auto& i : threads) {
		i.join();
	}
	threads.clear();
	backupThreads.clear();
}

ThreadTaskHandle ThreadPool::GetTask(Runnable<void()> func) {
	return ThreadTaskHandle(this, pool, std::move(func));
}
ThreadTaskHandle ThreadPool::GetFence() {
	return ThreadTaskHandle(this, pool, DoNothing);
}
ThreadTaskHandle ThreadPool::M_GetParallelTask(
	Runnable<void(size_t)>&& func,
	size_t parallelCount,
	size_t threadCount) {
	if (parallelCount) {
		return ThreadTaskHandle(this, pool, std::move(func), parallelCount, threadCount);
	}
	return GetFence();
}

ThreadTaskHandle ThreadPool::M_GetBeginEndTask(
	Runnable<void(size_t, size_t)>&& func,
	size_t parallelCount,
	size_t threadCount) {
	if (parallelCount) {
		return ThreadTaskHandle(this, pool, std::move(func), parallelCount, threadCount);
	}
	return GetFence();
}

void ThreadPool::ExecuteTask(ObjectPtr<ThreadTaskHandle::TaskData> const& task) {
	ThreadTaskHandle::TaskData* ptr = task;
	uint8_t expected = static_cast<uint8_t>(ThreadTaskHandle::TaskState::Waiting);
	if (!ptr->state.compare_exchange_strong(
			expected,
			static_cast<uint8_t>(ThreadTaskHandle::TaskState::Executed),
			std::memory_order_acq_rel)) {
		return;
	}
	if (ptr->dependCount.load(std::memory_order_acquire) > 0) {
		for (auto& i : ptr->dependingJob) {
			ExecuteTask(i);
		}
		ptr->dependingJob.clear();
	} else {
		taskList.Push(task);
		std::lock_guard lck(threadLock);
		cv.notify_one();
	}
}