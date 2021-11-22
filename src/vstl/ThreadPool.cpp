#pragma vengine_package vengine_dll
#include <vstl/ThreadPool.h>
static thread_local bool vengineTpool_isWorkerThread = false;
bool ThreadPool::IsWorkerThread() {
	return vengineTpool_isWorkerThread;
}

ThreadPool::ThreadPool(size_t workerThreadCount)
	: counters(32) {
    auto Max = []<typename T>(T const& a, T const&b) {
        if (a > b) return a;
        return b;
    };
	workerThreadCount = Max.operator()<size_t>(workerThreadCount, 1);
	this->workerThreadCount = workerThreadCount;
	enabled.test_and_set(std::memory_order_release);
	threads.reserve(workerThreadCount);
	auto LockThread = [this](std::mutex& mtx, std::condition_variable& cv, auto&& beforeLock, auto&& afterLock) {
		beforeLock();

		while (
			enabled.test(std::memory_order_acquire)
			&& taskList.Length() == 0) {
			std::unique_lock lck(mtx);
			cv.wait(lck);
		}
		afterLock();
	};
	vstd::function<void()> runThread = [this, LockThread]() {
		vengineTpool_isWorkerThread = true;
		while (enabled.test(std::memory_order_acquire)) {
			while (auto func = taskList.Pop()) {
				ThreadTaskHandle::TaskData* ptr = *func;
				ThreadExecute(ptr);
			}
			LockThread(
				threadLock, cv, []() {}, []() {});
		}
	};
	runBackupThread = [this, LockThread](size_t threadIndex) {
		vengineTpool_isWorkerThread = true;
		while (enabled.test(std::memory_order_acquire)) {
			while (auto func = taskList.Pop()) {
				ThreadExecute(*func);
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
	std::lock_guard ll(threadVectorLock);
	auto sz = backupThreads.size();
	std::thread t(runBackupThread, sz);
	backupThreads.emplace_back(std::move(t));
}

void ThreadPool::ThreadExecute(ThreadTaskHandle::TaskData* ptr) {

	uint8_t expected = static_cast<uint8_t>(ThreadTaskHandle::TaskState::Executed);
	if (!ptr->state.compare_exchange_strong(
			expected,
			static_cast<uint8_t>(ThreadTaskHandle::TaskState::Working),
			std::memory_order_acq_rel,
			std::memory_order_relaxed)) {
		return;
	}

	if (ptr->func)
		ptr->func();
	ptr->state.store(static_cast<uint8_t>(ThreadTaskHandle::TaskState::Finished), std::memory_order_release);
	if (ptr->isWaitable)
	{
		auto&& mtxPtr = ptr->mainThreadLocker;
		{
			std::lock_guard lck(mtxPtr->first);
			mtxPtr->second.notify_all();
		}
		ptr->ReleaseThreadLocker();
	}
	int64 needNotifyThread = -1;
	for (auto& i : ptr->dependedJobs) {
		ThreadTaskHandle::TaskData* d = i;
		if (d->dependCount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
			if (d->state.load(std::memory_order_acquire) != static_cast<uint8_t>(ThreadTaskHandle::TaskState::Executed)) continue;
			taskList.Push(std::move(i));
			needNotifyThread++;
		}
	}
	ptr->dependedJobs.clear();
	if (needNotifyThread > 0) {
		if (needNotifyThread < workerThreadCount) {
			std::lock_guard lck(threadLock);
			for (auto i : vstd::range(needNotifyThread))
				cv.notify_one();
		} else {
			std::lock_guard lck(threadLock);
			cv.notify_all();
		}
	}
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

ThreadTaskHandle ThreadPool::GetTask(vstd::function<void()> func, bool waitable) {
	return ThreadTaskHandle(this, std::move(func), waitable);
}
ThreadTaskHandle ThreadPool::GetFence(bool waitable) {
	return ThreadTaskHandle(this, waitable);
}
ThreadTaskHandle ThreadPool::M_GetParallelTask(
	vstd::function<void(size_t)>&& func,
	size_t parallelCount,
	size_t threadCount,
	bool waitable) {
	if (parallelCount) {
		return ThreadTaskHandle(this, std::move(func), parallelCount, threadCount, waitable);
	}
	return GetFence(waitable);
}

ThreadTaskHandle ThreadPool::M_GetBeginEndTask(
	vstd::function<void(size_t, size_t)>&& func,
	size_t parallelCount,
	size_t threadCount,
	bool waitable) {
	if (parallelCount) {
		return ThreadTaskHandle(this, std::move(func), parallelCount, threadCount, waitable);
	}
	return GetFence(waitable);
}

void ThreadPool::ExecuteTask(vstd::ObjectPtr<ThreadTaskHandle::TaskData> const& task, int64& accumulateCount) {
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
			ExecuteTask(i, accumulateCount);
		}
		ptr->dependingJob.clear();
	} else {
		taskList.Push(task);
		accumulateCount++;
	}
}
void ThreadPool::EnableThread(int64 enableCount) {
	enableCount -= vengineTpool_isWorkerThread ? 1 : 0;
	if (enableCount <= 0)
		return;
	if (enableCount < workerThreadCount) {
		std::lock_guard lck(threadLock);
		for (auto i : vstd::range(enableCount))
			cv.notify_one();
	} else {
		std::lock_guard lck(threadLock);
		cv.notify_all();
	}
}