#include <JobSystem/ThreadPool.h>
static thread_local bool vengineTpool_isWorkerThread = false;
ThreadPool::ThreadPool(size_t workerThreadCount, size_t backupThreadCount)
	: pool(MakeObjectPtr(
		[&]() {
			void* ptr = vengine_malloc(sizeof(PoolType));
			new (ptr) PoolType();
			return reinterpret_cast<PoolType*>(ptr);
		}(),
		[](void* ptr) {
			reinterpret_cast<PoolType*>(ptr)->~PoolType();
			vengine_free(ptr);
		})) {
	workerThreadCount = Max<size_t>(workerThreadCount, 1);
	backupThreadCount = Max<size_t>(backupThreadCount, 1);
	this->workerThreadCount = workerThreadCount;
	this->backupThreadCount = backupThreadCount;
	enabled.test_and_set(std::memory_order_release);
	threads.reserve(workerThreadCount + backupThreadCount);
	auto LockThread = [this](std::mutex& mtx, std::condition_variable& cv) {
		std::unique_lock lck(mtx);
		while (enabled.test(std::memory_order_acquire) && taskList.Length() == 0) {
			cv.wait(lck);
		}
	};
	Runnable<void()> runThread = [this, LockThread]() {
		vengineTpool_isWorkerThread = true;
		while (enabled.test(std::memory_order_acquire)) {
			ThreadExecute();
			LockThread(threadLock, cv);
		}
	};
	Runnable<void()> runBackupThread = [this, LockThread]() {
		vengineTpool_isWorkerThread = true;
		while (enabled.test(std::memory_order_acquire)) {
			LockThread(backupThreadLock, backupCV);
			ThreadExecute();
			LockThread(threadLock, cv);
		}
	};
	for (auto i : vstd::range(workerThreadCount - 1)) {
		threads.emplace_back(runThread);
	}
	threads.emplace_back(std::move(runThread));
	for (auto i : vstd::range(backupThreadCount - 1)) {
		threads.emplace_back(runBackupThread);
	}
	threads.emplace_back(std::move(runBackupThread));
}
void ThreadPool::ActiveOneBackupThread() {
	if (!vengineTpool_isWorkerThread) return;
	std::lock_guard lck(backupThreadLock);
	backupCV.notify_one();
}

void ThreadPool::ThreadExecute() {
	ObjectPtr<ThreadTaskHandle::TaskData> func;
	while (taskList.Pop(&func)) {
		ThreadTaskHandle::TaskData* ptr = func;
		uint8_t expected = static_cast<uint8_t>(ThreadTaskHandle::TaskState::Executed);

		if (!ptr->state.compare_exchange_strong(
				expected, static_cast<uint8_t>(ThreadTaskHandle::TaskState::Working), std::memory_order_acq_rel))
			continue;
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
	for (auto& i : threads) {
		i.join();
	}
	threads.clear();
}

ThreadTaskHandle ThreadPool::GetTask(Runnable<void()> func) {
	return ThreadTaskHandle(this, pool, std::move(func));
}
ThreadTaskHandle ThreadPool::GetParallelTask(
	Runnable<void(size_t)> func,
	size_t parallelCount,
	size_t threadCount) {
	vstd::vector<ObjectPtr<ThreadTaskHandle::TaskData>> tasks;
	threadCount = Min(threadCount, workerThreadCount);
	return ThreadTaskHandle(this, pool, std::move(func), parallelCount, threadCount);
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