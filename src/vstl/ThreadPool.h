#pragma once
#include <vstl/Common.h>
#include <vstl/LockFreeStepQueue.h>
#include <vstl/ThreadTaskHandle.h>
#include <EASTL/shared_ptr.h>
class LC_VSTL_API ThreadPool final {
	friend class ThreadTaskHandle;
	vstd::vector<std::thread> threads;
	vstd::vector<std::thread> backupThreads;
	vstd::LockFreeStepQueue<eastl::shared_ptr<ThreadTaskHandle::TaskData>, 2> taskList;
	void ThreadExecute(ThreadTaskHandle::TaskData*);
	std::atomic_flag enabled;
	std::mutex threadLock;
	std::mutex backupThreadLock;
	vstd::spin_mutex threadVectorLock;
	std::condition_variable cv;
	std::condition_variable backupCV;
	void ExecuteTask(eastl::shared_ptr<ThreadTaskHandle::TaskData> const& task, int64& accumulateCount);
	void EnableThread(int64 enableCount);
	size_t workerThreadCount;
	void ActiveOneBackupThread();
	ThreadTaskHandle M_GetParallelTask(vstd::function<void(size_t)>&& func, size_t parallelCount, size_t threadCount, bool waitable);
	ThreadTaskHandle M_GetBeginEndTask(vstd::function<void(size_t, size_t)>&& func, size_t parallelCount, size_t threadCount, bool waitable);
	vstd::function<void(size_t)> runBackupThread;
	std::atomic_int64_t waitingBackupThread = 0;
	std::atomic_int64_t pausedWorkingThread = 0;
	struct Counter {
		std::atomic_size_t counter;
		std::atomic_size_t finishedCount;
		Counter(
			size_t f)
			: counter(0), finishedCount(f) {}
	};
	vstd::Pool<Counter> counters;
	vstd::spin_mutex counterMtx;

public:
	//Thread Execute
	ThreadPool(size_t targetThreadCount);
	~ThreadPool();
	static bool IsWorkerThread();
	size_t WorkerThreadCount() const { return workerThreadCount; }
	ThreadTaskHandle GetTask(vstd::function<void()> func, bool waitable = false);
	ThreadTaskHandle GetFence(bool waitable = false);
	ThreadTaskHandle GetParallelTask(vstd::function<void(size_t)> func, size_t parallelCount, size_t threadCount, bool waitable = false) {
		return M_GetParallelTask(std::move(func), parallelCount, threadCount, waitable);
	}
	ThreadTaskHandle GetParallelTask(vstd::function<void(size_t)> func, size_t parallelCount, bool waitable = false) {
		return M_GetParallelTask(std::move(func), parallelCount, parallelCount, waitable);
	}

	ThreadTaskHandle GetBeginEndTask(vstd::function<void(size_t, size_t)> func, size_t parallelCount, size_t threadCount, bool waitable = false) {
		return M_GetBeginEndTask(std::move(func), parallelCount, threadCount, waitable);
	}
	ThreadTaskHandle GetBeginEndTask(vstd::function<void(size_t, size_t)> func, size_t parallelCount, bool waitable = false) {
		return M_GetBeginEndTask(std::move(func), parallelCount, parallelCount, waitable);
	}
};