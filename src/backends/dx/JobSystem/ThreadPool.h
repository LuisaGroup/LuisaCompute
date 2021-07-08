#pragma once
#include <Common/Common.h>
#include <Common/VObject.h>
#include <Common/LockFreeArrayQueue.h>
#include <JobSystem/ThreadTaskHandle.h>
class VENGINE_DLL_COMMON ThreadPool final {
	friend class ThreadTaskHandle;
	class ThreadPoolMethod {

		void operator()() {
		}
	};
	std::atomic_uint64_t executeCount = 1;
	using PoolType = typename ThreadTaskHandle::PoolType;
	ObjectPtr<PoolType> pool;
	vstd::vector<std::thread> threads;
	LockFreeArrayQueue<ObjectPtr<ThreadTaskHandle::TaskData>> taskList;
	void ThreadExecute();
	std::atomic_flag enabled;
	std::mutex threadLock;
	std::mutex backupThreadLock;
	std::condition_variable cv;
	std::condition_variable backupCV;
	void ExecuteTask(ObjectPtr<ThreadTaskHandle::TaskData> const& task);
	size_t workerThreadCount;
	size_t backupThreadCount;
	void ActiveOneBackupThread();
public:
	//Thread Execute
	ThreadPool(size_t targetThreadCount, size_t backupThreadCount);
	~ThreadPool();
	ThreadTaskHandle GetTask(Runnable<void()> func);
	ThreadTaskHandle GetParallelTask(Runnable<void(size_t)> func, size_t parallelCount, size_t threadCount);
};