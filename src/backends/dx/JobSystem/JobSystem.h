#pragma once
#include <util/vstlconfig.h>
#include <util/Pool.h>
#include <util/LockFreeArrayQueue.h>
#include <atomic>
#include <stdint.h>
#include <util/MetaLib.h>
class JobNode;
class JobBucket;
class JobThreadRunnable;

class LUISA_DLL JobSystem
{
	friend class JobBucket;
	friend class JobThreadRunnable;
	friend class JobNode;
private:
	std::mutex threadMtx;
	void UpdateNewBucket();
	size_t mThreadCount;
	JobPool<JobNode> jobNodePool;
	LockFreeArrayQueue<JobNode*> executingNode;
	ArrayList<std::thread*> allThreads;
	std::atomic<int64> bucketMissionCount;
	size_t currentBucketPos;
	ArrayList<JobBucket*> buckets;
	std::condition_variable cv;
	ArrayList<JobBucket*> usedBuckets;
	ArrayList<JobBucket*> releasedBuckets;
	std::mutex mainThreadWaitMutex;
	std::condition_variable mainThreadWaitCV;
	std::atomic_bool mainThreadFinished;
	std::atomic_bool JobSystemInitialized = true;
	std::atomic_bool jobSystemStart = false;
	//void* AllocFuncMemory(uint64_t size);
	//void FreeAllMemory();
public:
	size_t GetThreadCount() const noexcept {
		return mThreadCount;
	}
	JobSystem(size_t threadCount) noexcept;
	void ExecuteBucket(JobBucket** bucket, size_t bucketCount);
	void ExecuteBucket(JobBucket* bucket, size_t bucketCount);
	void Wait();
	~JobSystem() noexcept;
	JobBucket* GetJobBucket();
	void ReleaseJobBucket(JobBucket* node);
	VSTL_OVERRIDE_OPERATOR_NEW
};

