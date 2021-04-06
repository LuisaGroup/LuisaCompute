#pragma once
#include <VEngineConfig.h>
#include "../Common/Pool.h"
#include "../Common/LockFreeArrayQueue.h"
#include <atomic>
#include <stdint.h>
#include "../Common/MetaLib.h"
class JobNode;
class JobBucket;
class JobThreadRunnable;

class VENGINE_DLL_COMMON JobSystem
{
	friend class JobBucket;
	friend class JobThreadRunnable;
	friend class JobNode;
private:
	std::mutex threadMtx;
	void UpdateNewBucket();
	uint32_t mThreadCount;
	JobPool<JobNode> jobNodePool;
	LockFreeArrayQueue<JobNode*> executingNode;
	ArrayList<std::thread*> allThreads;
	std::atomic<int32_t> bucketMissionCount;
	uint32_t currentBucketPos;
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
	uint32_t GetThreadCount() const noexcept {
		return mThreadCount;
	}
	JobSystem(uint32_t threadCount) noexcept;
	void ExecuteBucket(JobBucket** bucket, uint32_t bucketCount);
	void ExecuteBucket(JobBucket* bucket, uint32_t bucketCount);
	void Wait();
	~JobSystem() noexcept;
	JobBucket* GetJobBucket();
	void ReleaseJobBucket(JobBucket* node);
	DECLARE_VENGINE_OVERRIDE_OPERATOR_NEW
};

