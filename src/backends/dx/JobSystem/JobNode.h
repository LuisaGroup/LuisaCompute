#pragma once
#include <util/vstlconfig.h>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <util/LockFreeArrayQueue.h>
#include <util/Pool.h>
#include <Common/TypeWiper.h>
#include <util/MetaLib.h>
#include <util/Runnable.h>
#include <span>
class JobHandle;
class JobThreadRunnable;
class JobBucket;
class VectorPool;
class JobSystem;
class LUISA_DLL JobNode
{
	friend class JobBucket;
	friend class JobSystem;
	friend class JobHandle;
	friend class JobThreadRunnable;
private:
	enum class RunnableType : uint8_t
	{
		UnAvaliable = 0,
		SingleTask = 1,
		Parallel = 2
	};
	StackObject<ArrayList<JobNode*>, true> dependingEvent;
	std::mutex* threadMtx;
	StackObject<Runnable<void()>> runnable;
	std::atomic<size_t> targetDepending;
	size_t parallelStart = 0;
	size_t parallelEnd = 0;
	size_t executeIndex;
	RunnableType runnableState = RunnableType::UnAvaliable;
	void Create(JobBucket* bucket, Runnable<void()>&& runnable, JobSystem* sys);
	void CreateParallel(JobBucket* bucket, Runnable<void()>&& runnable, size_t parallelStart, size_t parallelEnd, JobSystem* sys);
	void CreateEmpty(JobBucket* bucket, JobSystem* sys);
	void RemoveFromExecuteList(JobBucket* bucket);
	void AddDependency(JobBucket* bucket, JobHandle const& handle);
	void AddDependency(JobBucket* bucket, JobHandle const* handle, size_t handles);
	JobNode* Execute(LockFreeArrayQueue<JobNode*>& taskList, std::condition_variable& cv);

public:
	void Reset();
	void Dispose();
	~JobNode();
	VSTL_OVERRIDE_OPERATOR_NEW
};
