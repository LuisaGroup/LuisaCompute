#pragma once
#include <VEngineConfig.h>
#include <initializer_list>
#include <JobSystem/JobHandle.h>
#include <Common/Pool.h>
#include <Common/TypeWiper.h>
#include <Common/Runnable.h>
#include <Common/MetaLib.h>
#include <JobSystem/JobSystem.h>
class JobSystem;
class JobThreadRunnable;
class JobNode;
class VENGINE_DLL_COMMON JobBucket {
	friend class JobSystem;
	friend class JobNode;
	friend class JobHandle;
	friend class JobThreadRunnable;

private:
	ArrayList<JobNode*> allJobNodes;
	ArrayList<JobNode*> executeJobs;
	JobSystem* sys = nullptr;
	JobHandle _GetTask( Runnable<void()>&& runnable);
	JobHandle _GetParallelTask(
		 size_t parallelCount, size_t threadCount,
		Runnable<void(size_t)>&& func);

public:
	JobHandle GetTask( Runnable<void()> runnable) {
		return _GetTask(std::move(runnable));
	}

	JobHandle GetParallelTask(
		 size_t parallelCount, size_t threadCount,
		Runnable<void(size_t)> func) {
		return _GetParallelTask(parallelCount, threadCount, std::move(func));
	}
	DECLARE_VENGINE_OVERRIDE_OPERATOR_NEW
	JobBucket(JobSystem* sys) noexcept;
	~JobBucket() noexcept {}
	
	JobHandle GetFence();
};