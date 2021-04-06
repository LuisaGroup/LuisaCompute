#pragma once
#include <VEngineConfig.h>
#include <initializer_list>
#include "JobHandle.h"
#include <Common/Pool.h>
#include <Common/TypeWiper.h>
#include <Common/Runnable.h>
#include <Common/MetaLib.h>
#include "JobSystem.h"
class JobSystem;
class JobThreadRunnable;
class JobNode;
typedef uint32_t uint;
class VENGINE_DLL_COMMON JobBucket {
	friend class JobSystem;
	friend class JobNode;
	friend class JobHandle;
	friend class JobThreadRunnable;

private:
	ArrayList<JobNode*> jobNodesVec;
	ArrayList<JobNode*> allJobNodes;
	JobSystem* sys = nullptr;
	JobHandle _GetTask(JobHandle const* dependedJobs, uint32_t dependCount, Runnable<void()>&& runnable);
	JobHandle _GetParallelTask(
		JobHandle const* dependedJobs, uint32_t dependCount, uint parallelCount, uint threadCount,
		Runnable<void(uint)>&& func);

public:
	JobHandle GetTask(JobHandle const* dependedJobs, uint32_t dependCount, Runnable<void()> runnable) {
		return _GetTask(dependedJobs, dependCount, std::move(runnable));
	}
	JobHandle GetTask(std::initializer_list<JobHandle> handle, Runnable<void()> runnable) {
		return _GetTask(handle.begin(), handle.size(), std::move(runnable));
	}
	JobHandle GetParallelTask(
		JobHandle const* dependedJobs, uint32_t dependCount, uint parallelCount, uint threadCount,
		Runnable<void(uint)> func) {
		return _GetParallelTask(dependedJobs, dependCount, parallelCount, threadCount, std::move(func));
	}
	JobHandle GetParallelTask(
		std::initializer_list<JobHandle> handle, uint parallelCount, uint threadCount,
		Runnable<void(uint)> func) {
		return _GetParallelTask(handle.begin(), handle.size(), parallelCount, threadCount, std::move(func));
	}
	DECLARE_VENGINE_OVERRIDE_OPERATOR_NEW
	JobBucket(JobSystem* sys) noexcept;
	~JobBucket() noexcept {}
	
	JobHandle GetFence(JobHandle const* dependedJobs, uint32_t dependCount);
};