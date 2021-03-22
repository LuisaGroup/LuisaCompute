#pragma once
#include <Common/DLL.h>
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
class DLL_COMMON JobBucket {
	friend class JobSystem;
	friend class JobNode;
	friend class JobHandle;
	friend class JobThreadRunnable;

private:
	ArrayList<JobNode*> jobNodesVec;
	ArrayList<JobNode*> allJobNodes;
	JobSystem* sys = nullptr;
	void Split_GetParallelTask(JobHandle const* dependedJobs, uint32_t dependCount, Runnable<void()> runnable, uint parallelStart, uint parallelEnd);

public:
	DECLARE_VENGINE_OVERRIDE_OPERATOR_NEW
	JobHandle GetTask(JobHandle const* dependedJobs, uint32_t dependCount, Runnable<void()> runnable);
	JobHandle GetTask(std::initializer_list<JobHandle> handle, Runnable<void()> runnable) {
		return GetTask(handle.begin(), handle.size(), std::move(runnable));
	}
	JobBucket(JobSystem* sys) noexcept;
	~JobBucket() noexcept {}
	JobHandle GetParallelTask(
		JobHandle const* dependedJobs, uint32_t dependCount, uint parallelCount, uint threadCount,
		Runnable<void(uint)> func);
	JobHandle GetParallelTask(
		std::initializer_list<JobHandle> handle, uint parallelCount, uint threadCount,
		Runnable<void(uint)> func) {
		return GetParallelTask(handle.begin(), handle.size(), parallelCount, threadCount, std::move(func));
	}
	JobHandle GetFence(JobHandle const* dependedJobs, uint32_t dependCount);
};