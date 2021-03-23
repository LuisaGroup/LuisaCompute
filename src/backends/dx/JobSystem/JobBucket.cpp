#include "JobBucket.h"
#include "JobNode.h"
#include "JobSystem.h"
JobBucket::JobBucket(JobSystem* sys) noexcept : sys(sys) {
	jobNodesVec.reserve(20);
}
JobHandle JobBucket::_GetTask(JobHandle const* dependedJobs, uint32_t dependCount, Runnable<void()>&& runnable) {
	JobNode* node = sys->jobNodePool.New();
	node->Create(this, std::move(runnable), sys, dependedJobs, dependCount);
	if (dependCount == 0) jobNodesVec.push_back(node);
	uint value = allJobNodes.size();
	allJobNodes.push_back(node);
	return JobHandle(value, value);
}
JobHandle JobBucket::_GetParallelTask(
	JobHandle const* dependedJobs, uint32_t dependCount, uint parallelCount, uint threadCount,
	Runnable<void(uint)>&& runnable) {
	auto ParallelTask = [&](Runnable<void()> copyedFunc, uint parallelStart, uint parallelEnd) {
		JobNode* node = sys->jobNodePool.New();
		node->CreateParallel(this, std::move(copyedFunc), parallelStart, parallelEnd, sys, dependedJobs, dependCount);
		if (dependCount == 0) jobNodesVec.push_back(node);
		allJobNodes.push_back(node);
	};
	if (threadCount > sys->GetThreadCount())
		threadCount = sys->GetThreadCount();
	uint eachJobCount = parallelCount / threadCount;
	JobHandle handle;
	for (uint i = 0; i < threadCount; ++i) {
		ParallelTask(
			reinterpret_cast<Runnable<void()>&>(runnable), i * eachJobCount, (i + 1) * eachJobCount);
	}
	uint count = threadCount;
	uint full = eachJobCount * threadCount;
	uint lefted = parallelCount - full;
	if (lefted > 0) {
		ParallelTask(
			reinterpret_cast<Runnable<void()>&>(runnable), full, parallelCount);
		count++;
	}
	return JobHandle(allJobNodes.size() - count, allJobNodes.size() - 1);
}
JobHandle JobBucket::GetFence(JobHandle const* dependedJobs, uint32_t dependCount) {
	JobNode* node = sys->jobNodePool.New();
	node->CreateEmpty(this, sys, dependedJobs, dependCount);
	if (dependCount == 0) jobNodesVec.push_back(node);
	uint value = allJobNodes.size();
	allJobNodes.push_back(node);
	return JobHandle(value, value);
}
