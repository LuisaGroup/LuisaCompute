#include <JobSystem/JobBucket.h>
#include <JobSystem/JobNode.h>
#include <JobSystem/JobSystem.h>
JobBucket::JobBucket(JobSystem* sys) noexcept : sys(sys) {
}
JobHandle JobBucket::_GetTask( Runnable<void()>&& runnable) {
	JobNode* node = sys->jobNodePool.New();
	node->Create(this, std::move(runnable), sys);
	size_t value = allJobNodes.size();
	allJobNodes.push_back(node);
	return JobHandle(this, value, value);
}
JobHandle JobBucket::_GetParallelTask(
	 size_t parallelCount, size_t threadCount,
	Runnable<void(size_t)>&& runnable) {
	auto ParallelTask = [&](Runnable<void()> copyedFunc, size_t parallelStart, size_t parallelEnd) {
		JobNode* node = sys->jobNodePool.New();
		node->CreateParallel(this, std::move(copyedFunc), parallelStart, parallelEnd, sys);
		allJobNodes.push_back(node);
	};
	if (threadCount > sys->GetThreadCount())
		threadCount = sys->GetThreadCount();
	size_t eachJobCount = parallelCount / threadCount;
	JobHandle handle;
	for (size_t i = 0; i < threadCount; ++i) {
		ParallelTask(
			reinterpret_cast<Runnable<void()>&>(runnable), i * eachJobCount, (i + 1) * eachJobCount);
	}
	size_t count = threadCount;
	size_t full = eachJobCount * threadCount;
	size_t lefted = parallelCount - full;
	if (lefted > 0) {
		ParallelTask(
			reinterpret_cast<Runnable<void()>&>(runnable), full, parallelCount);
		count++;
	}
	return JobHandle(this, allJobNodes.size() - count, allJobNodes.size() - 1);
}
JobHandle JobBucket::GetFence() {
	JobNode* node = sys->jobNodePool.New();
	node->CreateEmpty(this, sys);
	size_t value = allJobNodes.size();
	allJobNodes.push_back(node);
	return JobHandle(this, value, value);
}

