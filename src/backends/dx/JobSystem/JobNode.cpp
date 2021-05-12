#include <JobSystem/JobNode.h>
#include <JobSystem/JobSystem.h>
#include <JobSystem/JobBucket.h>
JobNode::~JobNode() {
	Dispose();
	dependingEvent.Delete();
}
JobNode* JobNode::Execute(LockFreeArrayQueue<JobNode*>& taskList, std::condition_variable& cv) {
	switch (runnableState) {
		case RunnableType::SingleTask:
			(*runnable)();
			break;
		case RunnableType::Parallel: {
			StackObject<Runnable<void(size_t)>>& parameterRunnable = reinterpret_cast<StackObject<Runnable<void(size_t)>>&>(runnable);
			for (size_t i = parallelStart; i < parallelEnd; ++i)
				(*parameterRunnable)(i);
		} break;
	}
	auto ite = dependingEvent->begin();
	JobNode* nextNode = nullptr;
	while (ite != dependingEvent->end()) {
		JobNode* node = *ite;
		size_t dependingCount = --node->targetDepending;
		if (dependingCount == 0) {
			nextNode = node;
			++ite;
			break;
		}
		++ite;
	}
	for (; ite != dependingEvent->end(); ++ite) {
		JobNode* node = *ite;
		size_t dependingCount = --node->targetDepending;
		if (dependingCount == 0) {
			taskList.Push(node);
			{
				lockGuard lck(*threadMtx);
				cv.notify_one();
			}
		}
	}
	return nextNode;
}
/*
void JobNode::Precede(JobNode* depending)
{
	depending->targetDepending++;
	dependingEvent->emplace_back(depending);
	if (depending->vecIndex >= 0)
	{
		JobBucket* bucket = depending->bucket;
		auto lastIte = bucket->jobNodesVec.end() - 1;
		auto indIte = bucket->jobNodesVec.begin() + depending->vecIndex;
		*indIte = *lastIte;
		(*indIte)->vecIndex = depending->vecIndex;
		bucket->jobNodesVec.erase(lastIte);
		depending->vecIndex = -1;
	}
}
*/

void JobNode::Create(JobBucket* bucket, Runnable<void()>&& runnable, JobSystem* sys) {
	std::mutex* threadMtx = &sys->threadMtx;
	targetDepending = 0;
	this->threadMtx = threadMtx;
	this->runnable.New(std::move(runnable));
	runnableState = RunnableType::SingleTask;
	executeIndex = bucket->executeJobs.size();
	bucket->executeJobs.push_back(this);
}
void JobNode::CreateParallel(JobBucket* bucket, Runnable<void()>&& runnable, size_t parallelStart, size_t parallelEnd, JobSystem* sys) {
	std::mutex* threadMtx = &sys->threadMtx;
	targetDepending = 0;
	this->threadMtx = threadMtx;
	this->runnable.New(std::move(runnable));
	runnableState = RunnableType::Parallel;
	this->parallelStart = parallelStart;
	this->parallelEnd = parallelEnd;
	executeIndex = bucket->executeJobs.size();
	bucket->executeJobs.push_back(this);
}
void JobNode::CreateEmpty(JobBucket* bucket, JobSystem* sys) {
	std::mutex* threadMtx = &sys->threadMtx;
	targetDepending = 0;
	this->threadMtx = threadMtx;
	runnableState = RunnableType::UnAvaliable;
	executeIndex = bucket->executeJobs.size();
	bucket->executeJobs.push_back(this);
}
void JobNode::AddDependency(JobBucket* bucket, JobHandle const* handles, size_t handlesCount) {
	auto endHandle = handles + handlesCount;
	size_t lastDep = targetDepending;
	for (auto hPtr = handles; hPtr != endHandle; ++hPtr) {
		auto&& handle = *hPtr;
		if (handle.bucket != bucket) continue;
		targetDepending += handle.Count();
		auto startIte = bucket->allJobNodes.begin() + handle.start;
		auto endIte = bucket->allJobNodes.begin() + handle.end;
		for (auto ite = startIte;; ite++) {
			(*ite)->dependingEvent->push_back(this);
			if (ite == endIte)
				break;
		}
	}
	if (lastDep == 0 && targetDepending != 0)
		RemoveFromExecuteList(bucket);
}
void JobNode::RemoveFromExecuteList(JobBucket* bucket) {
	auto&& ptr = bucket->executeJobs[executeIndex];
	auto ite = bucket->executeJobs.end() - 1;
	ptr = *ite;
	ptr->executeIndex = executeIndex;
	bucket->executeJobs.erase(ite);
}
void JobNode::AddDependency(JobBucket* bucket, JobHandle const& handle) {
	if (handle.bucket != bucket) return;
	if (targetDepending == 0)
		RemoveFromExecuteList(bucket);
	targetDepending += handle.Count();
	auto startIte = bucket->allJobNodes.begin() + handle.start;
	auto endIte = bucket->allJobNodes.begin() + handle.end;
	for (auto ite = startIte;; ite++) {
		(*ite)->dependingEvent->push_back(this);
		if (ite == endIte)
			break;
	}
}
void JobHandle::AddDependency(JobHandle const& handle) {
	auto startIte = bucket->allJobNodes.begin() + start;
	auto endIte = bucket->allJobNodes.begin() + end;
	for (auto ite = startIte;; ite++) {
		(*ite)->AddDependency(bucket, handle);
		if (ite == endIte) break;
	}
}
void JobHandle::AddDependency(std::initializer_list<JobHandle const> handle) {
	auto startIte = bucket->allJobNodes.begin() + start;
	auto endIte = bucket->allJobNodes.begin() + end;
	for (auto ite = startIte;; ite++) {
		(*ite)->AddDependency(bucket, handle.begin(), handle.size());
		if (ite == endIte) break;
	}
}
void JobHandle::AddDependency(JobHandle const* handles, size_t handleCount) {
	auto startIte = bucket->allJobNodes.begin() + start;
	auto endIte = bucket->allJobNodes.begin() + end;
	for (auto ite = startIte;; ite++) {
		(*ite)->AddDependency(bucket, handles, handleCount);
		if (ite == endIte) break;
	}
}
void JobNode::Reset() {
	dependingEvent.New();
}
void JobNode::Dispose() {
	switch (runnableState) {
		case RunnableType::SingleTask:
			runnable.Delete();
			break;
		case RunnableType::Parallel: {
			StackObject<Runnable<void(size_t)>>& parameterRunnable = reinterpret_cast<StackObject<Runnable<void(size_t)>>&>(runnable);
			parameterRunnable.Delete();
		} break;
	}
	runnableState = RunnableType::UnAvaliable;
	if (dependingEvent.Initialized()) {
		dependingEvent->clear();
	}
}
