#include <JobSystem/JobGraph.h>

void RuntimeType::Dispose() {
	if (disposer && ptr) {
		disposer(ptr);
	}
	ptr = nullptr;
}
RuntimeType::~RuntimeType() {
	Dispose();
}

GraphNodeData* GraphGlobalData::GetNode() {
	GraphNodeData* newNode = pool.New(this);
	data.push_back(newNode);
	return newNode;
}
GraphGlobalData::GraphGlobalData(
	size_t nodeProbablySize) : pool(nodeProbablySize) {}
GraphGlobalData ::~GraphGlobalData() {
	for (auto i : data) {
		pool.Delete(i);
	}
}
void GraphGlobalData::DisposeCompile() {
	runtimeType.Clear();
	for (auto i : data) {
		pool.Delete(i);
	}
	data.clear();
}

GraphNodeData ::~GraphNodeData() {
}
void GraphNodeData::Execute() const {
	retValue->getJobHandle();
}