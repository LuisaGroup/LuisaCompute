#include <render_graph/rg_system.h>
#include <render_graph/rg_executor.h>
#include <render_graph/rg_node.h>
namespace luisa::compute {
RGSystem::RGSystem() {
}
void RGSystem::execute(std::span<RGExecutor*> executors) {
	for (uint64_t i = 0; i < nonDependedJob.size(); ++i) {
		nonDependedJob[i]->execute_self(executors);
	}
	nonDependedJob.clear();
}
RGSystem::~RGSystem() {
}
}// namespace luisa::compute
