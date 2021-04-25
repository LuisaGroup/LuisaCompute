#include <render_graph/rg_node.h>
#include <render_graph/rg_system.h>
#include <core/logging.h>
#include <render_graph/rg_executor.h>
namespace luisa::compute {
RGNode::~RGNode() {
}
RGNode::RGNode(
	std::function<void()>&& cmd_buffer,
	uint job_type)
	: _buffer(std::move(cmd_buffer)) {
	_queue_type = job_type;
}
void RGNode::push_to_system(RGSystem* sys) {
	_rg_system = sys;
	if (_depending_job.empty()) {
		sys->nonDependedJob.push_back(this);
	}
}
void RGNode::wait() {
	if (_rg_system == nullptr) return;
	executor->cpu_sync(signal);
}
void RGNode::execute_self(std::span<RGExecutor*> executors) {
	executor = executors[_queue_type];
	//Wait
	for (auto i : _depending_job) {
		executor->gpu_wait(i->signal, i->executor);
	}
	//Execute
	_buffer();
	//Signal
	if (!_depended_job.empty()) {
		signal = executor->signal();
		for (auto i : _depended_job) {
			_rg_system->nonDependedJob.push_back(i);
		}
	}
	//Dispose
	_depended_job.clear();
	_depending_job.clear();
	_rg_system = nullptr;
}
void RGNode::add_depend_node(RGNode* node) {
	if (node->_rg_system != nullptr
		|| _rg_system != nullptr) [[unlikely]] {
		LUISA_ERROR_WITH_LOCATION("Invalid depend node");
	}
	node->_depended_job.push_back(this);
	_depending_job.push_back(node);
}
}// namespace luisa::compute
