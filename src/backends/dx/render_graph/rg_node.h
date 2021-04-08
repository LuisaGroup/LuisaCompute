#pragma once
#include <vector>
#include <type_traits>
#include <functional>
#include <stdint.h>
#include "rg_enum.h"
#include <span>
#include <initializer_list>
#include <runtime/command_buffer.h>
namespace luisa::compute {
class RGSystem;
class RGExecutor;
class RGNode {
	friend class RGSystem;

public:
	~RGNode();
	void add_depend_node(RGNode* node);
	void add_depend_nodes(std::span<RGNode*> nodes) {
		for (auto i : nodes) {
			add_depend_node(i);
		}
	}
	void add_depend_nodes(std::initializer_list<RGNode*> nodes) {
		for (auto i : nodes) {
			add_depend_node(i);
		}
	}
	RGNode(
		std::function<void()>&& cmd_buffer,
		uint job_type);

	void push_to_system(RGSystem* sys);
	void wait();

private:
	void execute_self(std::span<RGExecutor*> executors);
	RGExecutor* executor;
	uint _queue_type;
	RGSystem* _rg_system = nullptr;
	std::vector<RGNode*> _depending_job;
	uint64_t signal = 0;
	std::vector<RGNode*> _depended_job;
	std::function<void()> _buffer;
};
}// namespace luisa::compute
