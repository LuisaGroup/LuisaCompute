#pragma once
#include <stdint.h>
#include <vector>
#include <render_graph/rg_enum.h>
#include <span>
namespace luisa::compute {
class RGNode;
class RGExecutor;
class RGSystem {
	friend class RGNode;

public:
	RGSystem();
	void execute(std::span<RGExecutor*> executors);
	~RGSystem();

private:
	RGNodeState _state = RGNodeState::Preparing;
	std::vector<RGNode*> nonDependedJob;
};
}// namespace luisa::compute