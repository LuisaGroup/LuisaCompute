#pragma once

#include <cstdint>
#include <vector>
#include "rg_enum.h"
#include <span>

namespace luisa::compute {

class RGNode;
class RGExecutor;

class RGSystem {

    friend class RGNode;

private:
    RGNodeState _state = RGNodeState::Preparing;
    std::vector<RGNode *> nonDependedJob;

public:
    RGSystem();
    ~RGSystem();

    void execute(std::span<RGExecutor *> executors);
};

}// namespace luisa::compute
