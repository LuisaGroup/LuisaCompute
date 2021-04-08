#pragma once

#include <cstdint>
#include <functional>

namespace luisa::compute {

class RGExecutor {
public:
	virtual ~RGExecutor() = default;
	virtual uint64_t signal() = 0;
	virtual void gpu_wait(uint64_t signal, RGExecutor* signal_source) = 0;
	virtual void cpu_sync(uint64_t signal) = 0;
	virtual void execute(std::function<void()>&& func) = 0;
};

}// namespace luisa::compute
