#pragma once
#include <stdint.h>
namespace luisa::compute {
enum class RenderCommandType : uint32_t {
	DISPATCH_KERNEL,
	COMPILE_KERNEL,
};
class RenderCommandMethod {
public:
	virtual void Execute(void* data) = 0;
};

class RenderCommand {
public:
	RenderCommand* GetNextCommand() const {
		return reinterpret_cast<RenderCommand*>(
			(reinterpret_cast<size_t>(this) + sizeof(RenderCommand) + bufferSize));
	}
	void* GetData() const {
		return reinterpret_cast<void*>(reinterpret_cast<size_t>(this) + sizeof(RenderCommand));
	}
	RenderCommand(
		size_t bufferSize,
		RenderCommandType cmdType) : bufferSize(bufferSize), cmdType(cmdType) {
	}
	size_t GetBufferSize() const {
		return bufferSize;
	}
	RenderCommandType GetCmdType() const {
		return cmdType;
	}

private:
	size_t bufferSize;
	RenderCommandType cmdType;
};

}// namespace luisa::compute