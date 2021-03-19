#pragma once
#include <vector>
#include <span>
#include "RenderCommand.h"
namespace luisa::compute {

class CommandBuffer {
public:
	void ClearCommand() {
		datas.clear();
		commandCount = 0;
	}
	void AddCommand(
		RenderCommand const& command,
		void const* data) {
		size_t offset = datas.size();
		datas.resize(offset + sizeof(RenderCommand) + command.GetBufferSize());
		*reinterpret_cast<RenderCommand*>(datas.data() + offset) = command;
		memcpy(datas.data() + offset + sizeof(RenderCommand), data, command.GetBufferSize());
		commandCount++;
	}

	void ExecuteCommands(std::span<RenderCommandMethod*> methods) {
		RenderCommand* command = reinterpret_cast<RenderCommand*>(datas.data());
		for (size_t i = 0; i < commandCount; ++i) {
			methods[(uint32_t)command->GetCmdType()]->Execute(command->GetData());
			command = command->GetNextCommand();
		}
	}

private:
	std::vector<uint8_t> datas;
	size_t commandCount = 0;
};
}// namespace luisa::compute