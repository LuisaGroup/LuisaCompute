#pragma once
#include "../Common/GFXUtil.h"
#include "../Common/vector.h"
class CommandBuffer
{
private:
	ArrayList<GFXCommandList*> graphicsCmdLists;
	struct Fence
	{
		ID3D12Fence* fence;
		UINT64 frameIndex;
	};
	struct InnerCommand
	{
		enum CommandType
		{
			CommandType_Execute,
			CommandType_Signal,
			CommandType_Wait,
		};
		CommandType type;
		GFXCommandQueue* targetQueue;
		union
		{
			GFXCommandList* executeCmdList;
			Fence waitFence;
			Fence signalFence;
		};
		InnerCommand() {}
		InnerCommand(const InnerCommand& cmd)
		{
			memcpy(this, &cmd, sizeof(InnerCommand));
		}
		void operator=(const InnerCommand& cmd)
		{
			memcpy(this, &cmd, sizeof(InnerCommand));
		}
	};
	ArrayList<InnerCommand> executeCommands;
public:
	void Wait(GFXCommandQueue* queue, ID3D12Fence* computeFence, UINT64 currentFrame);
	void Signal(GFXCommandQueue* queue, ID3D12Fence* computeFence, UINT64 currentFrame);
	void Execute(GFXCommandQueue* queue, GFXCommandList* cmdList);
	void Submit();
	void Clear();
	CommandBuffer();
};