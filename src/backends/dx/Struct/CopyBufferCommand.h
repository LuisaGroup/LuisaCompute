#pragma once
#include <Common/GFXUtil.h>
#include <RenderComponent/IBuffer.h>
struct CopyBufferCommand
{
	IBuffer* sourceBuffer;
	uint64 sourceOffset;
	IBuffer* destBuffer;
	uint64 destOffset;
	uint64 NumByts;
};