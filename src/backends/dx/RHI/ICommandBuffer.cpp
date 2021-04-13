#include <RHI/ICommandBuffer.h>
#include <iostream>
#include <PipelineComponent/ThreadCommand.h>
#include <RenderComponent/RenderComponentInclude.h>
#include <PipelineComponent/DXAllocator.h>
#include <PipelineComponent/FrameResource.h>
#include <Singleton/Graphics.h>
namespace luisa::compute {
//VENGINE_CODEGEN [copy] [	void DXCommandVisitor::visit(## const* cmd) noexcept {}] [BufferUploadCommand] [BufferDownloadCommand] [BufferCopyCommand] [KernelLaunchCommand] [TextureUploadCommand] [TextureDownloadCommand] [EventSignalCommand] [EventWaitCommand]
//VENGINE_CODEGEN start
void DXCommandVisitor::visit(BufferUploadCommand const* cmd) noexcept {
	UploadBuffer* middleBuffer = new UploadBuffer(device, cmd->size(), false, 1, DXAllocator::GetBufferAllocator());
	StructuredBuffer* destBuffer = reinterpret_cast<StructuredBuffer*>(cmd->handle());
	res->deferredDeleteObj.emplace_back(std::move(MakeObjectPtr(middleBuffer)).CastTo<VObject>());
	middleBuffer->CopyDatas(0, cmd->size(), cmd->data());
	tCmd->UpdateResState(
		destBuffer->GetInitState(),
		GPUResourceState_CopyDest,
		destBuffer,
		true);
	Graphics::CopyBufferRegion(
		tCmd,
		destBuffer,
		cmd->offset(),
		middleBuffer,
		0,
		cmd->size());
}
void DXCommandVisitor::visit(BufferDownloadCommand const* cmd) noexcept {
	ReadbackBuffer* readBuffer = new ReadbackBuffer(device, cmd->size(), 1);
	StructuredBuffer* sbuffer = reinterpret_cast<StructuredBuffer*>(cmd->handle());
	Graphics::CopyBufferRegion(
		tCmd,
		readBuffer,
		0,
		sbuffer,
		cmd->offset(),
		cmd->size());
	res->afterSyncTask.emplace_back(std::move(Runnable<void()>(
		[=]() {
			readBuffer->Map();
			memcpy(cmd->data(), readBuffer->GetMappedPtr(0), cmd->size());
			readBuffer->UnMap();
			delete readBuffer;
		})));
}
void DXCommandVisitor::visit(BufferCopyCommand const* cmd) noexcept {
	StructuredBuffer* srcBuffer = reinterpret_cast<StructuredBuffer*>(cmd->src_handle());
	StructuredBuffer* destBuffer = reinterpret_cast<StructuredBuffer*>(cmd->dst_handle());
	Graphics::CopyBufferRegion(
		tCmd,
		destBuffer,
		cmd->dst_offset(),
		srcBuffer,
		cmd->src_offset(),
		cmd->size());
}
void DXCommandVisitor::visit(KernelLaunchCommand const* cmd) noexcept {
}
void DXCommandVisitor::visit(TextureUploadCommand const* cmd) noexcept {
}
void DXCommandVisitor::visit(TextureDownloadCommand const* cmd) noexcept {}
void DXCommandVisitor::visit(EventSignalCommand const* cmd) noexcept {}
void DXCommandVisitor::visit(EventWaitCommand const* cmd) noexcept {}
//VENGINE_CODEGEN end
}// namespace luisa::compute