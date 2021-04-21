#pragma once
#include <runtime/command.h>
#include <Common/GFXUtil.h>
#include <Common/Runnable.h>
class ThreadCommand;
class IShader;
namespace luisa::compute {
class FrameResource;
class InternalShaders;
class DXCommandVisitor final : public CommandVisitor {
public:
	//VENGINE_CODEGEN [copy] [	void visit(## const* cmd) noexcept override;] [BufferUploadCommand] [BufferDownloadCommand] [BufferCopyCommand] [KernelLaunchCommand] [TextureUploadCommand] [TextureDownloadCommand] [EventSignalCommand] [EventWaitCommand]
	//VENGINE_CODEGEN start
	void visit(BufferUploadCommand const* cmd) noexcept override;
	void visit(BufferDownloadCommand const* cmd) noexcept override;
	void visit(BufferCopyCommand const* cmd) noexcept override;
	void visit(KernelLaunchCommand const* cmd) noexcept override;
	void visit(TextureUploadCommand const* cmd) noexcept override;
	void visit(TextureDownloadCommand const* cmd) noexcept override;
	//VENGINE_CODEGEN end

private:
	GFXDevice* device;
	ThreadCommand* tCmd;
	FrameResource* res;
	InternalShaders* internalShaders;
	Runnable<IShader*(uint)> getFunction;
};
}// namespace luisa::compute