#pragma once
#include <runtime/command.h>
namespace luisa::compute {
class DirectXCommandBuffer : public CommandVisitor {
public:
	void visit(const BufferCopyCommand*) noexcept override;
	void visit(const BufferUploadCommand*) noexcept override;
	void visit(const BufferDownloadCommand*) noexcept override;
	void visit(const KernelLaunchCommand*) noexcept override;
};
}// namespace luisa::compute