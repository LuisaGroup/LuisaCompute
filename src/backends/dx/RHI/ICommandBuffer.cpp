#include "ICommandBuffer.h"
#include <iostream>
namespace luisa::compute {
void DirectXCommandBuffer::visit(const BufferCopyCommand* copyCmd) noexcept {
	std::cout << "Copy" << std::endl;
}
void DirectXCommandBuffer::visit(const BufferUploadCommand* upldCmd) noexcept {
	std::cout << "Upload" << std::endl;
}
void DirectXCommandBuffer::visit(const BufferDownloadCommand* dldCmd) noexcept {
	std::cout << "download" << std::endl;
}
void DirectXCommandBuffer::visit(const KernelLaunchCommand* launchCmd) noexcept {
	std::cout << "launch" << std::endl;
}
}// namespace luisa::compute