#include <RHI/DXCommandVisitor.h>
#include <ast/function.h>
#include <iostream>
#include <PipelineComponent/ThreadCommand.h>
#include <RenderComponent/RenderComponentInclude.h>
#include <PipelineComponent/DXAllocator.h>
#include <PipelineComponent/FrameResource.h>
#include <Singleton/Graphics.h>
#include <RHI/InternalShaders.h>
#include <RHI/RenderTexturePackage.h>
#include <Struct/ConstBuffer.h>
#include <Singleton/ShaderID.h>
#include <RHI/ShaderCompiler.h>

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
	ReadbackBuffer* readBuffer = new ReadbackBuffer(device, cmd->size(), 1, DXAllocator::GetBufferAllocator());
	StructuredBuffer* sbuffer = reinterpret_cast<StructuredBuffer*>(cmd->handle());
	tCmd->UpdateResState(
		sbuffer->GetInitState(),
		GPUResourceState_CopySource,
		sbuffer,
		true);
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
	tCmd->UpdateResState(
		srcBuffer->GetInitState(),
		GPUResourceState_CopySource,
		srcBuffer,
		true);
	tCmd->UpdateResState(
		destBuffer->GetInitState(),
		GPUResourceState_CopyDest,
		destBuffer,
		true);
	Graphics::CopyBufferRegion(
		tCmd,
		destBuffer,
		cmd->dst_offset(),
		srcBuffer,
		cmd->src_offset(),
		cmd->size());
}
void DXCommandVisitor::visit(KernelLaunchCommand const* cmd) noexcept {
	IShader* funcShader = getFunction(cmd->kernel_uid());
	if (funcShader->GetType() == typeid(ComputeShader)) {//Common compute
		ComputeShader* cs = static_cast<ComputeShader*>(funcShader);
		auto&& cbData = ShaderCompiler::GetCBufferData(cmd->kernel_uid());
		vengine::vector<uint8_t> cbufferData(cbData.cbufferSize);
		memset(cbufferData.data(), 0, cbufferData.size());
		cs->BindShader(tCmd);
		struct Functor {
			ComputeShader* cs;
			Function func;
			ThreadCommand* tCmd;
			uint8_t* cbuffer;
			ShaderCompiler::ConstBufferData const* cbData;
			void operator()(uint varID, KernelLaunchCommand::BufferArgument const& arg) {
				StructuredBuffer* buffer = reinterpret_cast<StructuredBuffer*>(arg.handle);
				bool isUAV = ((uint)func.variable_usage(varID) & (uint)Variable::Usage::WRITE) != 0;
				if (isUAV) {
					tCmd->UpdateResState(
						buffer->GetInitState(),
						GPUResourceState_UnorderedAccess,
						buffer, true);
					tCmd->UAVBarrier(buffer);
				} else {
					tCmd->UpdateResState(
						buffer->GetInitState(),
						GPUResourceState_NonPixelShaderRes,
						buffer, true);
				}

				cs->SetResource(
					tCmd,
					ShaderID::PropertyToID(varID),
					buffer,
					arg.offset);
			}
			void operator()(uint varID, KernelLaunchCommand::TextureArgument const& arg) {
				RenderTexture* tex = reinterpret_cast<RenderTexture*>(arg.handle);
				bool isUAV = ((uint)func.variable_usage(varID) & (uint)Variable::Usage::WRITE) != 0;
				if (isUAV) {
					tCmd->UpdateResState(
						tex->GetInitState(),
						GPUResourceState_UnorderedAccess,
						tex, true);
					tCmd->UAVBarrier(tex);
					cs->SetResource(
						tCmd,
						ShaderID::PropertyToID(varID),
						tex, 0);//TODO: uav mip level

				} else {
					tCmd->UpdateResState(
						tex->GetInitState(),
						GPUResourceState_NonPixelShaderRes,
						tex, true);
					cs->SetResource(
						tCmd,
						ShaderID::PropertyToID(varID),
						tex);
				}
			}
			void operator()(uint varID, std::span<const std::byte> const& arg) {
				auto ite = cbData->offsets.Find(varID);
				if (!ite) return;
				memcpy(cbuffer + ite.Value(), arg.data(), arg.size_bytes());
			}
		};
		Functor f{
			cs,
			Function::kernel(cmd->kernel_uid()),
			tCmd,
			cbufferData.data(),
			&cbData
		};
		cmd->decode(f);
		auto cbufferChunk = res->AllocateCBuffer(cbufferData.size());
		cbufferChunk.CopyData(cbufferData.data(), cbufferData.size());

		auto launchSize = cmd->launch_size();
		cs->Dispatch(
			tCmd,
			0,
			launchSize.x,
			launchSize.y,
			launchSize.z);
	} else {//Other Type Shaders
	}
}
void DXCommandVisitor::visit(TextureUploadCommand const* cmd) noexcept {
	struct Params {
		uint4 _Resolution;
		uint3 _PixelOffset;
	};
	Params param;
	uint3 resolution = cmd->size();
	param._Resolution = uint4(resolution.x, resolution.y, resolution.z, 1);
	param._PixelOffset = cmd->offset();
	CBufferChunk chunk = res->AllocateCBuffer(sizeof(param));
	chunk.CopyData(&param);

	uint64 sz = (uint64)resolution.x * (uint64)resolution.y * (uint64)resolution.z;
	UploadBuffer* middleBuffer = new UploadBuffer(
		device,
		sz * pixel_storage_size(cmd->storage()),
		false, 1,
		DXAllocator::GetBufferAllocator());
	res->deferredDeleteObj.emplace_back(MakeObjectPtr(middleBuffer).CastTo<VObject>());
	RenderTexturePackage* rt = reinterpret_cast<RenderTexturePackage*>(cmd->handle());
	bool dim = rt->rt->GetDimension() == TextureDimension::Tex2D;
	uint kernel = pixel_format_count * (dim ? 0 : 1) + (uint)(rt->format);
	uint3 dispKernel;

	auto cs = internalShaders->copyShader;
	middleBuffer->CopyDatas(
		0, middleBuffer->GetElementCount(),
		cmd->data());
	tCmd->UpdateResState(
		rt->rt->GetInitState(),
		GPUResourceState_UnorderedAccess,
		rt->rt,
		true);
	tCmd->UAVBarrier(rt->rt);
	cs->BindShader(tCmd);
	cs->SetResource(tCmd, internalShaders->_Buffer, middleBuffer, 0);
	if (dim) {
		dispKernel = uint3((resolution.x + 7) / 8, (resolution.y + 7) / 8, 1);
		cs->SetResource(tCmd, internalShaders->_Tex2D, rt->rt);
	} else {
		dispKernel = (resolution + 3u) / 4u;
		cs->SetResource(tCmd, internalShaders->_Tex3D, rt->rt);
	}
	cs->Dispatch(
		tCmd,
		kernel,
		dispKernel.x,
		dispKernel.y,
		dispKernel.z);
}
void DXCommandVisitor::visit(TextureDownloadCommand const* cmd) noexcept {
	struct Params {
		uint4 _Resolution;
		uint3 _PixelOffset;
	};
	Params param;
	uint3 resolution = cmd->size();
	param._Resolution = uint4(resolution.x, resolution.y, resolution.z, 1);
	param._PixelOffset = cmd->offset();
	CBufferChunk chunk = res->AllocateCBuffer(sizeof(param));
	chunk.CopyData(&param);
	uint64 sz = (uint64)resolution.x * (uint64)resolution.y * (uint64)resolution.z;
	uint64 byteSize = sz * pixel_storage_size(cmd->storage());
	ReadbackBuffer* readBuffer = new ReadbackBuffer(
		device,
		byteSize,
		1,
		DXAllocator::GetBufferAllocator());
	StructuredBuffer* middleBuffer = new StructuredBuffer(
		device,
		{StructuredBufferElement::Get(1, byteSize)},
		GPUResourceState_UnorderedAccess,
		DXAllocator::GetBufferAllocator());
	res->deferredDeleteObj.emplace_back(MakeObjectPtr(middleBuffer).CastTo<VObject>());

	RenderTexturePackage* rt = reinterpret_cast<RenderTexturePackage*>(cmd->handle());
	bool dim = rt->rt->GetDimension() == TextureDimension::Tex2D;
	uint kernel = pixel_format_count * (dim ? 2 : 3) + (uint)(rt->format);
	uint3 dispKernel;

	tCmd->UpdateResState(
		rt->rt->GetInitState(),
		GPUResourceState_NonPixelShaderRes,
		rt->rt,
		true);
	auto cs = internalShaders->copyShader;
	cs->BindShader(tCmd);
	if (dim) {
		dispKernel = uint3((resolution.x + 7) / 8, (resolution.y + 7) / 8, 1);
		cs->SetResource(tCmd, internalShaders->_Read_Tex2D, rt->rt);
	} else {
		dispKernel = (resolution + 3u) / 4u;
		cs->SetResource(tCmd, internalShaders->_Read_Tex3D, rt->rt);
	}
	cs->SetResource(tCmd, internalShaders->_Write_Buffer, middleBuffer, 0);
	cs->Dispatch(
		tCmd,
		kernel,
		dispKernel.x,
		dispKernel.y,
		dispKernel.z);
	tCmd->UpdateResState(
		middleBuffer->GetInitState(),
		GPUResourceState_CopySource,
		middleBuffer);
	Graphics::CopyBufferRegion(
		tCmd,
		readBuffer,
		0,
		middleBuffer,
		0,
		byteSize);

	res->afterSyncTask.emplace_back(std::move(Runnable<void()>(
		[=]() {
			readBuffer->Map();
			memcpy(cmd->data(), readBuffer->GetMappedPtr(0), byteSize);
			readBuffer->UnMap();
			delete readBuffer;
		})));
}
//VENGINE_CODEGEN end
}// namespace luisa::compute
