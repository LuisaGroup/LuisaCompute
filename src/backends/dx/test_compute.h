#pragma once
#include <DllManager.h>
#include <Shader/ComputeShader.h>
#include <Shader/RTShader.h>
#include <Graphics/ShaderCompiler/ShaderCompiler.h>
#include <DXRuntime/Device.h>
#include <DXRuntime/CommandQueue.h>
#include <DXRuntime/CommandBuffer.h>
#include <DXRuntime/CommandAllocator.h>
#include <Resource/DefaultBuffer.h>
#include <Resource/ReadbackBuffer.h>
#include <Resource/UploadBuffer.h>
#include <Resource/RenderTexture.h>
#include <DXRuntime/ResourceStateTracker.h>
#include <Resource/BindlessArray.h>

vstd::vector<vbyte> GetShader(
	vstd::string code,
	toolhub::db::IJsonDatabase* db) {
	using namespace toolhub;
	using namespace toolhub::directx;

	auto backCompute = R"(
void main(uint3 thdId : SV_GroupThreadId, uint3 dspId : SV_DispatchThreadID, uint3 grpId : SV_GroupId){
)"_sv;
	auto gra = DllManager::GetGraphics();
	auto compiler = vstd::create_unique(gra->CreateDirectXCompiler());
	auto inlineProperty = compiler->GetHLSLInlineProperty(code, db);
	auto preprocess = compiler->PreProcessHLSL(std::move(code));
	auto blkArray = inlineProperty->Get("Dispatch").get_or<db::IJsonArray*>(nullptr);
	uint3 blockSize = inlineProperty ? ((blkArray && blkArray->Length() >= 3)
											? uint3(
												blkArray->Get(0).get_or<int64>(1),
												blkArray->Get(1).get_or<int64>(1),
												blkArray->Get(2).get_or<int64>(1))
											: uint3(1, 1, 1))
									 : uint3(1, 1, 1);
	auto entry = inlineProperty->Get("Entry").get_or<vstd::string_view>("run"_sv);
	preprocess.resultCode
		<< "[numthreads("_sv
		<< vstd::to_string(blockSize.x) << ','
		<< vstd::to_string(blockSize.y) << ','
		<< vstd::to_string(blockSize.z)
		<< ")]\n"
		<< backCompute
		<< entry
		<< "(thdId, dspId, grpId);}"_sv;
	auto result = compiler->CompileCompute(
		preprocess.resultCode,
		true);
	if (!result.multi_visit_or(
			false,
			[&](vstd::unique_ptr<graphics::IByteBlob> const& b) {
				std::cout << "DXIL size: "_sv << b->GetBufferSize() << '\n';
				return true;
			},
			[&](vstd::string const& b) {
				std::cout << b << '\n';
				return false;
			})) return 1;
	auto res = vstd::create_unique(
		static_cast<graphics::IComputeShader*>(
			gra->CreateResource(graphics::IResource::Tag::ComputeShader)));
	res->binBytes = std::move(result).get<0>();
	res->blockSize = blockSize;
	res->properties = std::move(preprocess.properties);
	vstd::vector<vbyte> bt;
	res->Save(bt, db);
	return bt;
}

int main() {
	using namespace toolhub;
	using namespace toolhub::directx;
	Device device;
	auto dbBase = DllManager::GetDatabase();
	auto db = vstd::create_unique(dbBase->CreateDatabase());
	auto str(R"(
/*PROPERTY
{
	"Dispatch" : [32,32,1],
	"Entry": "default_run"
}
*/
RWStructuredBuffer<float3> _Result;
void default_run(uint3 thdId, uint3 dspId, uint3 grpId){
uint2 id = dspId.xy;
uint width = 1024;
uint height = 1024;
uint inputId = width * id.y + id.x;
float2 uv = float2(id) / float2(width, height);
_Result[inputId] = float3(uv, 1);
}
)"_sv);

	auto stbi_write_jpg = DllManager::GetGraphicsDll()->GetDLLFunc<int(char const*, int, int, int, const void*, int)>("stbi_write_jpg");
	auto gra = DllManager::GetGraphics();
	auto res = vstd::create_unique(
		static_cast<graphics::IComputeShader*>(
			gra->CreateResource(db.get(), GetShader(vstd::string(str), db.get()))));

	ComputeShader cs(res.get(), device.device.Get());
	CommandQueue queue(&device, nullptr, D3D12_COMMAND_LIST_TYPE_COMPUTE);

	auto alloc = queue.CreateAllocator();

	DefaultBuffer outputBuffer(&device, 1024 * 1024 * sizeof(float3));

	ResourceStateTracker stateTracker;
	auto dispatchCmd = alloc->GetBuffer();
	vstd::vector<float> floatResult;
	floatResult.resize(1024 * 1024 * 3);
	//Compute
	{
		auto bf = dispatchCmd->Build();
		auto cmdlist = bf.CmdList();
		stateTracker.RecordState(
			&outputBuffer,
			D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
		stateTracker.UpdateState(bf);
		bf.DispatchCompute(
			&cs,
			{1024, 1024, 1},
			{BindProperty{"_Result", &outputBuffer}});
		stateTracker.RecordState(
			&outputBuffer,
            VEngineShaderResourceState);
		stateTracker.UpdateState(bf);
		bf.Readback(
			&outputBuffer,
			floatResult.data());
		stateTracker.RestoreState(bf);
	}
	queue.Complete(
		queue.Execute(std::move(alloc)));
	vstd::vector<vbyte> byteResult;
	byteResult.push_back_func(
		[&](size_t i) {
			return (vbyte)clamp(floatResult[i] * 255, 0, 255.99);
		},
		floatResult.size());

	stbi_write_jpg("hdr_result.jpg", 1024, 1024, 3, reinterpret_cast<float*>(byteResult.data()), 100);
}