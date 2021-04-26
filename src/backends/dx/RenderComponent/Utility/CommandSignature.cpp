//#endif
#include <RenderComponent/Utility/CommandSignature.h>
#include <Singleton/ShaderID.h>
CommandSignature::CommandSignature(GFXDevice* device, SignatureType sigType, Shader const* drawShader) : sigType(sigType) {
	switch (sigType) {
		case SignatureType::DispatchComputeIndirect: {
			D3D12_COMMAND_SIGNATURE_DESC desc;
			D3D12_INDIRECT_ARGUMENT_DESC indDesc;
			ZeroMemory(&desc, sizeof(D3D12_COMMAND_SIGNATURE_DESC));
			ZeroMemory(&indDesc, sizeof(D3D12_INDIRECT_ARGUMENT_DESC));
			indDesc.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
			desc.ByteStride = sizeof(uint) * 3;
			desc.NodeMask = 0;
			desc.NumArgumentDescs = 1;
			desc.pArgumentDescs = &indDesc;
			ThrowIfFailed(device->device()->CreateCommandSignature(&desc, nullptr, IID_PPV_ARGS(&mCommandSignature)));
		} break;
	}
}
