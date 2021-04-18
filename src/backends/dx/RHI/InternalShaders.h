#pragma once
#include <Singleton/ShaderID.h>
class ComputeShader;
namespace luisa::compute {
class InternalShaders {
public:
	ComputeShader const* copyShader;
	uint _Tex2D = ShaderID::PropertyToID("_Tex2D"_sv);
	uint _Tex3D = ShaderID::PropertyToID("_Tex3D"_sv);
	uint _Buffer = ShaderID::PropertyToID("_Buffer"_sv);
	uint _Read_Tex3D = ShaderID::PropertyToID("_Read_Tex3D"_sv);
	uint _Read_Tex2D = ShaderID::PropertyToID("_Read_Tex2D"_sv);
	uint _Write_Buffer = ShaderID::PropertyToID("_Write_Buffer"_sv);
};
}// namespace luisa::compute