#pragma once
#include <Singleton/ShaderID.h>
class ComputeShader;
namespace luisa::compute {
class InternalShaders {
public:
	ComputeShader const* copyShader;
	struct CopyShaderParam {
		static constexpr uint Params = 0;
		static constexpr uint _Tex2D = 1;
		static constexpr uint _Tex3D = 2;
		static constexpr uint _Read_Tex2D = 3;
		static constexpr uint _Read_Tex3D = 4;
		static constexpr uint _Buffer = 5;
		static constexpr uint _Write_Buffer = 6;

	};
	/*uint _Tex2D = ShaderID::PropertyToID("_Tex2D"_sv);
	uint _Tex3D = ShaderID::PropertyToID("_Tex3D"_sv);
	uint _Buffer = ShaderID::PropertyToID("_Buffer"_sv);
	uint _Read_Tex3D = ShaderID::PropertyToID("_Read_Tex3D"_sv);
	uint _Read_Tex2D = ShaderID::PropertyToID("_Read_Tex2D"_sv);
	uint _Write_Buffer = ShaderID::PropertyToID("_Write_Buffer"_sv);*/
	//TODO
};
}// namespace luisa::compute