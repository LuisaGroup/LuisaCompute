
#include <Resource/TextureBase.h>
namespace toolhub::directx {
TextureBase::TextureBase(
	Device* device,
	uint width,
	uint height,
	GFXFormat format,
	TextureDimension dimension,
	uint depth,
	uint mip)
	: Resource(device),
	  width(width),
	  height(height),
	  format(format),
	  dimension(dimension),
	  depth(depth),
	  mip(mip) {
    this->depth = std::max<uint>(this->depth, 1);
    this->mip = std::max<uint>(this->mip, 1);
	switch (dimension) {
		case TextureDimension::Tex1D:
			this->depth = 1;
			this->height = 1;
			break;
		case TextureDimension::Tex2D:
			this->depth = 1;
			break;
		case TextureDimension::Cubemap:
			this->depth = 6;
			break;

	}

}
TextureBase::~TextureBase() {
}
}