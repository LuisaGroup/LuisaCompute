#include "TextureBase.h"
#include "../Singleton/Graphics.h"
TextureBase::TextureBase() {
	srvDescID = Graphics::GetDescHeapIndexFromPool();
}
TextureBase::~TextureBase() {
	Graphics::ReturnDescHeapIndexToPool(srvDescID);
}
