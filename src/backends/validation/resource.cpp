#include "resource.h"
namespace lc::validation {
vstd::string Resource::get_tag_name(Tag tag) {
    switch (tag) {
        case Tag::BUFFER:
            return "Buffer";
        case Tag::TEXTURE:
        // TODO: 2D and 3D name
            return "Texture";
        case Tag::BINDLESS_ARRAY:
            return "Bindless-Array";
        case Tag::MESH:
            return "Mesh";
        case Tag::PROCEDURAL_PRIMITIVE:
            return "Procedural-Primitive";
        case Tag::ACCEL:
            return "Accel";
        case Tag::STREAM:
            return "Stream";
        case Tag::EVENT:
            return "Event";
        case Tag::SHADER:
            return "Shader";
        case Tag::RASTER_SHADER:
            return "Raster-Shader";
        case Tag::SWAP_CHAIN:
            return "Swap-chain";
        case Tag::DEPTH_BUFFER:
            return "Depth-Buffer";
        default:
            return {};
    }
}
vstd::string Resource::get_name() const {
    if (name.empty()) {
        return vstd::string{"Unnamed "}.append(get_tag_name(_tag));
    } else {
        auto result = get_tag_name(_tag);
        result += " ";
        result += name;
        return result;
    }
}
}// namespace lc::validation