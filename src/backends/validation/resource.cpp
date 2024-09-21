#include "resource.h"
#include "texture.h"
#include "stream.h"
#include <luisa/core/stl/format.h>

namespace lc::validation {

vstd::string Resource::get_tag_name(Tag tag) const {
    switch (tag) {
        case Tag::BUFFER:
            return "buffer";
        case Tag::TEXTURE:
            return luisa::format("{}D-image", static_cast<Texture const *>(this)->dim());
        case Tag::BINDLESS_ARRAY:
            return "bindless-array";
        case Tag::MESH:
            return "mesh";
        case Tag::CURVE:
            return "curve";
        case Tag::PROCEDURAL_PRIMITIVE:
            return "procedural-primitive";
        case Tag::MOTION_INSTANCE:
            return "motion-instance";
        case Tag::ACCEL:
            return "accel";
        case Tag::STREAM:
            return luisa::format("{}-stream", static_cast<Stream const *>(this)->stream_tag());
        case Tag::EVENT:
            return "event";
        case Tag::SHADER:
            return "shader";
        case Tag::RASTER_SHADER:
            return "raster-shader";
        case Tag::SWAP_CHAIN:
            return "swap-chain";
        case Tag::DEPTH_BUFFER:
            return "depth-buffer";
        case Tag::DSTORAGE_FILE:
            return "direct-storage file";
        default:
            return "unknown-resource";
    }
}
vstd::string Resource::get_name() const {
    auto result = get_tag_name(_tag);
    result += " ";
    if (name.empty()) {
        result += "\"un-named\""sv;
    } else {
        result += "\""sv;
        result += name;
        result += "\""sv;
    }
    return result;
}
}// namespace lc::validation
