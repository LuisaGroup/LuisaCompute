#include "mesh.h"
#include "procedural_primitives.h"
#include "buffer.h"
#include "accel.h"
namespace lc::validation {
void Mesh::set(Stream *stream, Usage usage) {
    set_usage(stream, this, usage);
    if (usage == Usage::READ) {
        if (vert)
            set_usage(stream, vert, usage);
        if (index)
            set_usage(stream, index, usage);
    }
}
void ProceduralPrimitives::set(Stream *stream, Usage usage) {
    set_usage(stream, this, usage);
    if (usage == Usage::READ) {
        if (bbox)
            set_usage(stream, bbox, usage);
    }
}
void Accel::set(Stream *stream, Usage usage) {
    set_usage(stream, this, usage);
    if (usage == Usage::READ) {
        for (auto &&i : _ref_count) {
            set_usage(stream, i.first, usage);
        }
    }
}
}// namespace lc::validation