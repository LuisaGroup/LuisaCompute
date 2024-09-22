//
// Created by Mike on 2024/9/22.
//

#include "buffer.h"
#include "curve.h"

namespace lc::validation {

void Curve::set(Stream *stream, Usage usage, Range range) {
    set_usage(stream, this, usage, range);
    LUISA_ASSERT(cp, "{}'s control point buffer must be set before use.", get_name());
    set_usage(stream, cp, Usage::READ, cp_range);
    LUISA_ASSERT(seg, "{}'s segment buffer must be set before use.", get_name());
    set_usage(stream, seg, Usage::READ, seg_range);
}

}// namespace lc::validation
