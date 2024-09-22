//
// Created by Mike on 2024/9/22.
//

#include "motion_instance.h"

namespace lc::validation {

void MotionInstance::set(Stream *stream, Usage usage, Range range) {
    set_usage(stream, this, usage, range);
    LUISA_ASSERT(child, "{}'s child must be set before use.", get_name());
    set_usage(stream, child, Usage::READ, Range{});
}

}// namespace lc::validation
