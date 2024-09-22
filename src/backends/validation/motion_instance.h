//
// Created by Mike on 2024/9/22.
//

#pragma once

#include "rw_resource.h"

namespace lc::validation {

class MotionInstance : public RWResource {

public:
    RWResource *child{};
    explicit MotionInstance(uint64_t handle)
        : RWResource(handle, Tag::MOTION_INSTANCE, false) {}
    void set(Stream *stream, Usage usage, Range range) override;
};

}// namespace lc::validation
