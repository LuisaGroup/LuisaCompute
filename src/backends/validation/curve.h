//
// Created by Mike on 2024/9/22.
//

#pragma once

#include "rw_resource.h"

namespace lc::validation {

class Buffer;

class Curve : public RWResource {

public:
    Buffer *cp{};
    Buffer *seg{};
    Range cp_range;
    Range seg_range;
    explicit Curve(uint64_t handle)
        : RWResource(handle, Tag::CURVE, false) {}
    void set(Stream *stream, Usage usage, Range range) override;
};

}// namespace lc::validation
