// OCR engine -- command-recording executor contract (Phase 3).
#include "ocr/runtime/executor.h"

#include <luisa/core/stl/format.h>

namespace luisa::ocr {

[[nodiscard]] Status DecodeRequest::validate() const {
    if (prompt_tokens.empty()) {
        return Status::error(StatusCode::invalid_argument, "decode prompt must not be empty", "prompt_tokens");
    }
    if (has_multimodal()) {
        if (!mm_features) {
            return Status::error(StatusCode::invalid_argument,
                                 "multimodal features view is invalid", "mm_features");
        }
        const uint64_t end = static_cast<uint64_t>(mm_start) + mm_count;
        if (end > prompt_tokens.size()) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("multimodal range [{}, {}) exceeds prompt length {}",
                                               mm_start, end, prompt_tokens.size()),
                                 "mm_start/mm_count");
        }
    } else if (mm_start != 0u) {
        return Status::error(StatusCode::invalid_argument,
                             "mm_start must be 0 when no multimodal features are present", "mm_start");
    }
    return Status::success();
}

}// namespace luisa::ocr