#include <luisa/xir/instructions/comment.h>

namespace luisa::compute::xir {

CommentInst::CommentInst(Pool *pool, luisa::string comment, const Name *name) noexcept
    : Instruction{pool, nullptr, name}, _comment{std::move(comment)} {}

void CommentInst::set_comment(luisa::string_view comment) noexcept {
    _comment = comment;
}

}// namespace luisa::compute::xir
