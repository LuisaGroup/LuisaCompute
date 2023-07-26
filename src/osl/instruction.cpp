#include <luisa/core/stl/format.h>

#include <luisa/osl/symbol.h>
#include <luisa/osl/type.h>
#include <luisa/osl/instruction.h>

namespace luisa::compute::osl {

luisa::string Instruction::dump() const noexcept {
    auto s = _opcode;
    if (!_args.empty()) {
        s.append("\t");
        for (auto &&a : _args) {
            s.append(a->identifier()).append(" ");
        }
        s.pop_back();
    }
    if (!_jump_targets.empty()) {
        s.append("\t");
        for (auto &&t : _jump_targets) {
            s.append(luisa::format("{} ", t));
        }
        s.pop_back();
    }
    if (!_hints.empty()) {
        s.append("\t");
        for (auto &&h : _hints) {
            s.append(h.dump()).append(" ");
        }
        s.pop_back();
    }
    // dump arg types
    if (!_args.empty()) {
        s.append("\t # arg types:");
        for (auto &&a : _args) {
            s.append(luisa::format(
                " {}({}", Symbol::dump(a->tag()),
                a->type()->identifier()));
            if (a->is_array()) {
                if (a->is_unbounded()) {
                    s.append("[]");
                } else {
                    s.append(luisa::format(
                        "[{}]", a->array_length()));
                }
            }
            s.append("),");
        }
        s.pop_back();
    }
    return s;
}

}// namespace luisa::compute::osl
