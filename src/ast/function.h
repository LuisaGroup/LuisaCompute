//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <core/memory.h>

namespace luisa::compute {

struct Statement;
struct Expression;

template<typename Impl>
class BasicFunction : public Noncopyable {

private:
    Arena _arena;
};

class GlobalFunction : public BasicFunction<GlobalFunction> {

};

}// namespace luisa
