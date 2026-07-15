#pragma once

#include <luisa/core/stl.h>

namespace tokenize {

[[nodiscard]] luisa::string porter_stem(luisa::string_view word);

}// namespace tokenize
