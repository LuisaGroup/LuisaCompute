#pragma once

#include <luisa/core/stl.h>

namespace tokenize {

[[nodiscard]] luisa::string soundex(luisa::string_view word);
[[nodiscard]] luisa::string metaphone(luisa::string_view word);

}// namespace tokenize
