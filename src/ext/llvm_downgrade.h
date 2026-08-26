#pragma once

#include <cstddef>
#include <memory>
#include <vector>

namespace llvm {
class Module;
}// namespace llvm

namespace luisa::compute {

[[nodiscard]] std::vector<std::byte>
llvm_downgrade_to_14(std::unique_ptr<llvm::Module> module);

}// namespace luisa::compute
