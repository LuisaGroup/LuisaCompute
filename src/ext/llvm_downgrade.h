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

// Consumes an already optimized module. Rejects freeze and types without an
// LLVM 7 encoding before invoking the legacy writer's pointer preparation.
[[nodiscard]] std::vector<std::byte>
llvm_downgrade_to_7(std::unique_ptr<llvm::Module> module) noexcept;

}// namespace luisa::compute
