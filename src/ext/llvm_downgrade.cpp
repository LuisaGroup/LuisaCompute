#include "llvm_downgrade.h"

#include <cstring>

#include <llvm/ADT/SmallVector.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>

namespace luisa::compute {

std::vector<std::byte>
llvm_downgrade_to_14(std::unique_ptr<llvm::Module> module) {
    llvm::BitcodeWriter140::prepareModule(*module);
    llvm::SmallVector<char, 0u> storage;
    llvm::raw_svector_ostream stream{storage};
    llvm::WriteBitcode140ToFile(*module, stream);
    std::vector<std::byte> bitcode(storage.size());
    std::memcpy(bitcode.data(), storage.data(), storage.size());
    return bitcode;
}

}// namespace luisa::compute
