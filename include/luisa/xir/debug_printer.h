#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/memory.h>
#include <luisa/xir/metadata.h>

namespace luisa::compute {
class Type;
}// namespace luisa::compute

namespace luisa::compute::xir {

class Value;
class Instruction;
class Function;
class Module;
class Constant;
class BasicBlock;

class LUISA_XIR_API XIRDebugPrinter {

private:
    struct Impl;
    luisa::unique_ptr<Impl> _impl;

public:
    XIRDebugPrinter() noexcept;
    ~XIRDebugPrinter() noexcept;

public:
    void reset() noexcept;
    void emit_type(luisa::string &s, const Type *type) noexcept;
    void emit_value_name(luisa::string &s, const Value *value) noexcept;
    void emit_value_debug_info(luisa::string &s, const Value *value) noexcept;
    void emit_operand(luisa::string &s, const Value *value) noexcept;
    void emit_instruction(luisa::string &s, const Instruction *instruction) noexcept;
    void emit_basic_block(luisa::string &s, const BasicBlock *block) noexcept;
    void emit_constant(luisa::string &s, const Constant *value) noexcept;
    void emit_function_decl(luisa::string &s, const Function *function) noexcept;
    void emit_function(luisa::string &s, const Function *function) noexcept;
    void emit_module(luisa::string &s, const Module *module) noexcept;
    static void emit_metadata_list(luisa::string &s, const MetadataList &metadata) noexcept;
    [[nodiscard]] static XIRDebugPrinter &global() noexcept;
};

}// namespace luisa::compute::xir
