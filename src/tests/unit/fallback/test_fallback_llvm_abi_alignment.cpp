#include "fallback_codegen.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/raw_ostream.h>

#include <luisa/ast/type_registry.h>
#include <luisa/runtime/rtx/accel.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>

using namespace luisa::compute;

namespace luisa::compute::fallback::api {
// No ray query objects are instantiated by this codegen-only regression.
extern "C" size_t luisa_fallback_ray_query_object_size() noexcept { return 256u; }
extern "C" size_t luisa_fallback_ray_query_object_alignment() noexcept { return 16u; }
}// namespace luisa::compute::fallback::api

int main() {
    llvm::LLVMContext context;
    llvm::Module output{"fallback-abi-temporary-alignment", context};
    xir::Module input;
    std::unordered_map<std::string, std::vector<size_t>> contracts;
    auto *ptr = llvm::PointerType::getUnqual(context);
    auto declare = [&](const std::string &name, std::vector<size_t> alignment,
                       bool scalar_result = false) {
        std::vector<llvm::Type *> arguments(alignment.size(), ptr);
        auto *result = scalar_result ? llvm::Type::getFloatTy(context)
                                     : llvm::Type::getVoidTy(context);
        llvm::Function::Create(llvm::FunctionType::get(result, arguments, false),
                               llvm::GlobalValue::ExternalLinkage, name, output);
        contracts.emplace(name, std::move(alignment));
    };
    auto arithmetic = [&](const std::string &name, xir::ArithmeticOp op,
                           const Type *result, const Type *lhs, const Type *rhs = nullptr) {
        auto *function = input.create_callable(result);
        auto identifier = name;
        std::replace(identifier.begin(), identifier.end(), '.', '_');
        function->set_name(identifier);
        auto *a = function->create_value_argument(lhs);
        auto *b = rhs ? function->create_value_argument(rhs) : nullptr;
        xir::XIRBuilder builder;
        builder.set_insertion_point(function->create_body_block());
        auto *value = b ? builder.call(result, op, {a, b}) : builder.call(result, op, {a});
        builder.return_(value);
    };
    for (auto dim : {2u, 3u, 4u}) {
        auto *matrix = Type::matrix(dim);
        auto *vector = Type::vector(Type::of<float>(), dim);
        auto alignment = matrix->alignment();
        const auto prefix = "luisa.matrix" + std::to_string(dim) + "d.";
        for (auto [suffix, op] : std::array{
                 std::pair{"transpose", xir::ArithmeticOp::MATRIX_TRANSPOSE},
                 std::pair{"inverse", xir::ArithmeticOp::MATRIX_INVERSE}}) {
            declare(prefix + suffix, {alignment, alignment});
            arithmetic(prefix + suffix + ".probe", op, matrix, matrix);
        }
        declare(prefix + "determinant", {alignment}, true);
        arithmetic(prefix + "determinant.probe", xir::ArithmeticOp::MATRIX_DETERMINANT,
                   Type::of<float>(), matrix);
        for (auto [suffix, rhs] : std::array{
                 std::pair{"mul.matrix", matrix}, std::pair{"mul.vector", vector}}) {
            declare(prefix + suffix, {alignment, rhs->alignment(), rhs->alignment()});
            arithmetic(prefix + suffix + ".probe", xir::ArithmeticOp::MATRIX_LINALG_MUL,
                       rhs, matrix, rhs);
        }
        declare(prefix + "outer.product", {alignment, alignment, alignment});
        arithmetic(prefix + "outer.product.probe", xir::ArithmeticOp::OUTER_PRODUCT,
                   matrix, matrix, matrix);
        auto vector_name = "luisa.vector" + std::to_string(dim) + "d.outer.product";
        declare(vector_name, {vector->alignment(), vector->alignment(), alignment});
        arithmetic(vector_name + ".probe", xir::ArithmeticOp::OUTER_PRODUCT,
                   matrix, vector, vector);
    }
    // The production failure started at the out-parameter of instance_transform,
    // then continued through matrix inverse. Both boundaries must obey the ABI.
    {
        auto *matrix = Type::matrix(4u);
        auto *function = input.create_callable(matrix);
        function->set_name("instance_transform_probe");
        auto *accel = function->create_resource_argument(Type::of<Accel>());
        auto *index = function->create_value_argument(Type::of<luisa::uint>());
        xir::XIRBuilder builder;
        builder.set_insertion_point(function->create_body_block());
        builder.return_(builder.call(matrix, xir::ResourceQueryOp::RAY_TRACING_INSTANCE_TRANSFORM,
                                     {accel, index}));
        auto *signature = llvm::FunctionType::get(llvm::Type::getVoidTy(context),
            {ptr, llvm::Type::getInt32Ty(context), ptr}, false);
        llvm::Function::Create(signature, llvm::GlobalValue::ExternalLinkage,
                               "luisa.accel.instance.transform", output);
        contracts.emplace("luisa.accel.instance.transform", std::vector<size_t>{alignof(void *), 0u, matrix->alignment()});
    }
    static_cast<void>(fallback::luisa_fallback_backend_codegen(context, &output, &input, false));
    if (llvm::verifyModule(output, &llvm::errs())) { return 1; }
    auto passed = true;
    auto checked = 0u;
    for (auto &function : output) {
        for (auto &block : function) {
            for (auto &instruction : block) {
                auto *call = llvm::dyn_cast<llvm::CallBase>(&instruction);
                if (!call || !call->getCalledFunction()) { continue; }
                const auto name = call->getCalledFunction()->getName().str();
                auto contract = contracts.find(name);
                if (contract == contracts.end()) { continue; }
                for (auto i = 0u; i < contract->second.size(); ++i) {
                    auto required = contract->second[i];
                    if (!required) { continue; }
                    ++checked;
                    auto *storage = llvm::dyn_cast<llvm::AllocaInst>(call->getArgOperand(i));
                    if (!storage || storage->getAlign().value() < required) {
                        std::cerr << name << " argument " << i << ": required " << required
                                  << ", allocated " << (storage ? storage->getAlign().value() : 0u) << '\n';
                        passed = false;
                    }
                }
            }
        }
    }
    if (checked != 53u) {
        std::cerr << "Incomplete ABI coverage: " << checked << " pointer arguments\n";
        passed = false;
    }
    if (passed) { std::cout << "Fallback ABI: 53 temporary-pointer alignment contracts passed\n"; }
    return passed ? 0 : 1;
}

