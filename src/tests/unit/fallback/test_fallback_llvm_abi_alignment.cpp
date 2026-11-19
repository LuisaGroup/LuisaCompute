#include "fallback_codegen.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <llvm/IR/CallingConv.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TargetParser/Triple.h>

#include <luisa/ast/type_registry.h>
#include <luisa/runtime/rtx/accel.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/metadata/no_inline.h>
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
    // Internal callables must use LLVM fastcc with normalized aggregate/vec3 ABI
    // (0 FCA params/returns, 0 <3 x T> params/returns) and preserve NoInlineMD
    // across both x86-64 and AArch64 targets.
    {
        auto *agg_ty = Type::matrix(4u);
        auto *vec3_ty = Type::of<luisa::float3>();
        auto *callee = input.create_callable(agg_ty);
        callee->set_name("fastcc_noinline_agg_vec3_callee");
        (void)callee->find_or_create_metadata<xir::NoInlineMD>();
        auto *in_agg = callee->create_value_argument(agg_ty);
        auto *in_v3 = callee->create_value_argument(vec3_ty);
        auto *inout_v3 = callee->create_reference_argument(vec3_ty);
        xir::XIRBuilder b_callee;
        b_callee.set_insertion_point(callee->create_body_block());
        auto *sum_v3 = b_callee.call(vec3_ty, xir::ArithmeticOp::BINARY_ADD, {in_v3, in_v3});
        b_callee.store(inout_v3, sum_v3);
        b_callee.return_(in_agg);

        auto *caller = input.create_callable(vec3_ty);
        caller->set_name("fastcc_agg_vec3_caller");
        auto *c_agg = caller->create_value_argument(agg_ty);
        auto *c_v3 = caller->create_value_argument(vec3_ty);
        xir::XIRBuilder b_caller;
        b_caller.set_insertion_point(caller->create_body_block());
        auto *ref_slot = b_caller.alloca_local(vec3_ty);
        b_caller.store(ref_slot, c_v3);
        auto *ret_agg = b_caller.call(agg_ty, callee, {c_agg, c_v3, ref_slot});
        (void)ret_agg;
        auto *loaded_v3 = b_caller.load(vec3_ty, ref_slot);
        b_caller.return_(loaded_v3);
    }
    // Cached post-Coro XIR preserves struct/array ArithmeticOp::AGGREGATE
    // instructions (e.g., Ray = struct<16, array<float,3>, float, array<float,3>, float>).
    // Verify fallback codegen lowers array and padded/vector-bearing struct
    // aggregates cleanly on both host and AArch64 targets.
    {
        auto *f32_ty = Type::of<float>();
        auto *u8_ty = Type::of<uint8_t>();
        auto *f2_ty = Type::of<luisa::float2>();
        auto *f3_ty = Type::of<luisa::float3>();
        auto *arr_f3_ty = Type::array(f32_ty, 3u);
        auto *arr_vec3_ty = Type::array(f3_ty, 2u);
        auto *ray_ty = Type::structure(16u, {arr_f3_ty, f32_ty, arr_f3_ty, f32_ty});
        auto *padded_ty = Type::structure(16u, {u8_ty, f3_ty, f2_ty, arr_vec3_ty});

        auto *fn = input.create_callable(f32_ty);
        fn->set_name("aggregate_struct_array_probe");
        auto *a = fn->create_value_argument(f32_ty);
        auto *b_val = fn->create_value_argument(f32_ty);
        auto *tag = fn->create_value_argument(u8_ty);
        auto *v2 = fn->create_value_argument(f2_ty);
        auto *v3 = fn->create_value_argument(f3_ty);
        xir::XIRBuilder b_agg;
        b_agg.set_insertion_point(fn->create_body_block());
        auto *origin_arr = b_agg.call(arr_f3_ty, xir::ArithmeticOp::AGGREGATE, {a, b_val, a});
        auto *dir_arr = b_agg.call(arr_f3_ty, xir::ArithmeticOp::AGGREGATE, {b_val, a, b_val});
        auto *ray = b_agg.call(ray_ty, xir::ArithmeticOp::AGGREGATE, {origin_arr, a, dir_arr, b_val});
        auto *vec_arr = b_agg.call(arr_vec3_ty, xir::ArithmeticOp::AGGREGATE, {v3, v3});
        auto *padded = b_agg.call(padded_ty, xir::ArithmeticOp::AGGREGATE, {tag, v3, v2, vec_arr});
        auto k0 = 0u, k1 = 1u, k3 = 3u;
        auto *c0 = input.create_constant(Type::of<uint32_t>(), &k0);
        auto *c1 = input.create_constant(Type::of<uint32_t>(), &k1);
        auto *c3 = input.create_constant(Type::of<uint32_t>(), &k3);
        auto *r_tmax = b_agg.call(f32_ty, xir::ArithmeticOp::EXTRACT, {ray, c3});
        auto *p_v3 = b_agg.call(f3_ty, xir::ArithmeticOp::EXTRACT, {padded, c1});
        auto *p_x = b_agg.call(f32_ty, xir::ArithmeticOp::EXTRACT, {p_v3, c0});
        auto *sum = b_agg.call(f32_ty, xir::ArithmeticOp::BINARY_ADD, {r_tmax, p_x});
        b_agg.return_(sum);
    }
    static_cast<void>(fallback::luisa_fallback_backend_codegen(context, &output, &input, false));
    if (llvm::verifyModule(output, &llvm::errs())) { return 1; }
    auto passed = true;
    // Verify fastcc + normalized ABI + NoInline on both host module and an AArch64 module.
    auto check_fastcc_abi = [&](llvm::Module &mod, const char *label) {
        auto *callee_fn = mod.getFunction("fastcc_noinline_agg_vec3_callee");
        auto *caller_fn = mod.getFunction("fastcc_agg_vec3_caller");
        if (!callee_fn || !caller_fn) {
            std::cerr << "[" << label << "] Missing fastcc probe functions\n";
            passed = false;
            return;
        }
        if (!callee_fn->hasFnAttribute(llvm::Attribute::NoInline) ||
            callee_fn->hasFnAttribute(llvm::Attribute::AlwaysInline)) {
            std::cerr << "[" << label << "] NoInline attribute not preserved on callee\n";
            passed = false;
        }
        for (auto &fn : mod) {
            if (fn.isDeclaration()) {
                if (fn.getCallingConv() != llvm::CallingConv::C) {
                    std::cerr << "[" << label << "] External declaration " << fn.getName().str()
                              << " must retain C calling convention\n";
                    passed = false;
                }
                continue;
            }
            if (fn.hasLocalLinkage() && fn.hasFnAttribute("luisa.fallback.generated_callable")) {
                if (fn.getCallingConv() != llvm::CallingConv::Fast) {
                    std::cerr << "[" << label << "] Generated callable " << fn.getName().str()
                              << " is not fastcc\n";
                    passed = false;
                }
                auto *ft = fn.getFunctionType();
                auto bad_abi_ty = [](const llvm::Type *t) {
                    if (!t) { return false; }
                    if (t->isStructTy() || t->isArrayTy()) { return true; }
                    if (auto *vt = llvm::dyn_cast<llvm::FixedVectorType>(t)) {
                        return vt->getNumElements() == 3u;
                    }
                    return false;
                };
                if (bad_abi_ty(ft->getReturnType())) {
                    std::cerr << "[" << label << "] Unnormalized return type on " << fn.getName().str() << '\n';
                    passed = false;
                }
                for (unsigned i = 0u; i < ft->getNumParams(); ++i) {
                    if (bad_abi_ty(ft->getParamType(i))) {
                        std::cerr << "[" << label << "] Unnormalized param " << i << " on " << fn.getName().str() << '\n';
                        passed = false;
                    }
                }
                for (const auto *u : fn.users()) {
                    if (const auto *cb = llvm::dyn_cast<llvm::CallBase>(u)) {
                        if (cb->getCalledOperand() == &fn &&
                            cb->getCallingConv() != llvm::CallingConv::Fast) {
                            std::cerr << "[" << label << "] Call site to " << fn.getName().str()
                                      << " does not use fastcc\n";
                            passed = false;
                        }
                    }
                }
            }
        }
    };
    check_fastcc_abi(output, "default-layout");
    {
        llvm::LLVMContext aarch64_ctx;
        llvm::Module aarch64_mod{"fallback-abi-aarch64", aarch64_ctx};
#if LLVM_VERSION_MAJOR >= 21
        aarch64_mod.setTargetTriple(llvm::Triple{"aarch64-unknown-linux-gnu"});
#else
        aarch64_mod.setTargetTriple("aarch64-unknown-linux-gnu");
#endif
        aarch64_mod.setDataLayout("e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32");
        for (const auto &[name, align_vec] : contracts) {
            std::vector<llvm::Type *> args;
            for (auto a : align_vec) {
                args.emplace_back(a == 0u ? static_cast<llvm::Type *>(llvm::Type::getInt32Ty(aarch64_ctx))
                                          : static_cast<llvm::Type *>(llvm::PointerType::getUnqual(aarch64_ctx)));
            }
            auto *ret = name.find("determinant") != std::string::npos
                            ? llvm::Type::getFloatTy(aarch64_ctx)
                            : llvm::Type::getVoidTy(aarch64_ctx);
            llvm::Function::Create(llvm::FunctionType::get(ret, args, false),
                                   llvm::GlobalValue::ExternalLinkage, name, aarch64_mod);
        }
        static_cast<void>(fallback::luisa_fallback_backend_codegen(aarch64_ctx, &aarch64_mod, &input, false));
        if (llvm::verifyModule(aarch64_mod, &llvm::errs())) { return 1; }
        check_fastcc_abi(aarch64_mod, "aarch64-linux-gnu");
    }
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

