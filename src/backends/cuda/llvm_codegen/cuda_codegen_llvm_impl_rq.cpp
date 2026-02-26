//
// Created by mike on 11/1/25.
//

#include <llvm/IR/Dominators.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IR/InlineAsm.h>

#include "cuda_codegen_llvm_impl.h"

namespace luisa::compute::cuda {

void CUDACodegenLLVMImpl::_translate_ray_query_loop_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryLoopInst *inst) noexcept {
    b.GetInsertBlock()->setName("ray.query.loop");
    auto llvm_dispatch_block = func_ctx.get_local_value<llvm::BasicBlock>(inst->dispatch_block());
    llvm_dispatch_block->setName("ray.query.dispatch");
    b.CreateBr(llvm_dispatch_block);
}

void CUDACodegenLLVMImpl::_translate_ray_query_dispatch_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryDispatchInst *inst) noexcept {
    // luisa.ray.query.proceed();
    // switch (luisa.ray.query.state()) {
    //    case surface: br surface_block
    //    case procedural: br procedural_block
    //    default: br exit_block
    // }
    _call_ray_query_intrinsic(b, llvm_ray_query_intrinsic_name_proceed, b.getVoidTy(), {});
    auto llvm_state = _call_ray_query_intrinsic(b, llvm_ray_query_intrinsic_name_state, b.getInt8Ty(), {});
    auto llvm_exit_block = func_ctx.get_local_value<llvm::BasicBlock>(inst->exit_block());
    llvm_exit_block->setName("ray.query.exit");
    auto llvm_surface_block = func_ctx.get_local_value<llvm::BasicBlock>(inst->on_surface_candidate_block());
    llvm_surface_block->setName("ray.query.on.surface.candidate");
    auto llvm_procedural_block = func_ctx.get_local_value<llvm::BasicBlock>(inst->on_procedural_candidate_block());
    llvm_procedural_block->setName("ray.query.on.procedural.candidate");
    auto llvm_dispatch = b.CreateSwitch(llvm_state, llvm_exit_block, 2);
    llvm_dispatch->addCase(b.getInt8(llvm_ray_query_state_surface_candidate), llvm_surface_block);
    llvm_dispatch->addCase(b.getInt8(llvm_ray_query_state_procedural_candidate), llvm_procedural_block);
}

void CUDACodegenLLVMImpl::_lower_ray_query_intrinsics(llvm::Function *f) noexcept {
    IB b{_llvm_context};
    llvm::SmallVector<llvm::Instruction *, 16> to_remove;
    for (auto &bb : *f) {
        for (auto &inst : bb) {
            if (auto call = llvm::dyn_cast<llvm::CallInst>(&inst)) {
                auto callee = call->getCalledFunction();
                if (callee == nullptr) continue;
                auto name = callee->getName();
                b.SetInsertPoint(call);
                if (name == llvm::StringRef{llvm_ray_query_intrinsic_name_world_space_ray.data(), llvm_ray_query_intrinsic_name_world_space_ray.size()}) {
                    call->replaceAllUsesWith(_call_optix_get_world_space_ray(b));
                    to_remove.push_back(call);
                } else if (name == llvm::StringRef{llvm_ray_query_intrinsic_name_procedural_candidate_hit.data(), llvm_ray_query_intrinsic_name_procedural_candidate_hit.size()}) {
                    auto inst_id = _call_optix_read_instance_index(b);
                    auto prim_id = _call_optix_read_primitive_index(b);
                    auto hit = static_cast<llvm::Value *>(llvm::PoisonValue::get(_get_llvm_procedural_hit_type()));
                    hit = b.CreateInsertValue(hit, inst_id, llvm_procedural_hit_type_inst_id_index);
                    hit = b.CreateInsertValue(hit, prim_id, llvm_procedural_hit_type_prim_id_index);
                    call->replaceAllUsesWith(hit);
                    to_remove.push_back(call);
                } else if (name == llvm::StringRef{llvm_ray_query_intrinsic_name_surface_candidate_hit.data(), llvm_ray_query_intrinsic_name_surface_candidate_hit.size()}) {
                    auto inst_id = _call_optix_read_instance_index(b);
                    auto prim_id = _call_optix_read_primitive_index(b);
                    auto bary = _call_optix_get_triangle_barycentrics(b);
                    auto t = _call_optix_get_hit_distance(b);
                    auto hit = static_cast<llvm::Value *>(llvm::PoisonValue::get(_get_llvm_surface_hit_type()));
                    hit = b.CreateInsertValue(hit, inst_id, llvm_surface_hit_type_inst_id_index);
                    hit = b.CreateInsertValue(hit, prim_id, llvm_surface_hit_type_prim_id_index);
                    hit = b.CreateInsertValue(hit, bary, llvm_surface_hit_type_bary_index);
                    hit = b.CreateInsertValue(hit, t, llvm_surface_hit_type_t_index);
                    call->replaceAllUsesWith(hit);
                    to_remove.push_back(call);
                } else if (name == llvm::StringRef{llvm_ray_query_intrinsic_name_committed_hit.data(), llvm_ray_query_intrinsic_name_committed_hit.size()}) {
                    auto is_hit = _call_optix_hit_object_is_hit(b);
                    auto inst_id = b.CreateSelect(is_hit, _call_optix_hit_object_instance_index(b), b.getInt32(~0u));
                    auto prim_id = _call_optix_hit_object_primitive_index(b);
                    auto bary = _call_optix_hit_object_triangle_barycentrics(b);
                    auto hit_kind = _call_optix_hit_object_hit_kind(b);
                    auto kind = b.CreateSelect(is_hit,
                                               b.CreateSelect(b.CreateICmpUGT(hit_kind, b.getInt32(127u)),
                                                              b.getInt32(1u /* BUILTIN */),
                                                              b.getInt32(2u /* PROCEDURAL */)),
                                               b.getInt32(0u /* MISS */));
                    auto t = _call_optix_hit_object_ray_t_max(b);
                    auto hit = static_cast<llvm::Value *>(llvm::PoisonValue::get(_get_llvm_committed_hit_type()));
                    hit = b.CreateInsertValue(hit, inst_id, llvm_committed_hit_type_inst_id_index);
                    hit = b.CreateInsertValue(hit, prim_id, llvm_committed_hit_type_prim_id_index);
                    hit = b.CreateInsertValue(hit, bary, llvm_committed_hit_type_bary_index);
                    hit = b.CreateInsertValue(hit, kind, llvm_committed_hit_type_hit_kind_index);
                    hit = b.CreateInsertValue(hit, t, llvm_committed_hit_type_t_index);
                    call->replaceAllUsesWith(hit);
                    to_remove.push_back(call);
                } else if (name == llvm::StringRef{llvm_ray_query_intrinsic_name_is_terminated.data(), llvm_ray_query_intrinsic_name_is_terminated.size()}) {
                    call->replaceAllUsesWith(b.getInt1(true));
                    to_remove.push_back(call);
                } else if (name == llvm::StringRef{llvm_ray_query_intrinsic_name_is_surface_candidate.data(), llvm_ray_query_intrinsic_name_is_surface_candidate.size()}) {
                    auto llvm_state = _call_ray_query_intrinsic(b, llvm_ray_query_intrinsic_name_state, b.getInt8Ty(), {});
                    call->replaceAllUsesWith(b.CreateICmpEQ(llvm_state, b.getInt8(llvm_ray_query_state_surface_candidate)));
                    to_remove.push_back(call);
                } else if (name == llvm::StringRef{llvm_ray_query_intrinsic_name_is_procedural_candidate.data(), llvm_ray_query_intrinsic_name_is_procedural_candidate.size()}) {
                    auto llvm_state = _call_ray_query_intrinsic(b, llvm_ray_query_intrinsic_name_state, b.getInt8Ty(), {});
                    call->replaceAllUsesWith(b.CreateICmpEQ(llvm_state, b.getInt8(llvm_ray_query_state_procedural_candidate)));
                    to_remove.push_back(call);
                }
            }
        }
    }
    for (auto inst : to_remove) {
        inst->eraseFromParent();
    }
}

void CUDACodegenLLVMImpl::_materialize_ray_query_loops() noexcept {
    llvm::SmallVector<llvm::Function *, 4> extracted_funcs;
    for (auto &F : *_llvm_module) {
        if (F.getName().starts_with("ray.query.loop.extracted")) {
            extracted_funcs.push_back(&F);
        }
    }
    if (extracted_funcs.empty()) {
        for (auto &F : *_llvm_module) {
            _lower_ray_query_intrinsics(&F);
        }
        return;
    }

    for (size_t i = 0; i < extracted_funcs.size(); ++i) {
        auto F = extracted_funcs[i];

        llvm::CallInst *loop_call = nullptr;
        for (auto U : F->users()) {
            if (auto CI = llvm::dyn_cast<llvm::CallInst>(U)) {
                loop_call = CI;
                break;
            }
        }
        LUISA_ASSERT(loop_call != nullptr, "Ray query loop extracted function has no call site.");

        llvm::CallInst *spawn_call = nullptr;
        for (auto &BB : *loop_call->getFunction()) {
            for (auto &I : BB) {
                if (auto CI = llvm::dyn_cast<llvm::CallInst>(&I)) {
                    if (CI->getCalledFunction() && CI->getCalledFunction()->getName() == llvm::StringRef{llvm_ray_query_intrinsic_name_spawn.data(), llvm_ray_query_intrinsic_name_spawn.size()}) {
                        spawn_call = CI;
                        break;
                    }
                }
            }
            if (spawn_call) break;
        }
        LUISA_ASSERT(spawn_call != nullptr, "Spawn call not found for ray query loop.");

        IB b{loop_call};
        auto ctx_ptr = loop_call->getArgOperand(0);
        auto ctx_ptr_int = b.CreatePtrToInt(ctx_ptr, b.getInt64Ty());
        auto p_ctx_hi = b.CreateTrunc(b.CreateLShr(ctx_ptr_int, 32), b.getInt32Ty());
        auto p_ctx_lo = b.CreateTrunc(ctx_ptr_int, b.getInt32Ty());

        auto r0 = b.CreateOr(b.CreateShl(b.getInt32(static_cast<uint32_t>(i)), 24),
                             b.CreateAnd(p_ctx_hi, 0xffffffu));
        auto r1 = p_ctx_lo;

        auto accel = spawn_call->getArgOperand(0);
        auto ray = spawn_call->getArgOperand(1);
        auto time = spawn_call->getArgOperand(2);
        auto mask = spawn_call->getArgOperand(3);
        auto flags_val = spawn_call->getArgOperand(4);
        auto flags_int = llvm::dyn_cast<llvm::ConstantInt>(flags_val);
        LUISA_ASSERT(flags_int != nullptr, "Ray query flags must be constant.");

        _call_optix_trace(b, 5 /* LC_PAYLOAD_TYPE_RAY_QUERY */, 5 /* sbt_offset */, static_cast<uint32_t>(flags_int->getZExtValue()),
                          accel, ray, time, mask, {r0, r1});

        // Define return type: struct { float t, i8 committed, i8 terminated }
        auto result_type = llvm::StructType::get(_llvm_context, {b.getFloatTy(),
                                                                b.getInt8Ty(),
                                                                b.getInt8Ty()});

        auto create_intersection_func = [&](luisa::string name) {
            auto ft = llvm::FunctionType::get(result_type, {b.getPtrTy()}, false);
            auto new_f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, llvm::StringRef{name.data(), name.size()}, _llvm_module.get());
            new_f->addFnAttr(llvm::Attribute::AlwaysInline);
            return new_f;
        };

        auto triangle_f_name = luisa::format("lc_ray_query_triangle_intersection_{}", i);
        auto procedural_f_name = luisa::format("lc_ray_query_procedural_intersection_{}", i);
        auto triangle_f = create_intersection_func(triangle_f_name);
        auto procedural_f = create_intersection_func(procedural_f_name);

        auto transform_intersection_func = [&](llvm::Function *new_f, bool is_triangle) {
            llvm::ValueToValueMapTy vmap;
            vmap[F->getArg(0)] = new_f->getArg(0);
            llvm::SmallVector<llvm::ReturnInst *, 4> returns;
            llvm::CloneFunctionInto(new_f, F, vmap, llvm::CloneFunctionChangeType::LocalChangesOnly, returns);

            // Lower intrinsics in the new function
            IB nb{_llvm_context};
            auto &alloca_bb = new_f->getEntryBlock();
            nb.SetInsertPoint(&alloca_bb, alloca_bb.begin());
            auto t_hit_alloca = nb.CreateAlloca(nb.getFloatTy(), nullptr, "t_hit");
            auto committed_alloca = nb.CreateAlloca(nb.getInt8Ty(), nullptr, "committed");
            auto terminated_alloca = nb.CreateAlloca(nb.getInt8Ty(), nullptr, "terminated");
            nb.CreateStore(llvm::ConstantFP::get(nb.getFloatTy(), 0.0), t_hit_alloca);
            nb.CreateStore(nb.getInt8(0), committed_alloca);
            nb.CreateStore(nb.getInt8(0), terminated_alloca);

            llvm::SmallVector<llvm::Instruction *, 16> to_remove;
            llvm::BasicBlock *dispatch_bb = nullptr;
            llvm::BasicBlock *exit_bb = nullptr;

            for (auto &bb : *new_f) {
                for (auto &inst : bb) {
                    if (auto call = llvm::dyn_cast<llvm::CallInst>(&inst)) {
                        auto callee = call->getCalledFunction();
                        if (callee == nullptr) continue;
                        auto name = callee->getName();
                        if (name == llvm::StringRef{llvm_ray_query_intrinsic_name_state.data(), llvm_ray_query_intrinsic_name_state.size()}) {
                            call->replaceAllUsesWith(nb.getInt8(is_triangle ? llvm_ray_query_state_surface_candidate : llvm_ray_query_state_procedural_candidate));
                            to_remove.push_back(call);
                        } else if (name == llvm::StringRef{llvm_ray_query_intrinsic_name_commit_surface_hit.data(), llvm_ray_query_intrinsic_name_commit_surface_hit.size()}) {
                            nb.SetInsertPoint(call);
                            nb.CreateStore(nb.getInt8(1), committed_alloca);
                            to_remove.push_back(call);
                        } else if (name == llvm::StringRef{llvm_ray_query_intrinsic_name_commit_procedural_hit.data(), llvm_ray_query_intrinsic_name_commit_procedural_hit.size()}) {
                            nb.SetInsertPoint(call);
                            nb.CreateStore(nb.getInt8(1), committed_alloca);
                            nb.CreateStore(call->getArgOperand(0), t_hit_alloca);
                            to_remove.push_back(call);
                        } else if (name == llvm::StringRef{llvm_ray_query_intrinsic_name_terminate.data(), llvm_ray_query_intrinsic_name_terminate.size()}) {
                            nb.SetInsertPoint(call);
                            nb.CreateStore(nb.getInt8(1), terminated_alloca);
                            to_remove.push_back(call);
                        } else if (name == llvm::StringRef{llvm_ray_query_intrinsic_name_dispatch.data(), llvm_ray_query_intrinsic_name_dispatch.size()}) {
                            dispatch_bb = &bb;
                            to_remove.push_back(call);
                        }
                    }
                }
            }

            for (auto inst : to_remove) {
                inst->eraseFromParent();
            }

            LUISA_ASSERT(dispatch_bb != nullptr, "Dispatch block not found in ray query loop.");
            auto dispatch_term = dispatch_bb->getTerminator();
            if (auto sw = llvm::dyn_cast<llvm::SwitchInst>(dispatch_term)) {
                exit_bb = sw->getDefaultDest();
            } else if (auto br = llvm::dyn_cast<llvm::BranchInst>(dispatch_term)) {
                if (br->isConditional()) {
                    exit_bb = br->getSuccessor(1);
                }
            }
            LUISA_ASSERT(exit_bb != nullptr, "Exit block not found in ray query loop.");

            // Create a new return block
            auto ret_bb = llvm::BasicBlock::Create(_llvm_context, "return", new_f);
            nb.SetInsertPoint(ret_bb);
            auto t_hit = nb.CreateLoad(nb.getFloatTy(), t_hit_alloca);
            auto committed = nb.CreateLoad(nb.getInt8Ty(), committed_alloca);
            auto terminated = nb.CreateLoad(nb.getInt8Ty(), terminated_alloca);
            auto ret_val = static_cast<llvm::Value *>(llvm::PoisonValue::get(result_type));
            ret_val = nb.CreateInsertValue(ret_val, t_hit, 0);
            ret_val = nb.CreateInsertValue(ret_val, committed, 1);
            ret_val = nb.CreateInsertValue(ret_val, terminated, 2);
            nb.CreateRet(ret_val);

            // Replace all backedges to dispatch with branch to return
            for (auto &bb : *new_f) {
                auto term = bb.getTerminator();
                for (unsigned j = 0; j < term->getNumSuccessors(); ++j) {
                    if (term->getSuccessor(j) == dispatch_bb) {
                        term->setSuccessor(j, ret_bb);
                    }
                }
            }

            // Replace all branches to exit with terminated = 1 and branch to return
            for (auto &bb : *new_f) {
                if (&bb == ret_bb) continue;
                auto term = bb.getTerminator();
                for (unsigned j = 0; j < term->getNumSuccessors(); ++j) {
                    if (term->getSuccessor(j) == exit_bb) {
                        nb.SetInsertPoint(term);
                        nb.CreateStore(nb.getInt8(1), terminated_alloca);
                        term->setSuccessor(j, ret_bb);
                    }
                }
            }

            // Remove original return instructions if any (there should be none reachable now)
            for (auto ret : returns) {
                if (ret->getParent() != nullptr) {
                    nb.SetInsertPoint(ret);
                    nb.CreateBr(ret_bb);
                    ret->eraseFromParent();
                }
            }
        };

        transform_intersection_func(triangle_f, true);
        transform_intersection_func(procedural_f, false);

        // Remove the call to the extracted loop and the spawn call
        loop_call->eraseFromParent();
        spawn_call->eraseFromParent();
    }

    // Generate __anyhit__ray_query and __intersection__ray_query entry points
    auto generate_entry = [&](luisa::string entry_name, bool is_triangle) {
        auto ft = llvm::FunctionType::get(llvm::Type::getVoidTy(_llvm_context), {}, false);
        auto entry_f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, llvm::StringRef{entry_name.data(), entry_name.size()}, _llvm_module.get());
        auto entry_bb = llvm::BasicBlock::Create(_llvm_context, "entry", entry_f);
        IB eb{entry_bb};

        _call_optix_set_payload_types(eb, 1u << 1u /* LC_PAYLOAD_TYPE_RAY_QUERY */);
        auto r0 = _call_optix_get_payload(eb, 0u);
        auto r1 = _call_optix_get_payload(eb, 1u);
        auto query_id = eb.CreateLShr(r0, 24);
        auto p_ctx_hi = eb.CreateAnd(r0, 0xffffffu);
        auto p_ctx_lo = r1;
        auto p_ctx = eb.CreateIntToPtr(
            eb.CreateOr(eb.CreateShl(eb.CreateZExt(p_ctx_hi, eb.getInt64Ty()), 32),
                        eb.CreateZExt(p_ctx_lo, eb.getInt64Ty())),
            eb.getPtrTy());

        auto exit_bb = llvm::BasicBlock::Create(_llvm_context, "exit", entry_f);
        auto sw = eb.CreateSwitch(query_id, exit_bb, extracted_funcs.size());

        for (size_t i = 0; i < extracted_funcs.size(); ++i) {
            auto case_bb = llvm::BasicBlock::Create(_llvm_context, llvm::Twine{"case_"} + llvm::Twine{i}, entry_f);
            eb.SetInsertPoint(case_bb);
            auto intersection_f_name = luisa::format("lc_ray_query_{}_intersection_{}", is_triangle ? "triangle" : "procedural", i);
            auto intersection_f = _llvm_module->getFunction(llvm::StringRef{intersection_f_name.data(), intersection_f_name.size()});
            auto res = eb.CreateCall(intersection_f, {p_ctx});
            auto t_hit = eb.CreateExtractValue(res, 0);
            auto committed = eb.CreateExtractValue(res, 1);
            auto terminated = eb.CreateExtractValue(res, 2);

            if (is_triangle) {
                auto ignore_bb = llvm::BasicBlock::Create(_llvm_context, llvm::Twine{"ignore_"} + llvm::Twine{i}, entry_f);
                auto check_terminate_bb = llvm::BasicBlock::Create(_llvm_context, llvm::Twine{"check_terminate_"} + llvm::Twine{i}, entry_f);
                eb.CreateCondBr(eb.CreateICmpNE(committed, eb.getInt8(0)), check_terminate_bb, ignore_bb);
                eb.SetInsertPoint(ignore_bb);
                _call_optix_ignore_intersection(eb);
                eb.CreateBr(check_terminate_bb);
                eb.SetInsertPoint(check_terminate_bb);
                auto terminate_bb = llvm::BasicBlock::Create(_llvm_context, llvm::Twine{"terminate_"} + llvm::Twine{i}, entry_f);
                eb.CreateCondBr(eb.CreateICmpNE(terminated, eb.getInt8(0)), terminate_bb, exit_bb);
                eb.SetInsertPoint(terminate_bb);
                _call_optix_terminate_ray(eb);
                eb.CreateBr(exit_bb);
            } else {
                auto report_bb = llvm::BasicBlock::Create(_llvm_context, llvm::Twine{"report_"} + llvm::Twine{i}, entry_f);
                eb.CreateCondBr(eb.CreateICmpNE(committed, eb.getInt8(0)), report_bb, exit_bb);
                eb.SetInsertPoint(report_bb);
                auto hit_kind_val = eb.CreateSelect(eb.CreateICmpNE(terminated, eb.getInt8(0)),
                                                    eb.getInt32(0x02u /* LC_HIT_KIND_PROCEDURAL_TERMINATED */),
                                                    eb.getInt32(0x01u /* LC_HIT_KIND_PROCEDURAL */));
                _call_optix_report_intersection(eb, hit_kind_val, t_hit);
                eb.CreateBr(exit_bb);
            }
            sw->addCase(eb.getInt32(static_cast<uint32_t>(i)), case_bb);
        }

        eb.SetInsertPoint(exit_bb);
        eb.CreateRetVoid();
    };

    generate_entry("__anyhit__ray_query", true);
    generate_entry("__intersection__ray_query", false);

    // Lower ray query intrinsics in all functions
    for (auto &f : *_llvm_module) {
        _lower_ray_query_intrinsics(&f);
    }

    // Remove the original extracted functions
    for (auto f : extracted_funcs) {
        f->eraseFromParent();
    }
}

}// namespace luisa::compute::cuda
