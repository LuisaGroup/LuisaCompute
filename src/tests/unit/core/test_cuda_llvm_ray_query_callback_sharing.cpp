// Host-only tests for CUDA ray-query callback dispatch sharing.
// Check actual switch targets, equivalence boundaries and valid LLVM IR.
#include "cuda_ray_query_callback_sharing.h"
#include "ut/ut.hpp"

#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>

using namespace boost::ut;
using namespace boost::ut::literals;
using luisa::compute::cuda::RayQueryCallbackSharing;

namespace {

struct Fixture {
    llvm::LLVMContext context;
    llvm::Module module{"callback-sharing", context};
    llvm::Function *dispatcher{llvm::Function::Create(
        llvm::FunctionType::get(llvm::Type::getVoidTy(context), {llvm::Type::getInt32Ty(context)}, false),
        llvm::Function::ExternalLinkage, "dispatch", module)};
    RayQueryCallbackSharing sharing{dispatcher};
    llvm::SwitchInst *dispatch;
    uint32_t next_id{0u};

    Fixture() {
        auto entry = llvm::BasicBlock::Create(context, "entry", dispatcher);
        auto exit = llvm::BasicBlock::Create(context, "exit", dispatcher);
        llvm::IRBuilder<> b{exit};
        b.CreateRetVoid();
        b.SetInsertPoint(entry);
        dispatch = b.CreateSwitch(dispatcher->getArg(0u), exit);
    }

    llvm::Function *callback(llvm::StringRef name, uint32_t value = 1u, bool volatile_store = false,
                             uint32_t pointer_address_space = 0u, uint32_t dispatch_lanes = 3u) {
        auto type = llvm::FunctionType::get(llvm::Type::getVoidTy(context),
            {llvm::PointerType::get(context, pointer_address_space),
             llvm::FixedVectorType::get(llvm::Type::getInt32Ty(context), dispatch_lanes),
             llvm::Type::getInt32Ty(context)}, false);
        auto f = llvm::Function::Create(type, llvm::Function::PrivateLinkage, name, module);
        f->setCallingConv(llvm::CallingConv::PTX_Device);
        auto entry = llvm::BasicBlock::Create(context, "entry", f);
        auto body = llvm::BasicBlock::Create(context, "body", f);
        llvm::IRBuilder<> b{entry};
        b.CreateBr(body);
        b.SetInsertPoint(body);
        b.CreateStore(b.getInt32(value), f->getArg(0u), volatile_store);
        b.CreateRetVoid();
        return f;
    }

    RayQueryCallbackSharing::Target add(llvm::Function *callback, bool hardware = true, size_t captures = 0u) {
        auto target = sharing.get_or_create(callback, hardware, captures, "filter");
        dispatch->addCase(llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), next_id++), target.block);
        if (target.created) {
            llvm::IRBuilder<> b{target.block};
            if (callback != nullptr) {
                llvm::SmallVector<llvm::Value *> args;
                for (auto &&arg : callback->args()) { args.emplace_back(llvm::UndefValue::get(arg.getType())); }
                auto call = b.CreateCall(callback, args);
                call->setCallingConv(callback->getCallingConv());
            }
            b.CreateRetVoid();
        }
        return target;
    }

    void check_valid() { expect(!llvm::verifyModule(module, &llvm::errs())); }
};

static auto suite = [] {
    "equivalent ray query callbacks share a complete dispatch target"_test = [] {
        Fixture f;
        auto first = f.callback("first");
        auto duplicate = f.callback("different_name");
        duplicate->getArg(0u)->setName("other_query_name");
        auto different = f.callback("different_constant", 2u);
        auto a = f.add(first);
        auto b = f.add(duplicate);
        auto c = f.add(different);
        expect(a.created && !b.created && c.created);
        expect(a.block == b.block && a.block != c.block);
        expect(f.sharing.shared_count() == 1u);
        expect(f.dispatch->getNumCases() == 3u);
        // entry + default + two distinct filter blocks, rather than three.
        expect(f.dispatcher->size() == 4u);
        auto cases = f.dispatch->case_begin();
        auto first_target = cases->getCaseSuccessor();
        ++cases;
        expect(cases->getCaseSuccessor() == first_target);
        ++cases;
        expect(cases->getCaseSuccessor() != first_target);
        f.check_valid();
    };

    "ray query sharing preserves constants side effects and attributes"_test = [] {
        Fixture f;
        auto original = f.callback("original");
        auto different_constant = f.callback("different_constant", 2u);
        auto volatile_store = f.callback("volatile_store", 1u, true);
        auto function_attribute = f.callback("function_attribute");
        function_attribute->addFnAttr(llvm::Attribute::NoInline);
        auto parameter_attribute = f.callback("parameter_attribute");
        parameter_attribute->addParamAttr(0u, llvm::Attribute::NoAlias);
        auto calling_convention = f.callback("calling_convention");
        calling_convention->setCallingConv(llvm::CallingConv::C);
        auto first = f.add(original);
        for (auto other : {different_constant, volatile_store, function_attribute, parameter_attribute, calling_convention}) {
            auto target = f.add(other);
            expect(target.created && target.block != first.block);
        }
        expect(f.sharing.shared_count() == 0u);
        f.check_valid();
    };

    "ray query sharing requires zero captures and hardware results"_test = [] {
        Fixture f;
        auto callback = f.callback("callback");
        auto ordinary = f.add(callback);
        for (auto count : {1u, 32u, 33u}) {
            auto first = f.add(callback, true, count);
            auto second = f.add(callback, true, count);
            expect(first.created && second.created && first.block != second.block);
            expect(first.block != ordinary.block);
        }
        auto general = f.add(callback, false);
        auto general_again = f.add(callback, false);
        expect(general.created && general_again.created && general.block != general_again.block);
        auto missing = f.add(nullptr);
        auto missing_again = f.add(nullptr);
        expect(missing.created && missing_again.created && missing.block != missing_again.block);
        expect(f.sharing.shared_count() == 0u);
        f.check_valid();
    };

    "ray query sharing rejects metadata and observable callback identity"_test = [] {
        Fixture f;
        auto function_metadata = f.callback("function_metadata");
        function_metadata->setMetadata("test.contract", llvm::MDNode::get(f.context, {}));
        auto instruction_metadata = f.callback("instruction_metadata");
        instruction_metadata->back().front().setMetadata("test.contract", llvm::MDNode::get(f.context, {}));
        auto personality = f.callback("personality");
        auto personality_symbol = llvm::Function::Create(
            llvm::FunctionType::get(llvm::Type::getInt32Ty(f.context), false),
            llvm::Function::ExternalLinkage, "personality_symbol", f.module);
        personality->setPersonalityFn(personality_symbol);
        auto prefix = f.callback("prefix");
        prefix->setPrefixData(llvm::ConstantInt::get(llvm::Type::getInt32Ty(f.context), 1u));
        auto prologue = f.callback("prologue");
        prologue->setPrologueData(llvm::ConstantInt::get(llvm::Type::getInt32Ty(f.context), 1u));
        auto address_taken = f.callback("address_taken");
        new llvm::GlobalVariable(f.module, address_taken->getType(), true,
            llvm::GlobalValue::ExternalLinkage, address_taken, "callback_address");
        auto block_address = f.callback("block_address");
        auto address = llvm::BlockAddress::get(block_address, &block_address->back());
        new llvm::GlobalVariable(f.module, address->getType(), true,
            llvm::GlobalValue::ExternalLinkage, address, "block_address_value");
        for (auto callback : {function_metadata, instruction_metadata, personality, prefix, prologue, address_taken, block_address}) {
            auto first = f.add(callback);
            auto second = f.add(callback);
            expect(first.created && second.created && first.block != second.block);
        }
        expect(f.sharing.shared_count() == 0u);
        f.check_valid();
    };

    "ray query sharing requires the exact callback ABI"_test = [] {
        Fixture f;
        auto pointer_address_space = f.callback("pointer_address_space", 1u, false, 1u);
        auto dispatch_lanes = f.callback("dispatch_lanes", 1u, false, 0u, 2u);
        auto external = f.callback("external");
        external->setLinkage(llvm::GlobalValue::ExternalLinkage);
        for (auto callback : {pointer_address_space, dispatch_lanes, external}) {
            auto first = f.add(callback);
            auto second = f.add(callback);
            expect(first.created && second.created && first.block != second.block);
        }
        expect(f.sharing.shared_count() == 0u);
        f.check_valid();
    };

    "ray query sharing preserves GEP inbounds nuw and nusw flags"_test = [] {
        for (auto flags : {llvm::GEPNoWrapFlags::inBounds(),
                           llvm::GEPNoWrapFlags::noUnsignedWrap(),
                           llvm::GEPNoWrapFlags::noUnsignedSignedWrap()}) {
            Fixture f;
            auto original_flags = flags == llvm::GEPNoWrapFlags::inBounds() ?
                                      llvm::GEPNoWrapFlags::noUnsignedSignedWrap() :
                                      llvm::GEPNoWrapFlags::none();
            auto make_callback = [&](llvm::StringRef name, llvm::GEPNoWrapFlags no_wrap) {
                auto callback = f.callback(name);
                auto body = &callback->back();
                auto store = llvm::cast<llvm::StoreInst>(&body->front());
                llvm::IRBuilder<> b{store};
                auto pointer = b.CreateGEP(b.getInt32Ty(), callback->getArg(0u),
                                           {callback->getArg(2u)}, "state", no_wrap);
                store->setOperand(1u, pointer);
                auto exit = llvm::BasicBlock::Create(f.context, "exit", callback);
                body->getTerminator()->eraseFromParent();
                b.SetInsertPoint(body);
                b.CreateBr(exit);
                b.SetInsertPoint(exit);
                b.CreateRetVoid();
                return callback;
            };
            auto original = make_callback("original", original_flags);
            auto reordered = make_callback("reordered", original_flags);
            // Same CFG, different block-list order. The strict flag check must
            // follow the comparator's pairing instead of zipping block lists.
            reordered->back().moveBefore(reordered->getEntryBlock().getNextNode());
            auto changed = make_callback("changed_flags", flags);
            auto first = f.add(original);
            auto same = f.add(reordered);
            auto different = f.add(changed);
            expect(first.created && !same.created && different.created);
            expect(first.block == same.block && first.block != different.block);
            expect(f.sharing.shared_count() == 1u);
            expect(f.dispatcher->size() == 4u);
            f.check_valid();
        }
    };

    "ray query sharing checks flags of unused GEP instructions"_test = [] {
        Fixture f;
        auto original = f.callback("original");
        auto changed = f.callback("changed");
        for (auto callback : {original, changed}) {
            llvm::IRBuilder<> b{&callback->back().front()};
            auto flags = callback == original ? llvm::GEPNoWrapFlags::none() :
                                               llvm::GEPNoWrapFlags::noUnsignedWrap();
            auto unused = b.CreateGEP(b.getInt32Ty(), callback->getArg(0u),
                                      {callback->getArg(2u)}, "unused", flags);
            expect(unused->use_empty());
        }
        auto first = f.add(original);
        auto second = f.add(changed);
        expect(first.created && second.created && first.block != second.block);
        expect(f.sharing.shared_count() == 0u);
        f.check_valid();
    };

    "ray query sharing preserves global operand identity and fast math flags"_test = [] {
        Fixture f;
        auto global_a = new llvm::GlobalVariable(f.module, llvm::Type::getInt32Ty(f.context), false,
            llvm::GlobalValue::ExternalLinkage, nullptr, "global_a");
        auto global_b = new llvm::GlobalVariable(f.module, llvm::Type::getInt32Ty(f.context), false,
            llvm::GlobalValue::ExternalLinkage, nullptr, "global_b");
        auto first = f.callback("first_global");
        auto same = f.callback("same_global");
        auto other = f.callback("other_global");
        for (auto callback : {first, same, other}) {
            llvm::IRBuilder<> b{&callback->back().front()};
            auto loaded = b.CreateLoad(b.getInt32Ty(), callback == other ? global_b : global_a);
            auto store = llvm::cast<llvm::StoreInst>(callback->back().getTerminator()->getPrevNode());
            store->setOperand(0u, loaded);
        }
        expect(f.add(first).created);
        expect(!f.add(same).created);
        expect(f.add(other).created);
        auto precise = f.callback("precise");
        auto fast = f.callback("fast");
        for (auto callback : {precise, fast}) {
            llvm::IRBuilder<> b{&callback->back().front()};
            auto x = b.CreateBitCast(callback->getArg(2u), b.getFloatTy());
            auto sum = b.CreateFAdd(x, llvm::ConstantFP::get(b.getFloatTy(), 1.0));
            if (callback == fast) {
                llvm::FastMathFlags flags;
                flags.setFast();
                llvm::cast<llvm::Instruction>(sum)->setFastMathFlags(flags);
            }
            auto store = llvm::cast<llvm::StoreInst>(callback->back().getTerminator()->getPrevNode());
            store->setOperand(0u, b.CreateBitCast(sum, b.getInt32Ty()));
        }
        expect(f.add(precise).created);
        expect(f.add(fast).created);
        expect(f.sharing.shared_count() == 1u);
        f.check_valid();
    };
};

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
}
