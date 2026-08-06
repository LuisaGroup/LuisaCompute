//
// Created by swfly on 2024/11/21.
//

#include <charconv>
#include <fstream>

#include <llvm/ADT/SmallVector.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Analysis/AliasAnalysis.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Operator.h>

#include <luisa/core/stl.h>
#include <luisa/core/logging.h>
#include <luisa/core/clock.h>

#include <luisa/xir/translators/ast2xir.h>
#include <luisa/xir/translators/xir2text.h>
#include <luisa/xir/instructions/print.h>

#include <luisa/xir/passes/dce.h>
#include <luisa/xir/passes/local_store_forward.h>
#include <luisa/xir/passes/local_load_elimination.h>
#include <luisa/xir/passes/mem2reg.h>
#include <luisa/xir/passes/promote_ref_arg.h>
#include <luisa/xir/passes/lower_ray_query_loop.h>
#include <luisa/xir/passes/destructure_cfg.h>
#include <luisa/xir/passes/simplify_cfg.h>
#include <luisa/xir/passes/restructure_cfg.h>
#include <luisa/xir/passes/early_return_elimination.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/verifier.h>

#include "../common/shader_print_formatter.h"

#include "fallback_device.h"
#include "fallback_codegen.h"
#include "fallback_texture.h"
#include "fallback_accel.h"
#include "fallback_bindless_array.h"
#include "fallback_shader.h"
#include "fallback_buffer.h"
#include "fallback_command_queue.h"
#include "fallback_device_api.h"
#include "fallback_device_api_ir_module.h"

static const bool LUISA_SHOULD_DUMP_XIR = [] {
    if (auto env = getenv("LUISA_DUMP_XIR")) {
        return std::string_view{env} == "1";
    }
    return false;
}();

static const bool LUISA_SHOULD_DUMP_LLVM_IR = [] {
    if (auto env = getenv("LUISA_DUMP_LLVM_IR")) {
        return std::string_view{env} == "1";
    }
    return false;
}();

static const bool LUISA_SHOULD_DUMP_ASM = [] {
    if (auto env = getenv("LUISA_DUMP_ASM")) {
        return std::string_view{env} == "1";
    }
    return false;
}();

static const bool LUISA_XIR_NORMALIZE_CFG = [] {
    if (auto env = getenv("LUISA_XIR_NORMALIZE_CFG")) {
        return std::string_view{env} == "1";
    }
    return false;
}();

static const bool LUISA_XIR_RESTRUCTURE_CFG = [] {
    if (auto env = getenv("LUISA_XIR_RESTRUCTURE_CFG")) {
        return std::string_view{env} == "1";
    }
    return false;
}();

static const std::size_t LUISA_FALLBACK_OPTIMIZATION_INSTRUCTION_LIMIT = [] {
    constexpr std::size_t default_limit = 250'000u;
    if (auto env = getenv("LUISA_FALLBACK_OPTIMIZATION_INSTRUCTION_LIMIT")) {
        auto text = std::string_view{env};
        std::size_t value{};
        auto [end, error] = std::from_chars(
            text.data(), text.data() + text.size(), value);
        if (error == std::errc{} && end == text.data() + text.size() &&
            value != 0u) {
            return value;
        }
        LUISA_WARNING_WITH_LOCATION(
            "Ignoring invalid LUISA_FALLBACK_OPTIMIZATION_INSTRUCTION_LIMIT='{}'.",
            text);
    }
    return default_limit;
}();

static const bool LUISA_XIR_ELIMINATE_EARLY_RETURN = [] {
    if (auto env = getenv("LUISA_XIR_ELIMINATE_EARLY_RETURN")) {
        return std::string_view{env} == "1";
    }
    return false;
}();

namespace luisa::compute::fallback {

namespace {

// Increment whenever the persisted object or its external symbol contract
// changes in a way that makes an older cache artifact unsafe to load.
static constexpr auto fallback_shader_cache_abi = 6u;

void verify_xir_or_error(const xir::Module *module,
                         luisa::string_view stage) noexcept {
    auto verification = xir::xir_verify_module(module);
    if (!verification.succeeded()) {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid XIR at fallback {}: {} ({} error(s) total).",
            stage, verification.errors.front().message, verification.errors.size());
    }
}

[[nodiscard]] bool function_contains_debug_break(
    Function function,
    luisa::unordered_set<uint64_t> &visited) noexcept {
    if (!visited.emplace(function.hash()).second) { return false; }
    auto contains_debug_break = false;
    traverse_expressions<false>(
        function.body(),
        [](auto) noexcept {},
        [&](auto statement) noexcept {
            contains_debug_break |=
                statement->tag() == Statement::Tag::DEBUG_BREAK;
        },
        [](auto) noexcept {});
    if (contains_debug_break) { return true; }
    for (auto &&callable : function.custom_callables()) {
        if (function_contains_debug_break(
                callable->function(), visited)) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool function_contains_debug_break(
    Function function) noexcept {
    luisa::unordered_set<uint64_t> visited;
    return function_contains_debug_break(function, visited);
}

struct FallbackShaderCacheMetadata {
    uint64_t checksum;
    luisa::string object_name;
    luisa::string metadata_name;
    luisa::string serialized;
};

[[nodiscard]] FallbackShaderCacheMetadata make_shader_cache_metadata(
    Function kernel, const ShaderOption &option,
    const llvm::TargetMachine &target_machine) noexcept {
    auto copy_string = [](llvm::StringRef value) noexcept {
        return luisa::string{value.data(), value.size()};
    };
    auto builtin_module = fallback_backend_device_builtin_module();
    auto builtin_hash = luisa::hash64(
        builtin_module.data(), builtin_module.size(),
        luisa::hash64_default_seed);
    auto native_include_hash = luisa::hash64(
        option.native_include.data(), option.native_include.size(),
        luisa::hash64_default_seed);
    auto argument_hash = luisa::hash64_default_seed;
    for (auto argument : kernel.arguments()) {
        argument_hash = luisa::hash_value(
            argument.type()->description(), argument_hash);
        argument_hash = luisa::hash_value(
            luisa::to_underlying(argument.tag()), argument_hash);
        argument_hash = luisa::hash_value(
            luisa::to_underlying(
                kernel.variable_usage(argument.uid())),
            argument_hash);
    }
    auto data_layout =
        target_machine.createDataLayout().getStringRepresentation();
    auto target_triple =
        copy_string(target_machine.getTargetTriple().str());
    auto target_cpu =
        copy_string(target_machine.getTargetCPU());
    auto target_features =
        copy_string(target_machine.getTargetFeatureString());
#ifdef NDEBUG
    static constexpr auto release_build = true;
#else
    static constexpr auto release_build = false;
#endif
    auto body = luisa::format(
        "FALLBACK_CACHE_ABI {}\n"
        "KERNEL_HASH {:016x}\n"
        "ARGUMENT_HASH {:016x}\n"
        "ARGUMENT_COUNT {}\n"
        "BLOCK_SIZE {} {} {}\n"
        "LLVM_VERSION {}.{}.{}\n"
        "TARGET_TRIPLE {}\n"
        "TARGET_CPU {}\n"
        "TARGET_FEATURES {}\n"
        "DATA_LAYOUT {}\n"
        "BUILTIN_HASH {:016x}\n"
        "NATIVE_INCLUDE_HASH {:016x}\n"
        "FAST_MATH {}\n"
        "DEBUG_INFO {}\n"
        "RELEASE_BUILD {}\n"
        "XIR_NORMALIZE_CFG {}\n"
        "XIR_RESTRUCTURE_CFG {}\n"
        "XIR_ELIMINATE_EARLY_RETURN {}\n",
        fallback_shader_cache_abi,
        kernel.hash(),
        argument_hash,
        kernel.arguments().size(),
        kernel.block_size().x,
        kernel.block_size().y,
        kernel.block_size().z,
        LLVM_VERSION_MAJOR,
        LLVM_VERSION_MINOR,
        LLVM_VERSION_PATCH,
        target_triple,
        target_cpu,
        target_features,
        data_layout,
        builtin_hash,
        native_include_hash,
        option.enable_fast_math,
        option.enable_debug_info,
        release_build,
        LUISA_XIR_NORMALIZE_CFG,
        LUISA_XIR_RESTRUCTURE_CFG,
        LUISA_XIR_ELIMINATE_EARLY_RETURN);
    auto checksum = luisa::hash64(
        body.data(), body.size(), luisa::hash64_default_seed);
    auto object_name = luisa::format(
        "kernel_{:016x}.fallback.obj", checksum);
    return {
        .checksum = checksum,
        .object_name = object_name,
        .metadata_name = luisa::format("{}.metadata", object_name),
        .serialized = luisa::format(
            "CHECKSUM {:016x}\n{}", checksum, body)};
}

}// namespace

[[nodiscard]] static luisa::half luisa_fallback_asin_f16(luisa::half x) noexcept { return ::half_float::asin(x); }
[[nodiscard]] static float luisa_fallback_asin_f32(float x) noexcept { return std::asin(x); }
[[nodiscard]] static double luisa_fallback_asin_f64(double x) noexcept { return std::asin(x); }

[[nodiscard]] static luisa::half luisa_fallback_acos_f16(luisa::half x) noexcept { return ::half_float::acos(x); }
[[nodiscard]] static float luisa_fallback_acos_f32(float x) noexcept { return std::acos(x); }
[[nodiscard]] static double luisa_fallback_acos_f64(double x) noexcept { return std::acos(x); }

[[nodiscard]] static luisa::half luisa_fallback_atan_f16(luisa::half x) noexcept { return ::half_float::atan(x); }
[[nodiscard]] static float luisa_fallback_atan_f32(float x) noexcept { return std::atan(x); }
[[nodiscard]] static double luisa_fallback_atan_f64(double x) noexcept { return std::atan(x); }

[[nodiscard]] static luisa::half luisa_fallback_atan2_f16(luisa::half a, luisa::half b) noexcept { return ::half_float::atan2(a, b); }
[[nodiscard]] static float luisa_fallback_atan2_f32(float a, float b) noexcept { return std::atan2(a, b); }
[[nodiscard]] static double luisa_fallback_atan2_f64(double a, double b) noexcept { return std::atan2(a, b); }

[[nodiscard]] static size_t &luisa_coro_buffer_counter() noexcept {
    thread_local size_t counter = 0u;
    return counter;
}

static constexpr size_t luisa_coro_allocation_alignment = 2u * sizeof(intptr_t);

static void luisa_coro_reset_counter() noexcept {
    luisa_coro_buffer_counter() = 0u;
}

[[nodiscard]] static void *luisa_coro_alloc(size_t size) noexcept {
    alignas(luisa_coro_allocation_alignment) thread_local std::byte buffer[luisa::compute::fallback::max_thread_frame_size];
    size = luisa::align(size, luisa_coro_allocation_alignment);
    auto n = (luisa_coro_buffer_counter() += size);
    LUISA_ASSERT(n <= sizeof(buffer), "Coroutine buffer overflow.");
    return buffer + n - size;
}

static void luisa_coro_free(void *ptr) noexcept { /* do nothing */ }

static void *luisa_shared_memory() noexcept {
    alignas(luisa_coro_allocation_alignment) static thread_local std::byte buffer[luisa::compute::fallback::max_shared_memory_size];
    return buffer;
}

static void luisa_fallback_assert(bool condition, const char *message) noexcept {
    if (!condition) { LUISA_ERROR_WITH_LOCATION("Assertion failed: {}.", message); }
}

static thread_local const DeviceInterface::StreamLogCallback *current_device_log_callback{nullptr};

static void luisa_fallback_print(const FallbackShader *shader, size_t fmt_id, const std::byte *args) noexcept {
    static thread_local luisa::string scratch;
    scratch.clear();
    auto formatter = shader->print_formatter(fmt_id);
    (*formatter)(scratch, {args, formatter->size()});
    if (current_device_log_callback) {
        (*current_device_log_callback)(scratch);
    } else {
        LUISA_INFO("[DEVICE] {}", scratch);
    }
}

struct FallbackShaderLaunchConfig {
    uint3 block_id;
    uint3 dispatch_size;
    uint3 block_size;
};

FallbackShader::FallbackShader(FallbackDevice *device, const ShaderOption &option, Function kernel) noexcept {

    _initialize_target_machine_jit(option);

    _block_size = kernel.block_size();
    _build_bound_arguments(kernel.bound_arguments());

    // Compute the dispatch argument layout before trying the cache. Bound
    // resource handles and uniform values are intentionally absent from the
    // cache key: they are encoded into this buffer for each dispatch.
    _argument_buffer_size = 0u;
    static constexpr auto argument_alignment = 16u;
    for (auto arg : kernel.arguments()) {
        switch (arg.tag()) {
            case Variable::Tag::LOCAL: {
                _argument_buffer_size += arg.type()->size();
                _argument_buffer_size = luisa::align(
                    _argument_buffer_size, argument_alignment);
                break;
            }
            case Variable::Tag::BUFFER: {
                _argument_buffer_size += sizeof(FallbackBufferView);
                _argument_buffer_size = luisa::align(
                    _argument_buffer_size, argument_alignment);
                break;
            }
            case Variable::Tag::TEXTURE: {
                _argument_buffer_size += sizeof(FallbackTextureView);
                _argument_buffer_size = luisa::align(
                    _argument_buffer_size, argument_alignment);
                break;
            }
            case Variable::Tag::BINDLESS_ARRAY: {
                _argument_buffer_size += sizeof(FallbackBindlessArray *);
                _argument_buffer_size = luisa::align(
                    _argument_buffer_size, argument_alignment);
                break;
            }
            case Variable::Tag::ACCEL: {
                _argument_buffer_size += sizeof(FallbackAccel *);
                _argument_buffer_size = luisa::align(
                    _argument_buffer_size, argument_alignment);
                break;
            }
            default: LUISA_ERROR_WITH_LOCATION("Unsupported argument type.");
        }
    }

    auto define_common_symbols = [&]() noexcept {
        llvm::orc::SymbolMap symbol_map{};
        auto map_symbol = [
                              jit = _jit.get(),
                              &symbol_map]<typename T>(
                              const char *name, T *function) noexcept {
            auto address = llvm::orc::ExecutorAddr::fromPtr(function);
            auto symbol = llvm::orc::ExecutorSymbolDef{
                address, llvm::JITSymbolFlags::Callable};
            symbol_map.try_emplace(
                jit->mangleAndIntern(name), symbol);
        };

#include "fallback_device_api_map_symbols.inl.h"

        map_symbol("luisa.asin.f16", &luisa_fallback_asin_f16);
        map_symbol("luisa.asin.f32", &luisa_fallback_asin_f32);
        map_symbol("luisa.asin.f64", &luisa_fallback_asin_f64);
        map_symbol("luisa.acos.f16", &luisa_fallback_acos_f16);
        map_symbol("luisa.acos.f32", &luisa_fallback_acos_f32);
        map_symbol("luisa.acos.f64", &luisa_fallback_acos_f64);
        map_symbol("luisa.atan.f16", &luisa_fallback_atan_f16);
        map_symbol("luisa.atan.f32", &luisa_fallback_atan_f32);
        map_symbol("luisa.atan.f64", &luisa_fallback_atan_f64);
        map_symbol("luisa.atan2.f16", &luisa_fallback_atan2_f16);
        map_symbol("luisa.atan2.f32", &luisa_fallback_atan2_f32);
        map_symbol("luisa.atan2.f64", &luisa_fallback_atan2_f64);
        map_symbol("luisa.coro.alloc", &luisa_coro_alloc);
        map_symbol("luisa.coro.free", &luisa_coro_free);
        map_symbol("luisa.shared.memory", &luisa_shared_memory);
        map_symbol("luisa.assert", &luisa_fallback_assert);

        if (auto error = _jit->getMainJITDylib().define(
                llvm::orc::absoluteSymbols(std::move(symbol_map)))) {
            auto message = llvm::toString(std::move(error));
            LUISA_ERROR_WITH_LOCATION(
                "Failed to define fallback JIT symbols: {}.",
                message);
        }
    };
    define_common_symbols();

    auto define_codegen_symbols =
        [&](const FallbackCodeGenFeedback &feedback) noexcept {
            llvm::orc::SymbolMap symbol_map{};
            auto map_symbol = [
                                  jit = _jit.get(),
                                  &symbol_map]<typename T>(
                                  const char *name, T *function) noexcept {
                auto address =
                    llvm::orc::ExecutorAddr::fromPtr(function);
                auto symbol = llvm::orc::ExecutorSymbolDef{
                    address, llvm::JITSymbolFlags::Callable};
                symbol_map.try_emplace(
                    jit->mangleAndIntern(name), symbol);
            };

            if (!feedback.print_inst_map.empty()) {
                map_symbol("luisa.print.context", this);
                _print_formatters.reserve(
                    feedback.print_inst_map.size());
                for (auto format_id = 0u;
                     format_id < feedback.print_inst_map.size();
                     format_id++) {
                    auto &&[print_inst, llvm_symbol] =
                        feedback.print_inst_map[format_id];
                    map_symbol(
                        llvm_symbol.c_str(),
                        &luisa_fallback_print);
                    LUISA_INFO(
                        "Mapping print instruction #{}: \"{}\" -> {}",
                        format_id, print_inst->format(), llvm_symbol);
                    llvm::SmallVector<const Type *, 8u> argument_types;
                    for (auto operand : print_inst->operand_uses()) {
                        argument_types.emplace_back(
                            operand->value()->type());
                    }
                    auto argument_pack_type =
                        Type::structure(16u, argument_types);
                    _print_formatters.emplace_back(
                        luisa::make_unique<ShaderPrintFormatter>(
                            print_inst->format(),
                            argument_pack_type, false));
                }
            }
            for (auto &&[callback, llvm_symbol] :
                 feedback.debug_callback_map) {
                map_symbol(llvm_symbol.c_str(), callback);
                LUISA_INFO(
                    "Mapping debug callback: {} -> {}",
                    reinterpret_cast<void *>(callback), llvm_symbol);
            }
            if (!symbol_map.empty()) {
                if (auto error =
                        _jit->getMainJITDylib().define(
                            llvm::orc::absoluteSymbols(
                                std::move(symbol_map)))) {
                    auto message = llvm::toString(std::move(error));
                    LUISA_ERROR_WITH_LOCATION(
                        "Failed to define fallback shader-specific "
                        "JIT symbols: {}.",
                        message);
                }
            }
        };

    auto has_debug_break = function_contains_debug_break(kernel);
    auto cache_enabled =
        option.enable_cache &&
        option.name.empty() &&
        !kernel.requires_printing() &&
        !has_debug_break &&
        !LUISA_SHOULD_DUMP_XIR &&
        !LUISA_SHOULD_DUMP_LLVM_IR &&
        !LUISA_SHOULD_DUMP_ASM &&
        device->binary_io() != nullptr;
    auto cache_metadata = make_shader_cache_metadata(
        kernel, option, *_target_machine);

    auto lookup_kernel_entry = [&]() noexcept {
        auto address = _jit->lookup("kernel.main");
        if (!address) {
            auto message = llvm::toString(address.takeError());
            LUISA_WARNING_WITH_LOCATION(
                "Fallback JIT kernel lookup failed: {}.", message);
            return false;
        }
        _kernel_entry = address->toPtr<kernel_entry_t>();
        return true;
    };

    if (cache_enabled) {
        Clock cache_clock;
        auto metadata_stream = device->binary_io()->read_shader_cache(
            cache_metadata.metadata_name);
        auto object_stream = device->binary_io()->read_shader_cache(
            cache_metadata.object_name);
        auto metadata_matches = false;
        if (metadata_stream != nullptr &&
            metadata_stream->length() ==
                cache_metadata.serialized.size()) {
            luisa::string serialized;
            serialized.resize(metadata_stream->length());
            metadata_stream->read(luisa::span{
                reinterpret_cast<std::byte *>(serialized.data()),
                serialized.size()});
            metadata_matches =
                serialized == cache_metadata.serialized;
        }
        if (metadata_matches &&
            object_stream != nullptr &&
            object_stream->length() != 0u) {
            auto object = object_stream->read(
                object_stream->length());
            auto object_buffer =
                llvm::MemoryBuffer::getMemBufferCopy(
                    llvm::StringRef{
                        reinterpret_cast<const char *>(object.data()),
                        object.size()},
                    cache_metadata.object_name.c_str());
            if (auto error =
                    _jit->addObjectFile(std::move(object_buffer))) {
                auto message = llvm::toString(std::move(error));
                LUISA_WARNING_WITH_LOCATION(
                    "Fallback shader cache object '{}' is invalid: {}. "
                    "The shader will be recompiled.",
                    cache_metadata.object_name, message);
            } else if (lookup_kernel_entry()) {
                LUISA_VERBOSE(
                    "Fallback shader cache hit '{}' in {} ms.",
                    cache_metadata.object_name,
                    cache_clock.toc());
                return;
            }
            _initialize_target_machine_jit(option);
            define_common_symbols();
        } else {
            LUISA_VERBOSE(
                "Fallback shader cache miss '{}'.",
                cache_metadata.object_name);
        }
    } else if (option.enable_cache) {
        LUISA_VERBOSE(
            "Fallback shader cache disabled for kernel {:016x} "
            "(named={}, printing={}, debug_break={}, dumping={}).",
            kernel.hash(), !option.name.empty(),
            kernel.requires_printing(), has_debug_break,
            LUISA_SHOULD_DUMP_XIR ||
                LUISA_SHOULD_DUMP_LLVM_IR ||
                LUISA_SHOULD_DUMP_ASM);
    }

    LUISA_VERBOSE(
        "======= Fallback Backend JIT Shader Compilation =======");

    Clock translate_clk;
    auto xir_module = xir::ast_to_xir_translate(kernel, {});
    xir_module->set_name(luisa::format("kernel_{:016x}", kernel.hash()));
    if (!option.name.empty()) { xir_module->set_location(option.name); }
    verify_xir_or_error(xir_module.get(), "AST translation");
    LUISA_VERBOSE("AST to XIR translation done in {} ms.", translate_clk.toc());

    // dump for debugging
    if (LUISA_SHOULD_DUMP_XIR) {
        auto filename = luisa::format("kernel.{:016x}.xir", kernel.hash());
        std::ofstream f{filename.c_str()};
        f << xir::xir_to_text_translate(xir_module.get(), true);
    }

    xir::PassPipeline pre_cfg;
    pre_cfg.add("dce", [](xir::Module *m, xir::PassReport &r) {
        auto i = xir::dce_pass_run_on_module(m, &r);
        return i.changed();
    });
    pre_cfg.add("local-store-forward", [](xir::Module *m, xir::PassReport &r) {
        auto i = xir::local_store_forward_pass_run_on_module(m, &r);
        return i.removed_load_count > 0u;
    });
    pre_cfg.add("local-load-elimination", [](xir::Module *m, xir::PassReport &r) {
        auto i = xir::local_load_elimination_pass_run_on_module(m, &r);
        return i.removed_load_count > 0u;
    });
    pre_cfg.add("dce", [](xir::Module *m, xir::PassReport &r) {
        auto i = xir::dce_pass_run_on_module(m, &r);
        return i.changed();
    });
    // pre_cfg.add("promote-ref-arg", [](xir::Module *m, xir::PassReport &r) {
    //     auto i = xir::promote_ref_arg_pass_run_on_module(m, &r);
    //     return i.promoted_ref_arg_count > 0u;
    // });
    if (LUISA_XIR_ELIMINATE_EARLY_RETURN) {
        pre_cfg.add("early-return-elimination", [](xir::Module *m, xir::PassReport &r) {
            auto i = xir::early_return_elimination_pass_run_on_module(m, &r);
            return i.removed_return_count > 0u;
        });
    }
    pre_cfg.add("mem2reg", [](xir::Module *m, xir::PassReport &r) {
        auto i = xir::mem2reg_pass_run_on_module(m, &r);
        return i.changed();
    });
    pre_cfg.add("dce", [](xir::Module *m, xir::PassReport &r) {
        auto i = xir::dce_pass_run_on_module(m, &r);
        return i.changed();
    });
    auto pre_cfg_stats = pre_cfg.run(xir_module.get());
    pre_cfg_stats.log("Fallback backend pre-CFG optimization");
    if (LUISA_SHOULD_DUMP_XIR) {
        auto filename = luisa::format("kernel.{:016x}.opt.xir", kernel.hash());
        std::ofstream f{filename.c_str()};
        f << xir::xir_to_text_translate(xir_module.get(), true);
    }
    xir::PassPipeline cfg;
    cfg.add("lower-ray-query-loop", [](xir::Module *m, xir::PassReport &r) {
        auto i = xir::lower_ray_query_loop_pass_run_on_module(m, &r);
        if (!i.succeeded()) {
            LUISA_ERROR_WITH_LOCATION(
                "Fallback XIR ray-query lowering rejected {} loop(s).",
                i.error_count);
        }
        return i.lowered_loop_count > 0u;
    });
    if (LUISA_XIR_NORMALIZE_CFG) {
        cfg.add("destructure-cfg", [](xir::Module *m, xir::PassReport &r) {
            auto i = xir::destructure_cfg_pass_run_on_module(m, &r);
            if (!i.succeeded()) {
                LUISA_ERROR_WITH_LOCATION(
                    "Fallback XIR destructuring failed (errors={}, leaked_blocks={}).",
                    i.error_count, i.leaked_block_count);
            }
            return i.changed();
        });
        cfg.add("simplify-cfg", [](xir::Module *m, xir::PassReport &r) {
            auto i = xir::simplify_cfg_pass_run_on_module(m, &r);
            return i.changed();
        });
        if (LUISA_XIR_RESTRUCTURE_CFG) {
            cfg.add("restructure-cfg", [](xir::Module *m, xir::PassReport &r) {
                auto i = xir::restructure_cfg_pass_run_on_module(m, &r);
                if (!i.succeeded()) {
                    LUISA_ERROR_WITH_LOCATION(
                        "Fallback XIR restructuring failed (irreducible={}, unstructured={}, invalid={}, iteration_limit={}).",
                        i.irreducible_region_count, i.unstructured_branch_count,
                        i.invalid_construct_count, i.iteration_limit_count);
                }
                return i.changed();
            });
        }
    }
    auto cfg_stats = cfg.run(xir_module.get());
    verify_xir_or_error(xir_module.get(), "codegen handoff");
    cfg_stats.log("Fallback backend CFG normalization");
    if (LUISA_XIR_NORMALIZE_CFG && LUISA_SHOULD_DUMP_XIR) {
        auto filename = luisa::format("kernel.{:016x}.norm.xir", kernel.hash());
        std::ofstream f{filename.c_str()};
        f << xir::xir_to_text_translate(xir_module.get(), true);
    }

    // dump for debugging
    if (LUISA_SHOULD_DUMP_XIR) {
        auto filename = luisa::format("kernel.{:016x}.opt.rq.xir", kernel.hash());
        std::ofstream f{filename.c_str()};
        f << xir::xir_to_text_translate(xir_module.get(), true);
    }

    auto llvm_ctx = std::make_unique<llvm::LLVMContext>();
    auto builtin_module = fallback_backend_device_builtin_module();
    llvm::SMDiagnostic parse_error;
    auto llvm_module = llvm::parseIR(llvm::MemoryBufferRef{builtin_module, ""}, parse_error, *llvm_ctx);
    if (!llvm_module) {
        LUISA_ERROR_WITH_LOCATION("Failed to generate LLVM IR: {}.",
                                  luisa::string_view{parse_error.getMessage()});
    }

    Clock codegen_clk;
    auto codegen_feedback = luisa_fallback_backend_codegen(*llvm_ctx, llvm_module.get(), xir_module.get());
    LUISA_VERBOSE("XIR to LLVM IR code generation done in {} ms.", codegen_clk.toc());

    if (llvm::verifyModule(*llvm_module, &llvm::errs())) {
        auto filename = luisa::format("kernel.{:016x}.err.ll", kernel.hash());
        std::error_code ec;
        llvm::raw_fd_ostream ofs{llvm::StringRef{filename}, ec};
        if (ec) {
            LUISA_ERROR_WITH_LOCATION("LLVM module verification failed. Failed to open file for dumping LLVM IR: {}.", ec.message());
        }
        llvm_module->print(ofs, nullptr, false, true);
        LUISA_ERROR_WITH_LOCATION("LLVM module verification failed. IR dumped to '{}'.", filename);
    }

    define_codegen_symbols(codegen_feedback);

    llvm_module->setDataLayout(_target_machine->createDataLayout());
#if LLVM_VERSION_MAJOR >= 21
    llvm_module->setTargetTriple(_target_machine->getTargetTriple());
#else
    llvm_module->setTargetTriple(_target_machine->getTargetTriple().str());
#endif

    // add fast-math flags to instructions
    for (auto &&f : *llvm_module) {
        for (auto &&bb : f) {
            for (auto &&inst : bb) {
                if (llvm::isa<llvm::FPMathOperator>(inst)) {
                    inst.setFast(option.enable_fast_math);
                }
            }
        }
    }

    if (LUISA_SHOULD_DUMP_LLVM_IR) {
        auto filename = luisa::format("kernel.{:016x}.ll", kernel.hash());
        std::error_code ec;
        llvm::raw_fd_ostream ofs{llvm::StringRef{filename}, ec};
        if (ec) {
            LUISA_WARNING_WITH_LOCATION("Failed to open file for dumping LLVM IR: {}.", ec.message());
        } else {
            llvm_module->print(ofs, nullptr, false, true);
        }
    }

    // optimize
    ::llvm::LoopAnalysisManager LAM;
    ::llvm::FunctionAnalysisManager FAM;
    ::llvm::CGSCCAnalysisManager CGAM;
    ::llvm::ModuleAnalysisManager MAM;
    ::llvm::PipelineTuningOptions PTO;
    std::size_t largest_function_instruction_count = 0u;
    for (const auto &function : *llvm_module) {
        std::size_t instruction_count = 0u;
        for (const auto &block : function) {
            instruction_count += block.size();
        }
        largest_function_instruction_count = std::max(
            largest_function_instruction_count,
            instruction_count);
    }
    // LLVM's O3 pipeline has superlinear compile-time behavior on very large
    // generated dispatch functions (notably in SROA and SLP vectorization).
    // Keep it for normal kernels, but use the minimal O0 pipeline once a
    // single function is large enough that optimization time and memory
    // dominate execution-time savings. Keep the IR pipeline at O0, but retain
    // the O1 machine-code pipeline: LLVM's x86 O0 code generator can fold
    // four-byte-aligned scalar spill slots into aligned vector loads in very
    // large functions. O1 uses SelectionDAG with the optimizing register
    // allocator while avoiding the superlinear IR transforms above.
    const auto use_minimal_optimization =
        largest_function_instruction_count >
        LUISA_FALLBACK_OPTIMIZATION_INSTRUCTION_LIMIT;
    if (use_minimal_optimization) {
        LUISA_WARNING_WITH_LOCATION(
            "Fallback LLVM function has {} instructions; using the minimal "
            "O0 IR pipeline and O1 machine-code generation above the "
            "{}-instruction scalability limit.",
            largest_function_instruction_count,
            LUISA_FALLBACK_OPTIMIZATION_INSTRUCTION_LIMIT);
        _target_machine->setOptLevel(::llvm::CodeGenOptLevel::Less);
    }
    PTO.LoopInterleaving = true;
#if LLVM_VERSION_MAJOR >= 21
    PTO.LoopInterchange = true;
#endif
    PTO.LoopVectorization = true;
    PTO.SLPVectorization = !use_minimal_optimization;
    PTO.LoopUnrolling = true;
    PTO.MergeFunctions = true;
    ::llvm::PassBuilder PB{_target_machine.get(), PTO};
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
#if LLVM_VERSION_MAJOR >= 19
    _target_machine->registerPassBuilderCallbacks(PB);
#else
    _target_machine->registerPassBuilderCallbacks(PB, true);
#endif
    Clock clk;
    clk.tic();
    auto MPM = use_minimal_optimization
                   ? PB.buildO0DefaultPipeline(::llvm::OptimizationLevel::O0)
                   : PB.buildPerModuleDefaultPipeline(::llvm::OptimizationLevel::O3);
    MPM.run(*llvm_module, MAM);

    LUISA_VERBOSE("Optimized LLVM module in {} ms.", clk.toc());
    if (::llvm::verifyModule(*llvm_module, &::llvm::errs())) {
        LUISA_ERROR_WITH_LOCATION("Failed to verify module.");
    }

    if (LUISA_SHOULD_DUMP_LLVM_IR) {
        auto filename = luisa::format("kernel.{:016x}.opt.ll", kernel.hash());
        std::error_code ec;
        llvm::raw_fd_ostream ofs{llvm::StringRef{filename}, ec};
        if (ec) {
            LUISA_WARNING_WITH_LOCATION("Failed to open file for dumping optimized LLVM IR: {}.", ec.message());
        } else {
            llvm_module->print(ofs, nullptr, false, true);
        }
    }

    if (LUISA_SHOULD_DUMP_ASM) {
        auto asm_name = luisa::format("kernel.{:016x}.s", kernel.hash());
        std::error_code ec;
        llvm::raw_fd_ostream ofs{llvm::StringRef{asm_name}, ec};
        if (ec) {
            LUISA_WARNING_WITH_LOCATION("Failed to open file for dumping assembly: {}.", ec.message());
        } else {
            llvm::legacy::PassManager pass;
            if (_target_machine->addPassesToEmitFile(pass, ofs, nullptr, llvm::CodeGenFileType::AssemblyFile)) {
                LUISA_ERROR_WITH_LOCATION("TheTargetMachine can't emit a file of this type");
            }
            pass.run(*llvm_module);
        }
    }

    // Persist the fully optimized, host-specific relocatable object. This is
    // the first representation that skips all expensive work on a cache hit:
    // AST-to-XIR, XIR passes, LLVM IR generation, O3, and machine codegen.
    if (cache_enabled) {
        Clock object_clock;
        llvm::SmallVector<char, 0u> object_code;
        llvm::raw_svector_ostream object_stream{object_code};
        llvm::legacy::PassManager object_pass;
        if (_target_machine->addPassesToEmitFile(
                object_pass, object_stream, nullptr,
                llvm::CodeGenFileType::ObjectFile)) {
            LUISA_ERROR_WITH_LOCATION(
                "Fallback target machine cannot emit object files.");
        }
        object_pass.run(*llvm_module);
        LUISA_VERBOSE(
            "Fallback machine object generated in {} ms ({} bytes).",
            object_clock.toc(), object_code.size());
        luisa::span<const std::byte> object_bytes{
            reinterpret_cast<const std::byte *>(object_code.data()),
            object_code.size()};
        static_cast<void>(
            device->binary_io()->write_shader_cache(
                cache_metadata.object_name, object_bytes));
        luisa::span<const std::byte> metadata_bytes{
            reinterpret_cast<const std::byte *>(
                cache_metadata.serialized.data()),
            cache_metadata.serialized.size()};
        // Metadata is written last. A partial object write therefore cannot
        // become a valid cache hit.
        static_cast<void>(
            device->binary_io()->write_shader_cache(
                cache_metadata.metadata_name, metadata_bytes));
        auto object_buffer =
            llvm::MemoryBuffer::getMemBufferCopy(
                llvm::StringRef{
                    object_code.data(), object_code.size()},
                cache_metadata.object_name.c_str());
        if (auto error =
                _jit->addObjectFile(std::move(object_buffer))) {
            auto message = llvm::toString(std::move(error));
            LUISA_ERROR_WITH_LOCATION(
                "Failed to add fallback machine object '{}': {}.",
                cache_metadata.object_name, message);
        }
    } else {
        auto module = llvm::orc::ThreadSafeModule(
            std::move(llvm_module), std::move(llvm_ctx));
        if (auto error = _jit->addIRModule(std::move(module))) {
            auto message = llvm::toString(std::move(error));
            LUISA_ERROR_WITH_LOCATION(
                "Failed to add fallback LLVM IR module: {}.",
                message);
        }
    }
    LUISA_ASSERT(
        lookup_kernel_entry(), "Fallback JIT compilation failed.");
}

class FallbackShaderDispatchBuffer {

public:
    struct alignas(16) Config {
        FallbackShader::kernel_entry_t *kernel{};
        std::array<uint, 3> dispatch_size;
        std::array<uint, 3> block_size;
    };

private:
    static constexpr auto argument_buffer_offset = sizeof(Config);// grid size
    std::byte *_data;

public:
    explicit FallbackShaderDispatchBuffer(size_t size) noexcept
        : _data{luisa::allocate_with_allocator<std::byte>(argument_buffer_offset + size)} {}
    ~FallbackShaderDispatchBuffer() noexcept {
        if (_data != nullptr) {
            luisa::deallocate_with_allocator(_data);
        }
    }
    FallbackShaderDispatchBuffer(FallbackShaderDispatchBuffer &&other) noexcept
        : _data{std::exchange(other._data, nullptr)} {}
    FallbackShaderDispatchBuffer(const FallbackShaderDispatchBuffer &) = delete;
    FallbackShaderDispatchBuffer &operator=(FallbackShaderDispatchBuffer &&) noexcept = delete;
    FallbackShaderDispatchBuffer &operator=(const FallbackShaderDispatchBuffer &) = delete;
    [[nodiscard]] auto argument_buffer() noexcept { return _data + argument_buffer_offset; }
    [[nodiscard]] auto argument_buffer() const noexcept { return const_cast<FallbackShaderDispatchBuffer *>(this)->argument_buffer(); }
    [[nodiscard]] auto config() noexcept { return reinterpret_cast<Config *>(_data); }
    [[nodiscard]] auto config() const noexcept { return const_cast<FallbackShaderDispatchBuffer *>(this)->config(); }
};

void FallbackShader::dispatch(FallbackCommandQueue *queue, luisa::unique_ptr<ShaderDispatchCommand> command) noexcept {

    auto dispatch_size = command->dispatch_size();
    auto block_size = _block_size;

    FallbackShaderDispatchBuffer dispatch_buffer{_argument_buffer_size};
    auto dispatch_config = dispatch_buffer.config();
    dispatch_config->kernel = _kernel_entry;
    dispatch_config->dispatch_size = {dispatch_size.x, dispatch_size.y, dispatch_size.z};
    dispatch_config->block_size = {block_size.x, block_size.y, block_size.z};

    auto argument_buffer = dispatch_buffer.argument_buffer();
    auto argument_buffer_offset = static_cast<size_t>(0u);
    auto allocate_argument = [&](size_t bytes) noexcept {
        static constexpr auto alignment = 16u;
        auto offset = (argument_buffer_offset + alignment - 1u) / alignment * alignment;
        LUISA_ASSERT(offset + bytes <= _argument_buffer_size,
                     "Too many arguments in ShaderDispatchCommand");
        argument_buffer_offset = offset + bytes;
        return argument_buffer + offset;
    };

    auto encode_argument = [&allocate_argument, &command](const auto &arg) noexcept {
        using Tag = ShaderDispatchCommand::Argument::Tag;
        switch (arg.tag) {
            case Tag::BUFFER: {
                auto buffer = reinterpret_cast<FallbackBuffer *>(arg.buffer.handle);
                auto buffer_view = buffer->view(arg.buffer.offset, arg.buffer.size);
                auto ptr = allocate_argument(sizeof(buffer_view));
                std::memcpy(ptr, &buffer_view, sizeof(buffer_view));
                break;
            }
            case Tag::TEXTURE: {
                auto texture = reinterpret_cast<const FallbackTexture *>(arg.texture.handle);
                auto view = texture->view(arg.texture.level);
                auto ptr = allocate_argument(sizeof(view));
                std::memcpy(ptr, &view, sizeof(view));
                break;
            }
            case Tag::UNIFORM: {
                auto uniform = command->uniform(arg.uniform);
                auto ptr = allocate_argument(uniform.size_bytes());
                std::memcpy(ptr, uniform.data(), uniform.size_bytes());
                break;
            }
            case Tag::BINDLESS_ARRAY: {
                auto bindless = reinterpret_cast<FallbackBindlessArray *>(arg.bindless_array.handle);
                auto view = bindless->view();
                auto ptr = allocate_argument(sizeof(view));
                std::memcpy(ptr, &view, sizeof(view));
                break;
            }
            case Tag::ACCEL: {
                auto accel = reinterpret_cast<FallbackAccel *>(arg.accel.handle);
                auto view = accel->view();
                auto ptr = allocate_argument(sizeof(view));
                std::memcpy(ptr, &view, sizeof(view));
                break;
            }
            default: LUISA_ERROR_WITH_LOCATION("Unsupported argument type.");
        }
    };
    for (auto &&arg : _bound_arguments) { encode_argument(arg); }
    for (auto &&arg : command->arguments()) { encode_argument(arg); }

    static constexpr auto roundup_div = [](auto a, auto b) noexcept {
        return (a + b - 1u) / b;
    };

    auto grid_size = roundup_div(dispatch_size, block_size);
    auto grid_count = grid_size.x * grid_size.y * grid_size.z;

    queue->enqueue_parallel(grid_count, [queue, dispatch_buffer = std::move(dispatch_buffer)](auto block) noexcept {
        auto config = dispatch_buffer.config();
        auto dispatch_size = config->dispatch_size;
        auto block_size = config->block_size;
        auto grid_size_x = roundup_div(dispatch_size[0], block_size[0]);
        auto grid_size_y = roundup_div(dispatch_size[1], block_size[1]);
        auto bx = block % grid_size_x;
        auto by = (block / grid_size_x) % grid_size_y;
        auto bz = block / (grid_size_x * grid_size_y);
        FallbackShaderLaunchConfig launch_config{
            .block_id = make_uint3(bx, by, bz),
            .dispatch_size = {dispatch_size[0], dispatch_size[1], dispatch_size[2]},
            .block_size = {block_size[0], block_size[1], block_size[2]},
        };
        auto launch_params = dispatch_buffer.argument_buffer();
        luisa_coro_reset_counter();
        current_device_log_callback = queue->log_callback() ? &queue->log_callback() : nullptr;
        config->kernel(launch_params, &launch_config);
        current_device_log_callback = nullptr;
    });
}

FallbackShader::~FallbackShader() noexcept = default;

void FallbackShader::_initialize_target_machine_jit(const ShaderOption &option) noexcept {

    // build JIT engine
    ::llvm::orc::LLJITBuilder jit_builder;
    if (auto host = ::llvm::orc::JITTargetMachineBuilder::detectHost()) {
        ::llvm::TargetOptions options;
        if (option.enable_fast_math) {
#if LLVM_VERSION_MAJOR <= 21
            options.UnsafeFPMath = true;
            options.ApproxFuncFPMath = true;
#endif
            options.NoInfsFPMath = true;
            options.NoNaNsFPMath = true;
            options.NoSignedZerosFPMath = true;
        }
        options.NoTrappingFPMath = true;
        options.AllowFPOpFusion = ::llvm::FPOpFusion::Fast;
        options.EnableIPRA = false;// true causes crash
        options.StackSymbolOrdering = true;
#ifndef NDEBUG
        options.TrapUnreachable = true;
#else
        options.TrapUnreachable = false;
#endif
        options.EnableMachineFunctionSplitter = true;
        options.EnableMachineOutliner = false;
        options.NoTrapAfterNoreturn = true;
        host->setOptions(options);
        host->setCodeGenOptLevel(::llvm::CodeGenOptLevel::Aggressive);
#ifdef __aarch64__
        host->addFeatures({"+neon"});
#endif
        LUISA_VERBOSE("LLVM JIT target: triplet = {}, features = {}.",
                      host->getTargetTriple().str(), host->getFeatures().getString());
        if (auto machine = host->createTargetMachine()) {
            _target_machine = std::move(machine.get());
        } else {
            ::llvm::handleAllErrors(machine.takeError(), [&](const ::llvm::ErrorInfoBase &e) {
                LUISA_WARNING_WITH_LOCATION("JITTargetMachineBuilder::createTargetMachine(): {}.", e.message());
            });
            LUISA_ERROR_WITH_LOCATION("Failed to create target machine.");
        }
        jit_builder.setJITTargetMachineBuilder(std::move(*host));
    } else {
        ::llvm::handleAllErrors(host.takeError(), [&](const ::llvm::ErrorInfoBase &e) {
            LUISA_WARNING_WITH_LOCATION("JITTargetMachineBuilder::detectHost(): {}.", e.message());
        });
        LUISA_ERROR_WITH_LOCATION("Failed to detect host.");
    }

    if (auto expected_jit = jit_builder.create()) {
        _jit = std::move(expected_jit.get());
    } else {
        ::llvm::handleAllErrors(expected_jit.takeError(), [](const ::llvm::ErrorInfoBase &err) {
            LUISA_WARNING_WITH_LOCATION("LLJITBuilder::create(): {}", err.message());
        });
        LUISA_ERROR_WITH_LOCATION("Failed to create LLJIT.");
    }
}

void FallbackShader::_build_bound_arguments(luisa::span<const Function::Binding> bindings) noexcept {
    _bound_arguments.reserve(bindings.size());
    for (auto &&arg : bindings) {
        luisa::visit(
            [&]<typename T>(T binding) noexcept {
                ShaderDispatchCommand::Argument argument{};
                if constexpr (std::is_same_v<T, Function::BufferBinding>) {
                    argument.tag = ShaderDispatchCommand::Argument::Tag::BUFFER;
                    argument.buffer.handle = binding.handle;
                    argument.buffer.offset = binding.offset;
                    argument.buffer.size = binding.size;
                } else if constexpr (std::is_same_v<T, Function::TextureBinding>) {
                    argument.tag = ShaderDispatchCommand::Argument::Tag::TEXTURE;
                    argument.texture.handle = binding.handle;
                    argument.texture.level = binding.level;
                } else if constexpr (std::is_same_v<T, Function::BindlessArrayBinding>) {
                    argument.tag = ShaderDispatchCommand::Argument::Tag::BINDLESS_ARRAY;
                    argument.bindless_array.handle = binding.handle;
                } else if constexpr (std::is_same_v<T, Function::AccelBinding>) {
                    argument.tag = ShaderDispatchCommand::Argument::Tag::ACCEL;
                    argument.accel.handle = binding.handle;
                } else {
                    LUISA_ERROR_WITH_LOCATION("Unsupported binding type.");
                }
                _bound_arguments.emplace_back(argument);
            },
            arg);
    }
}

}// namespace luisa::compute::fallback
