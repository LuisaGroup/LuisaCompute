#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Analysis/AliasAnalysis.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/IR/LegacyPassManager.h>

#include <luisa/core/logging.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/stream.h>

#include "fallback_buffer.h"
#include "fallback_texture.h"
#include "fallback_stream.h"
#include "fallback_command_queue.h"
#include "fallback_tex_compress.h"

namespace luisa::compute::fallback {

namespace {

#ifdef LUISA_ARCH_ARM64
#include "fallback_builtin/fallback_tex_compress.arm64.inl"
#else
#include "fallback_builtin/fallback_tex_compress.x86_64.inl"
#endif

struct BC6HEncodeConfig {
    bool slow_mode;
    bool fast_mode;
    int refineIterations_1p;
    int refineIterations_2p;
    int fastSkipTreshold;
};

struct BC7EncodeConfig {
    bool mode_selection[4];
    int refineIterations[8];

    bool skip_mode2;
    int fastSkipTreshold_mode1;
    int fastSkipTreshold_mode3;
    int fastSkipTreshold_mode7;

    int mode45_channel0;
    int refineIterations_channel;

    int channels;
};

[[nodiscard]] auto make_bc7_encode_config() noexcept {
    BC7EncodeConfig config{};
    config.channels = 3;

    int moreRefine = 2;
    // mode02
    config.mode_selection[0] = true;
    config.skip_mode2 = false;

    config.refineIterations[0] = 2 + moreRefine;
    config.refineIterations[2] = 2 + moreRefine;

    // mode13
    config.mode_selection[1] = true;
    config.fastSkipTreshold_mode1 = 64;
    config.fastSkipTreshold_mode3 = 64;
    config.fastSkipTreshold_mode7 = 0;

    config.refineIterations[1] = 2 + moreRefine;
    config.refineIterations[3] = 2 + moreRefine;

    // mode45
    config.mode_selection[2] = true;

    config.mode45_channel0 = 0;
    config.refineIterations_channel = 2 + moreRefine;
    config.refineIterations[4] = 2 + moreRefine;
    config.refineIterations[5] = 2 + moreRefine;

    // mode6
    config.mode_selection[3] = true;

    config.refineIterations[6] = 2 + moreRefine;

    return config;
}

[[nodiscard]] auto make_bc6h_encode_config() noexcept {
    BC6HEncodeConfig config{};
    config.slow_mode = true;
    config.fast_mode = false;
    config.fastSkipTreshold = 32;
    config.refineIterations_1p = 2;
    config.refineIterations_2p = 2;
    return config;
}

struct RGBASurface {
    std::byte *ptr;
    uint32_t width;
    uint32_t height;
    uint32_t stride;// in bytes
};

using func_type_bc6h_encode_blocks = void(const RGBASurface &src, std::byte *dst, const BC6HEncodeConfig &config);
using func_type_bc7_encode_blocks = void(const RGBASurface &src, std::byte *dst, const BC7EncodeConfig &config);

template<typename T>
[[nodiscard]] auto create_rgba_surface(FallbackCommandQueue &queue, ImageView<float> src) noexcept {
    auto size = src.size();
    auto padded_size = (size + 3u) / 4u * 4u;
    auto memory = luisa::allocate_with_allocator<luisa::Vector<T, 4u>>(padded_size.x * padded_size.y);
    auto view = reinterpret_cast<FallbackTexture *>(src.handle())->view(src.level());
    static constexpr auto scanline_height = 4u;
    auto scanline_count = (size.y + scanline_height - 1u) / scanline_height;
    queue.enqueue_parallel(scanline_count, [view, memory, padded_w = padded_size.x](uint32_t scanline) noexcept {
        auto size = view.size2d();
        auto y_begin = scanline * scanline_height;
        auto dst = memory + y_begin * padded_w;
        for (auto y = y_begin; y < y_begin + scanline_height; y++) {
            for (auto x = 0u; x < size.x; x++) {
                *(dst++) = view.template read2d<T>(luisa::min(make_uint2(x, y), size - 1u));
            }
        }
    });
    return RGBASurface{.ptr = reinterpret_cast<std::byte *>(memory),
                       .width = padded_size.x,
                       .height = padded_size.y,
                       .stride = static_cast<uint>(padded_size.x * sizeof(T) * 4u)};
}

template<typename Func, typename Config>
void run_bc_encode_shader(Func *f, FallbackCommandQueue &queue, const RGBASurface &src, std::byte *dst, const Config &config) noexcept {
    constexpr auto block_size = 4u;
    LUISA_DEBUG_ASSERT(src.width % block_size == 0 && src.height % block_size == 0, "Invalid image size for BC compression.");
    auto task_count = src.height / block_size;
    queue.enqueue_parallel(task_count, [=](uint block_y) noexcept {
        RGBASurface line_src{.ptr = src.ptr + block_y * src.stride * block_size,
                             .width = src.width,
                             .height = block_size,
                             .stride = src.stride};
        auto line_dst = dst + block_y * (src.width / block_size) * sizeof(uint4);
        f(line_src, line_dst, config);
    });
}

}// namespace

class FallbackTexCompressExt final : public FallbackTexCompressInterface {

public:
    FallbackTexCompressExt() noexcept {
        ::llvm::orc::LLJITBuilder jit_builder;
        if (auto host = ::llvm::orc::JITTargetMachineBuilder::detectHost()) {
            ::llvm::TargetOptions options;
#if LLVM_VERSION_MAJOR <= 21
            options.UnsafeFPMath = true;
            options.ApproxFuncFPMath = true;
#endif
#if LLVM_VERSION_MAJOR <= 22
            options.NoInfsFPMath = true;
            options.NoNaNsFPMath = true;
            options.NoSignedZerosFPMath = true;
#endif
            options.NoTrappingFPMath = true;
            options.AllowFPOpFusion = ::llvm::FPOpFusion::Fast;
            options.EnableIPRA = false;// true causes crash
            options.StackSymbolOrdering = true;
            options.TrapUnreachable = false;
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

        auto llvm_ctx = std::make_unique<llvm::LLVMContext>();
        llvm::MemoryBufferRef llvm_bc{llvm::StringRef{reinterpret_cast<const char *>(fallback_tex_compress_llvm_bc),
                                                      std::size(fallback_tex_compress_llvm_bc)},
                                      "fallback_tex_compress_bc"};
        llvm::SMDiagnostic parse_error;
        auto llvm_module = llvm::parseIR(llvm_bc, parse_error, *llvm_ctx);
        if (!llvm_module) {
            LUISA_ERROR_WITH_LOCATION("Failed to generate LLVM IR: {}.",
                                      luisa::string_view{parse_error.getMessage()});
        }
        llvm_module->setDataLayout(_target_machine->createDataLayout());
#if LLVM_VERSION_MAJOR >= 21
        llvm_module->setTargetTriple(_target_machine->getTargetTriple());
#else
        llvm_module->setTargetTriple(_target_machine->getTargetTriple().str());
#endif

        // compile to machine code
        auto m = llvm::orc::ThreadSafeModule(std::move(llvm_module), std::move(llvm_ctx));
        if (auto error = _jit->addIRModule(std::move(m))) {
            ::llvm::handleAllErrors(std::move(error), [](const ::llvm::ErrorInfoBase &err) {
                LUISA_WARNING_WITH_LOCATION("LLJIT::addIRModule(): {}", err.message());
            });
        }
        _bc6h_encode_blocks = _jit->lookup(fallback_tex_compress_llvm_bc_bc6h_kernel_name)
                                  ->toPtr<func_type_bc6h_encode_blocks>();
        _bc7_encode_blocks = _jit->lookup(fallback_tex_compress_llvm_bc_bc7_kernel_name)
                                 ->toPtr<func_type_bc7_encode_blocks>();
    }
    std::unique_ptr<::llvm::orc::LLJIT> _jit;
    std::unique_ptr<::llvm::TargetMachine> _target_machine;
    func_type_bc6h_encode_blocks *_bc6h_encode_blocks;
    func_type_bc7_encode_blocks *_bc7_encode_blocks;

public:
    Result check_builtin_shader() noexcept override { return TexCompressExt::Result::Success; }
    Result compress_bc6h(Stream &stream, const ImageView<float> &src, const BufferView<uint> &result) noexcept override {
        auto queue = reinterpret_cast<FallbackStream *>(stream.handle())->queue();
        auto dst = reinterpret_cast<FallbackBuffer *>(result.handle())->data();
        auto surface = create_rgba_surface<luisa::half>(*queue, src);
        run_bc_encode_shader(_bc6h_encode_blocks, *queue, surface, dst, make_bc6h_encode_config());
        queue->enqueue([p = surface.ptr] { luisa::deallocate_with_allocator(p); });
        return Result::Success;
    }
    Result compress_bc7(Stream &stream, const ImageView<float> &src, const BufferView<uint> &result, float alpha_importance) noexcept override {
        auto queue = reinterpret_cast<FallbackStream *>(stream.handle())->queue();
        auto dst = reinterpret_cast<FallbackBuffer *>(result.handle())->data();
        auto surface = create_rgba_surface<uint8_t>(*queue, src);
        run_bc_encode_shader(_bc7_encode_blocks, *queue, surface, dst, make_bc7_encode_config());
        queue->enqueue([p = surface.ptr] { luisa::deallocate_with_allocator(p); });
        return Result::Success;
    }
};

luisa::unique_ptr<FallbackTexCompressInterface>
create_fallback_tex_compress_ext() noexcept {
    return luisa::make_unique<FallbackTexCompressExt>();
}

}// namespace luisa::compute::fallback
