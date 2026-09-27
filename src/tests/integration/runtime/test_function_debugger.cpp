// Debug function generator (FunctionDebugger) tests.
//
// The debug generator rewrites a kernel (and its callables) so that every
// operation that can fail at runtime is guarded: on failure a diagnostic is
// printed through the device printer (captured here through the stream's log
// callback) and the thread stops early. Failures inside callables propagate an
// error code up to the root kernel. See
// include/luisa/ast/function_builder_debugger.h and
// src/ast/function_builder_debugger.cpp.
//
// Run with an explicit backend, e.g.: test_function_debugger cuda
// (the acceleration-structure tests additionally require a ray-tracing
// capable device: cuda, dx or vk).

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/luisa-compute.h>
#include <luisa/ast/function_builder_debugger.h>
#include <luisa/ast/function.h>
#include <limits>
#include <mutex>
#include <string>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

struct LogCapture {
    std::mutex mutex;
    luisa::vector<luisa::string> messages;
};

/// Run `body` on a stream whose device messages are captured, and return the
/// captured messages (severity prefix stripped).
[[nodiscard]] luisa::vector<luisa::string>
run_with_log_capture(Device &device,
                     luisa::function<void(Stream &)> const &body) {
    auto capture = luisa::make_shared<LogCapture>();
    auto stream = device.create_stream();
    stream.set_log_callback([capture](luisa::string_view message) noexcept {
        auto payload = message.empty() ? message : message.substr(1u);
        std::scoped_lock lock{capture->mutex};
        capture->messages.emplace_back(luisa::string{payload});
    });
    body(stream);
    stream << synchronize();
    std::scoped_lock lock{capture->mutex};
    return capture->messages;
}

[[nodiscard]] size_t count_messages_with(
    luisa::vector<luisa::string> const &messages,
    luisa::string_view needle) noexcept {
    size_t count = 0u;
    for (auto &message : messages) {
        if (message.find(needle) != luisa::string::npos) { ++count; }
    }
    return count;
}

// Recursively count the statements of a scope (the debug generator must not
// alter the statement count of a function without checkable operations).
[[nodiscard]] size_t count_statements(const ScopeStmt *scope) noexcept {
    size_t count = 0u;
    for (auto s : scope->statements()) {
        ++count;
        switch (s->tag()) {
            case Statement::Tag::IF: {
                auto x = static_cast<const IfStmt *>(s);
                count += count_statements(x->true_branch()) +
                         count_statements(x->false_branch());
                break;
            }
            case Statement::Tag::LOOP:
                count += count_statements(
                    static_cast<const LoopStmt *>(s)->body());
                break;
            case Statement::Tag::SCOPE:
                count += count_statements(static_cast<const ScopeStmt *>(s));
                break;
            case Statement::Tag::SWITCH:
                count += count_statements(
                    static_cast<const SwitchStmt *>(s)->body());
                break;
            case Statement::Tag::SWITCH_CASE:
            case Statement::Tag::SWITCH_CASE_GROUP:
                count += count_statements(
                    static_cast<const SwitchCaseStmt *>(s)->body());
                break;
            case Statement::Tag::SWITCH_DEFAULT:
                count += count_statements(
                    static_cast<const SwitchDefaultStmt *>(s)->body());
                break;
            case Statement::Tag::FOR:
                count += count_statements(
                    static_cast<const ForStmt *>(s)->body());
                break;
            case Statement::Tag::RAY_QUERY: {
                auto x = static_cast<const RayQueryStmt *>(s);
                count += count_statements(x->on_triangle_candidate()) +
                         count_statements(x->on_procedural_candidate());
                break;
            }
            case Statement::Tag::AUTO_DIFF:
                count += count_statements(
                    static_cast<const AutoDiffStmt *>(s)->body());
                break;
            default: break;
        }
    }
    return count;
}

[[nodiscard]] bool backend_supports_rtx(const Device &device) noexcept {
    auto name = device.backend_name();
    return name == "cuda" || name == "dx" || name == "vk" || name == "metal";
}

constexpr auto contains_buffer_index = "buffer index out of range"sv;
constexpr auto contains_byte_buffer = "byte-buffer range out of bounds"sv;
constexpr auto contains_bindless_index = "bindless element index out of range"sv;
constexpr auto contains_nan_inf = "NaN/Inf result"sv;
constexpr auto contains_float_div_zero = "float division by zero"sv;
constexpr auto contains_int_div_zero = "integer division/modulo by zero"sv;
constexpr auto contains_shift = "shift amount out of range"sv;
constexpr auto contains_accel_index = "accel instance index out of range"sv;
constexpr auto contains_callee_failed = "failed in"sv;

}// namespace

void test_function_debugger(Device &device) {

    // ------------------------------------------------------------------
    // 1. BUFFER_READ out of range: log + early stop (the sentinel write
    //    after the guarded read must not execute for faulting threads).
    // ------------------------------------------------------------------
    {
          constexpr auto thread_count = 4u;
          auto buffer = device.create_buffer<float>(8u);
          auto output = device.create_buffer<uint>(thread_count);
          auto kernel = add_debug_checks(Kernel1D{[&](BufferVar<float> buf,
                                                     BufferVar<uint> out) noexcept {
            auto index = dispatch_id().x;
            auto value = buf->read(index * 100u);// threads 1..3 out of range
            out->write(index, 1u);               // sentinel
            static_cast<void>(value);
        }});
        auto shader = device.compile(kernel, {.enable_debug_info = true});
        luisa::vector<uint> output_host(thread_count, 0u);
        luisa::vector<float> input_host(8u, 1.f);
        auto messages = run_with_log_capture(device, [&](Stream &stream) {
            stream << buffer.copy_from(luisa::span{input_host})
                   << shader(buffer, output).dispatch(thread_count)
                   << output.copy_to(luisa::span{output_host})
                   << synchronize();
        });
        expect(count_messages_with(messages, contains_buffer_index) >= 1u)
            << "buffer-read guard did not report";
        expect(output_host[0] == 1u) << "in-range thread must run past the guard";
        expect(output_host[1] == 0u && output_host[2] == 0u &&
               output_host[3] == 0u)
            << "out-of-range threads must stop at the guard";
    }

    // ------------------------------------------------------------------
    // 2. BUFFER_WRITE out of range: no write occurs.
    // ------------------------------------------------------------------
    {
        auto buffer = device.create_buffer<uint>(4u);
        auto kernel = add_debug_checks(Kernel1D{[&](BufferVar<uint> buf) noexcept {
            auto index = dispatch_id().x;
            buf->write(index * 100u, 0xDEADBEEFu);// thread 1 out of range
        }});
        auto shader = device.compile(kernel, {.enable_debug_info = true});
        luisa::vector<uint> host(4u, 0x12345678u);
        auto messages = run_with_log_capture(device, [&](Stream &stream) {
            stream << buffer.copy_from(luisa::span{host})
                   << shader(buffer).dispatch(2u)
                   << buffer.copy_to(luisa::span{host})
                   << synchronize();
        });
        expect(count_messages_with(messages, contains_buffer_index) >= 1u)
            << "buffer-write guard did not report";
        expect(host[1] == 0x12345678u) << "out-of-range write must not happen";
    }

  // ------------------------------------------------------------------
  // 3. BYTE_BUFFER_READ / _WRITE out of range (a byte buffer of 32
  //    bytes; offsets 0 and 100, the latter out of range).
  // ------------------------------------------------------------------
  {
      auto buffer = device.create_byte_buffer(32u);
      auto output = device.create_buffer<uint>(2u);
      auto read_kernel = add_debug_checks(Kernel1D{[&](ByteBufferVar buf,
                                                        BufferVar<uint> out) noexcept {
          auto index = dispatch_id().x;
          auto value = buf.template read<float>(index * 100u);
          out->write(index, 1u);
          static_cast<void>(value);
      }});
      auto shader = device.compile(read_kernel, {.enable_debug_info = true});
      luisa::vector<uint> output_host(2u, 0u);
      luisa::vector<float> input_host(8u, 1.f);
      auto messages = run_with_log_capture(device, [&](Stream &stream) {
          stream << buffer.copy_from(input_host.data())
                 << shader(buffer, output).dispatch(2u)
                 << output.copy_to(luisa::span{output_host})
                 << synchronize();
      });
      expect(count_messages_with(messages, contains_byte_buffer) >= 1u)
          << "byte-buffer read guard did not report";
      expect(output_host[0] == 1u && output_host[1] == 0u)
          << "out-of-range byte read must stop the thread";

      auto write_kernel = add_debug_checks(Kernel1D{[&](ByteBufferVar buf) noexcept {
          auto index = dispatch_id().x;
          buf.write(index * 100u, 1.0f);
      }});
      auto write_shader = device.compile(write_kernel, {.enable_debug_info = true});
      luisa::vector<float> host(8u, 0.f);
      auto write_messages = run_with_log_capture(device, [&](Stream &stream) {
          stream << buffer.copy_from(host.data())
                 << write_shader(buffer).dispatch(2u)
                 << buffer.copy_to(host.data())
                 << synchronize();
      });
      expect(count_messages_with(write_messages, contains_byte_buffer) >= 1u)
          << "byte-buffer write guard did not report";
      expect(host[2] == 0.f && host[3] == 0.f)
          << "out-of-range byte write must not happen";
  }

    // ------------------------------------------------------------------
    // 4. Float ADD with a NaN input: NaN/Inf result check.
    // ------------------------------------------------------------------
    {
        auto values = device.create_buffer<float>(2u);
        auto output = device.create_buffer<uint>(2u);
        auto kernel = add_debug_checks(Kernel1D{[&](BufferVar<float> in,
                                                    BufferVar<uint> out) noexcept {
            auto index = dispatch_id().x;
            auto x = in->read(index);
            auto y = x + 1.0f;// NaN + 1 = NaN
            out->write(index, 1u);
            static_cast<void>(y);
        }});
        auto shader = device.compile(kernel, {.enable_debug_info = true});
        luisa::vector<float> input_host{1.f, std::numeric_limits<float>::quiet_NaN()};
        luisa::vector<uint> output_host(2u, 0u);
        auto messages = run_with_log_capture(device, [&](Stream &stream) {
            stream << values.copy_from(luisa::span{input_host})
                   << shader(values, output).dispatch(2u)
                   << output.copy_to(luisa::span{output_host})
                   << synchronize();
        });
        expect(count_messages_with(messages, contains_nan_inf) >= 1u)
            << "NaN propagation through ADD did not report";
        expect(output_host[0] == 1u && output_host[1] == 0u)
            << "thread producing NaN must stop at the guard";
    }

    // ------------------------------------------------------------------
    // 5. Float DIV by zero and int DIV by zero.
    // ------------------------------------------------------------------
    {
        auto output = device.create_buffer<uint>(2u);
        auto kernel = add_debug_checks(Kernel1D{[&](BufferVar<uint> out) noexcept {
            auto index = dispatch_id().x;
            Float x = 1.0f;
            Int n = 1;
            // thread 0 faults on the float division, thread 1 on the integer
            // division (the first guard stops the thread, so each thread may
            // only fault once)
            auto divisor = ite(index == 0u, 0.0f, 1.0f);
            auto y = x / divisor;
            auto int_divisor = ite(index == 0u, 1, 0);
            auto m = n / int_divisor;
            out->write(index, 1u);
            static_cast<void>(y);
            static_cast<void>(m);
        }});
        auto shader = device.compile(kernel, {.enable_debug_info = true});
        luisa::vector<uint> output_host(2u, 0u);
        auto messages = run_with_log_capture(device, [&](Stream &stream) {
            stream << output.copy_from(luisa::span{output_host})
                   << shader(output).dispatch(2u)
                   << output.copy_to(luisa::span{output_host})
                   << synchronize();
        });
        expect(count_messages_with(messages, contains_float_div_zero) >= 1u)
            << "float division by zero did not report";
        expect(count_messages_with(messages, contains_int_div_zero) >= 1u)
            << "integer division by zero did not report";
        expect(output_host[0] == 0u && output_host[1] == 0u)
            << "both division-by-zero threads must stop at the guard";
    }

    // ------------------------------------------------------------------
    // 6. Int MOD by zero.
    // ------------------------------------------------------------------
    {
        auto output = device.create_buffer<uint>(2u);
        auto kernel = add_debug_checks(Kernel1D{[&](BufferVar<uint> out) noexcept {
            auto index = dispatch_id().x;
            Int n = 7;
            auto divisor = ite(index == 0u, 0, 3);
            auto m = n % divisor;
            out->write(index, 1u);
            static_cast<void>(m);
        }});
        auto shader = device.compile(kernel, {.enable_debug_info = true});
        luisa::vector<uint> output_host(2u, 0u);
        auto messages = run_with_log_capture(device, [&](Stream &stream) {
            stream << output.copy_from(luisa::span{output_host})
                   << shader(output).dispatch(2u)
                   << output.copy_to(luisa::span{output_host})
                   << synchronize();
        });
        expect(count_messages_with(messages, contains_int_div_zero) >= 1u)
            << "modulo by zero did not report";
        expect(output_host[0] == 0u && output_host[1] == 1u);
    }

    // ------------------------------------------------------------------
    // 7. SHL / SHR with a shift amount >= bit width.
    // ------------------------------------------------------------------
    {
        auto output = device.create_buffer<uint>(4u);
        auto kernel = add_debug_checks(Kernel1D{[&](BufferVar<uint> out) noexcept {
            auto index = dispatch_id().x;
            UInt x = 1u;
            auto amount = index * 16u;// 0, 16 (ok), 32, 48 (bad)
            auto shl = x << amount;
            out->write(index, 1u);
            static_cast<void>(shl);
        }});
        auto shader = device.compile(kernel, {.enable_debug_info = true});
        luisa::vector<uint> output_host(4u, 0u);
        auto messages = run_with_log_capture(device, [&](Stream &stream) {
            stream << output.copy_from(luisa::span{output_host})
                   << shader(output).dispatch(4u)
                   << output.copy_to(luisa::span{output_host})
                   << synchronize();
        });
        expect(count_messages_with(messages, contains_shift) >= 1u)
            << "shift-amount guard did not report";
        expect(output_host[0] == 1u && output_host[1] == 1u &&
               output_host[2] == 0u && output_host[3] == 0u)
            << "threads with shift >= 32 must stop at the guard";
    }

    // ------------------------------------------------------------------
    // 8. SQRT / LOG / ACOS domain errors.
    // ------------------------------------------------------------------
    {
        auto output = device.create_buffer<uint>(3u);
        auto kernel = add_debug_checks(Kernel1D{[&](BufferVar<uint> out) noexcept {
            auto index = dispatch_id().x;
            Float x = -1.0f;
            Float bad = 2.0f;
            auto y = sqrt(x);// thread 0
            auto z = log2(make_float2(0.0f, 1.0f));// thread 1 (log2(0) = -inf)
            auto w = acos(bad);// thread 2
            out->write(index, 1u);
            static_cast<void>(y);
            static_cast<void>(z);
            static_cast<void>(w);
        }});
        auto shader = device.compile(kernel, {.enable_debug_info = true});
        luisa::vector<uint> output_host(3u, 0u);
        auto messages = run_with_log_capture(device, [&](Stream &stream) {
            stream << output.copy_from(luisa::span{output_host})
                   << shader(output).dispatch(3u)
                   << output.copy_to(luisa::span{output_host})
                   << synchronize();
        });
        expect(count_messages_with(messages, contains_nan_inf) >= 1u)
            << "math domain errors did not report";
        expect(output_host[0] == 0u && output_host[1] == 0u &&
               output_host[2] == 0u)
            << "domain-error threads must stop at the guard";
    }

    // ------------------------------------------------------------------
    // 9. NORMALIZE: a NaN input propagates into a NaN result (a zero-length
    //    vector is flushed to zero by some fast-math toolchains, so the NaN
    //    input is the portable way to reach the check).
    // ------------------------------------------------------------------
    {
        auto values = device.create_buffer<float>(3u);
        auto output = device.create_buffer<uint>(1u);
        auto kernel = add_debug_checks(Kernel1D{[&](BufferVar<float> in,
                                                       BufferVar<uint> out) noexcept {
            auto v = normalize(make_float3(in->read(0u), in->read(1u), in->read(2u)));
            out->write(0u, 1u);
            static_cast<void>(v);
        }});
        auto shader = device.compile(kernel, {.enable_debug_info = true});
        luisa::vector<float> input_host{
            std::numeric_limits<float>::quiet_NaN(), 0.f, 0.f};
        luisa::vector<uint> output_host(1u, 0u);
          auto messages = run_with_log_capture(device, [&](Stream &stream) {
              stream << values.copy_from(luisa::span{input_host})
                     << output.copy_from(luisa::span{output_host})
                     << shader(values, output).dispatch(1u)
                     << output.copy_to(luisa::span{output_host})
                     << synchronize();
          });
          expect(count_messages_with(messages, contains_nan_inf) >= 1u)
              << "normalize(NaN) did not report";
          expect(output_host[0] == 0u);
    }

    // ------------------------------------------------------------------
    // 10. Nested callable error propagation: a fault in the innermost
    //     callable stops the whole kernel thread and is reported twice on
    //     the way up (callee checks at both call sites).
    // ------------------------------------------------------------------
    {
        Callable<float(float)> inner = [&](Float x) noexcept -> Float {
            return 1.0f / x;// division by zero when x == 0
        };
        Callable<float(float)> middle = [&](Float x) noexcept -> Float {
            return inner(x) + 1.0f;
        };
        auto output = device.create_buffer<float>(1u);
        auto kernel = add_debug_checks(Kernel1D{[&](BufferVar<float> out) noexcept {
            out->write(0u, middle(0.0f));
        }});
          auto shader = device.compile(kernel, {.enable_debug_info = true});
          luisa::vector<float> output_host(1u, 42.f);
          auto messages = run_with_log_capture(device, [&](Stream &stream) {
              stream << output.copy_from(luisa::span{output_host})
                     << shader(output).dispatch(1u)
                     << output.copy_to(luisa::span{output_host})
                     << synchronize();
          });
        expect(count_messages_with(messages, contains_float_div_zero) >= 1u)
            << "inner callable did not report the division";
        expect(count_messages_with(messages, contains_callee_failed) >= 1u)
            << "the error code did not propagate to the callers";
        expect(output_host[0] == 42.f)
            << "kernel must not run past the propagated failure";
    }

    // ------------------------------------------------------------------
    // 11. BINDLESS element index out of range.
    // ------------------------------------------------------------------
    {
        BindlessArray heap = device.create_bindless_array();
        auto buffer = device.create_buffer<float>(8u);
        auto output = device.create_buffer<uint>(2u);
        auto kernel = add_debug_checks(Kernel1D{[&](BindlessVar array,
                                                     BufferVar<uint> out) noexcept {
            auto index = dispatch_id().x;
            auto value = array->buffer<float>(0u).read(index * 100u);
            out->write(index, 1u);
            static_cast<void>(value);
        }});
        auto shader = device.compile(kernel, {.enable_debug_info = true});
        luisa::vector<uint> output_host(2u, 0u);
        luisa::vector<float> input_host(8u, 1.f);
        auto messages = run_with_log_capture(device, [&](Stream &stream) {
            heap.emplace_on_update(0u, buffer.view(0u, 8u));
            stream << heap.update()
                   << buffer.copy_from(luisa::span{input_host})
                   << shader(heap, output).dispatch(2u)
                   << output.copy_to(luisa::span{output_host})
                   << synchronize();
        });
        expect(count_messages_with(messages, contains_bindless_index) >= 1u)
            << "bindless element-index guard did not report";
        expect(output_host[0] == 1u && output_host[1] == 0u);
    }

    // ------------------------------------------------------------------
    // 12. buffer.size(): correct element count (host-injected bound on
    //     dx; native queries elsewhere).
    // ------------------------------------------------------------------
    {
        auto buffer = device.create_buffer<float>(7u);
        auto output = device.create_buffer<uint>(1u);
        auto kernel = Kernel1D{[&](BufferVar<float> buf,
                                   BufferVar<uint> out) noexcept {
            out->write(0u, buf->size());
        }};
        // The dx implementation of BUFFER_SIZE reads the host-injected
        // validation bound, so this shader needs debug info; the other
        // backends answer the query natively either way.
        auto shader = device.compile(kernel, {.enable_debug_info = true});
        luisa::vector<uint> output_host(1u, 0u);
        (void)run_with_log_capture(device, [&](Stream &stream) {
            stream << shader(buffer, output).dispatch(1u)
                   << output.copy_to(luisa::span{output_host})
                   << synchronize();
        });
        expect(output_host[0] == 7u)
            << "buffer.size() returned the wrong element count";
    }

  // ------------------------------------------------------------------
  // 13. Zero-overhead sanity: a kernel without any checkable operation
  // is rebuilt statement-for-statement identical (the sink is a shared
  // write, which the generator does not bound-check).
  // ------------------------------------------------------------------
  {
      auto kernel = Kernel1D{[&]() noexcept {
          Shared<uint> shared_values{16};
          auto index = dispatch_id().x;
          auto a = index & 0xFu;
          auto b = a | 0x10u;
          shared_values.write(index & 15u, b);
      }};
        auto original_count = count_statements(kernel.function()->function().body());
        auto transformed = add_debug_checks(kernel);
        auto transformed_count = count_statements(
            transformed.function()->function().body());
        expect(original_count == transformed_count)
            << "the debug generator altered a kernel without checkable "
               "operations ({} -> {} statements)";
      // and the guards do appear where required
      auto checked_plain = Kernel1D{[&](BufferVar<float> buf) noexcept {
          auto x = buf->read(0u);
          auto y = x / 0.0f;
          static_cast<void>(y);
      }};
      auto checked = add_debug_checks(checked_plain);
        expect(count_statements(checked.function()->function().body()) >
               count_statements(checked_plain.function()->function().body()))
            << "the debug generator did not guard a division";

        // Phase 5 checklist: guards advertise themselves through
        // propagated_builtin_callables() (BUFFER_SIZE / BYTE_BUFFER_SIZE /
        // ACCEL_SIZE) and set _requires_printing so backends enable the
        // device printer.
        auto buffer_probe = add_debug_checks(Kernel1D{[&](BufferVar<float> buf) noexcept {
            static_cast<void>(buf->read(dispatch_id().x));
        }});
        expect(buffer_probe.function()->function().propagated_builtin_callables().test(CallOp::BUFFER_SIZE))
            << "guarded buffer read does not advertise CallOp::BUFFER_SIZE";
        expect(buffer_probe.function()->function().requires_printing())
            << "guarded kernel did not set requires_printing";
        auto byte_probe = add_debug_checks(Kernel1D{[&](ByteBufferVar buf) noexcept {
            static_cast<void>(buf.template read<float>(dispatch_id().x * 4u));
        }});
        expect(byte_probe.function()->function().propagated_builtin_callables().test(CallOp::BYTE_BUFFER_SIZE))
            << "guarded byte-buffer read does not advertise CallOp::BYTE_BUFFER_SIZE";
        auto accel_probe = add_debug_checks(Kernel1D{[&](AccelVar acc) noexcept {
            static_cast<void>(acc->instance_user_id(dispatch_id().x));
        }});
        expect(accel_probe.function()->function().propagated_builtin_callables().test(CallOp::ACCEL_SIZE))
            << "guarded accel access does not advertise CallOp::ACCEL_SIZE";
        expect(accel_probe.function()->function().requires_printing())
            << "accel-guarded kernel did not set requires_printing";
    }

    // ------------------------------------------------------------------
    // 14. Volatile buffer read out of range.
    // ------------------------------------------------------------------
    {
        auto buffer = device.create_buffer<float>(8u);
        auto output = device.create_buffer<uint>(2u);
        auto kernel = add_debug_checks(Kernel1D{[&](BufferVar<float> buf,
                                                     BufferVar<uint> out) noexcept {
            auto index = dispatch_id().x;
            auto value = buf->volatile_read(index * 100u);
            out->write(index, 1u);
            static_cast<void>(value);
        }});
        auto shader = device.compile(kernel, {.enable_debug_info = true});
        luisa::vector<uint> output_host(2u, 0u);
        luisa::vector<float> input_host(8u, 1.f);
        auto messages = run_with_log_capture(device, [&](Stream &stream) {
            stream << buffer.copy_from(luisa::span{input_host})
                   << shader(buffer, output).dispatch(2u)
                   << output.copy_to(luisa::span{output_host})
                   << synchronize();
        });
        expect(count_messages_with(messages, contains_buffer_index) >= 1u)
            << "volatile buffer-read guard did not report";
        expect(output_host[0] == 1u && output_host[1] == 0u);
    }

    // ------------------------------------------------------------------
    // 15-18. Acceleration structures (ray-tracing capable backends only).
    // ------------------------------------------------------------------
    if (backend_supports_rtx(device)) {

        // 15. accel.size() equals the instance count used at build.
        {
            Buffer<float3> vertices = device.create_buffer<float3>(3u);
            Buffer<Triangle> triangles = device.create_buffer<Triangle>(1u);
            std::array<float3, 3u> vertex_data{
                float3(-0.5f, -0.5f, 0.0f),
                float3(0.5f, -0.5f, 0.0f),
                float3(0.0f, 0.5f, 0.0f)};
            std::array<Triangle, 1u> triangle_data{Triangle{0u, 1u, 2u}};
            Accel accel = device.create_accel();
            Mesh mesh = device.create_mesh(vertices, triangles);
            accel.emplace_back(mesh, make_float4x4(1.0f));
            accel.emplace_back(mesh, translation(make_float3(1.f, 0.f, 0.f)));
            accel.emplace_back(mesh, translation(make_float3(0.f, 1.f, 0.f)));
            auto output = device.create_buffer<uint>(1u);
            auto kernel = Kernel1D{[&](AccelVar acc, BufferVar<uint> out) noexcept {
                out->write(0u, acc->size());
            }};
            auto shader = device.compile(kernel, {.enable_debug_info = true});
            luisa::vector<uint> output_host(1u, 0u);
            run_with_log_capture(device, [&](Stream &stream) {
                stream << vertices.copy_from(luisa::span{vertex_data})
                       << triangles.copy_from(luisa::span{triangle_data})
                       << mesh.build()
                       << accel.build()
                       << shader(accel, output).dispatch(1u)
                       << output.copy_to(luisa::span{output_host})
                       << synchronize();
            });
            expect(output_host[0] == 3u)
                << "accel.size() returned {} instead of the built instance count";
        }

        // 16. RAY_TRACING_INSTANCE_USER_ID out of range: log + early stop.
        {
            Buffer<float3> vertices = device.create_buffer<float3>(3u);
            Buffer<Triangle> triangles = device.create_buffer<Triangle>(1u);
            std::array<float3, 3u> vertex_data{
                float3(-0.5f, -0.5f, 0.0f),
                float3(0.5f, -0.5f, 0.0f),
                float3(0.0f, 0.5f, 0.0f)};
            std::array<Triangle, 1u> triangle_data{Triangle{0u, 1u, 2u}};
            Accel accel = device.create_accel();
            Mesh mesh = device.create_mesh(vertices, triangles);
            accel.emplace_back(mesh, make_float4x4(1.0f));
            auto output = device.create_buffer<uint>(4u);
            auto kernel = add_debug_checks(Kernel1D{[&](AccelVar acc,
                                                         BufferVar<uint> out) noexcept {
                auto index = dispatch_id().x;// 2 of the 4 threads are OOB
                auto user_id = acc->instance_user_id(index);
                out->write(index, 1u);
                static_cast<void>(user_id);
            }});
            auto shader = device.compile(kernel, {.enable_debug_info = true});
            luisa::vector<uint> output_host(4u, 0u);
            auto messages = run_with_log_capture(device, [&](Stream &stream) {
                stream << vertices.copy_from(luisa::span{vertex_data})
                       << triangles.copy_from(luisa::span{triangle_data})
                       << mesh.build()
                       << accel.build()
                       << shader(accel, output).dispatch(4u)
                       << output.copy_to(luisa::span{output_host})
                       << synchronize();
            });
            expect(count_messages_with(messages, contains_accel_index) >= 1u)
                << "accel instance-index guard did not report";
            expect(output_host[0] == 1u && output_host[1] == 0u &&
                   output_host[2] == 0u && output_host[3] == 0u)
                << "out-of-range instance reads must stop the thread";
        }

        // 17. RAY_TRACING_SET_INSTANCE_TRANSFORM out of range: log, and the
        //     valid instances are untouched (verified device-side, the host
        //     Accel has no instance-transform getter). Thread 0 writes a
        //     2-matrix into the valid instance 0, thread 1 tries to write a
        //     3-matrix into the out-of-range instance 2 (which would land on
        //     the instance-1 record without the guard).
        {
            Buffer<float3> vertices = device.create_buffer<float3>(3u);
            Buffer<Triangle> triangles = device.create_buffer<Triangle>(1u);
            std::array<float3, 3u> vertex_data{
                float3(-0.5f, -0.5f, 0.0f),
                float3(0.5f, -0.5f, 0.0f),
                float3(0.0f, 0.5f, 0.0f)};
            std::array<Triangle, 1u> triangle_data{Triangle{0u, 1u, 2u}};
            Accel accel = device.create_accel();
            Mesh mesh = device.create_mesh(vertices, triangles);
            accel.emplace_back(mesh, make_float4x4(1.0f));
            accel.emplace_back(mesh, make_float4x4(1.0f));
            auto kernel = add_debug_checks(Kernel1D{[&](AccelVar acc) noexcept {
                auto index = dispatch_id().x * 2u;// 0 (valid), 2 (out of range)
                auto matrix = ite(dispatch_id().x == 0u,
                                  make_float4x4(2.0f), make_float4x4(3.0f));
                acc->set_instance_transform(index, matrix);
            }});
            auto shader = device.compile(kernel, {.enable_debug_info = true});
            auto diag_kernel = Kernel1D{[&](AccelVar acc,
                                             BufferVar<float2> out) noexcept {
                auto m0 = acc->instance_transform(0u);
                auto m1 = acc->instance_transform(1u);
                out->write(0u, make_float2(m0[0].x, m1[0].x));
            }};
            auto diag_shader = device.compile(diag_kernel);
            auto diag_out = device.create_buffer<float2>(1u);
            luisa::vector<float2> diag_host(1u);
            auto messages = run_with_log_capture(device, [&](Stream &stream) {
                stream << vertices.copy_from(luisa::span{vertex_data})
                       << triangles.copy_from(luisa::span{triangle_data})
                       << mesh.build()
                       << accel.build()
                       << shader(accel).dispatch(2u)
                       << diag_shader(accel, diag_out).dispatch(1u)
                       << diag_out.copy_to(luisa::span{diag_host})
                       << synchronize();
            });
            expect(count_messages_with(messages, contains_accel_index) >= 1u)
                << "accel instance-index write guard did not report";
            expect(diag_host[0].x == 2.0f)
                << "the valid instance write was not applied";
            expect(diag_host[0].y == 1.0f)
                << "the out-of-range write touched a valid instance";
        }

        // 18. Zero-overhead sanity (accel): tracing does not pull in an
        //     instance-index guard. The trace result goes to a shared write,
        //     which the generator does not bound-check, so the statement
        //     count must be unchanged.
        {
            Buffer<float3> vertices = device.create_buffer<float3>(3u);
            Buffer<Triangle> triangles = device.create_buffer<Triangle>(1u);
            std::array<float3, 3u> vertex_data{
                float3(-0.5f, -0.5f, 0.0f),
                float3(0.5f, -0.5f, 0.0f),
                float3(0.0f, 0.5f, 0.0f)};
            std::array<Triangle, 1u> triangle_data{Triangle{0u, 1u, 2u}};
            Accel accel = device.create_accel();
            Mesh mesh = device.create_mesh(vertices, triangles);
            accel.emplace_back(mesh, make_float4x4(1.0f));
            auto kernel = Kernel1D{[&](AccelVar acc) noexcept {
                Shared<float> shared_result{1u};
                auto ray = make_ray(make_float3(0.f),
                                    make_float3(0.f, 0.f, -1.f));
                auto hit = acc->intersect(ray, {.visibility_mask = 0xffu});
                shared_result.write(0u, hit->distance());
            }};
            auto original_count = count_statements(
                kernel.function()->function().body());
            auto transformed = add_debug_checks(kernel);
            auto transformed_count = count_statements(
                transformed.function()->function().body());
            expect(original_count == transformed_count)
                << "the debug generator guarded a trace without an instance "
                   "access";
        }
    }
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    test_function_debugger(device);
}
