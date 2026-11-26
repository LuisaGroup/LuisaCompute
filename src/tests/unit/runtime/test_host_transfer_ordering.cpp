// Regression tests for host transfer ordering (Cases A and B):
//   - Case A (LuisaCompute#143): same-stream D2H into host memory followed by
//     H2D from overlapping host memory without intervening host sync, tested
//     both within a single CommandList and across separate submissions, for
//     same-buffer, cross-buffer, partial-overlap, same-list WAW, and texture
//     / cross-type transfers.
//   - Case B (HIP ed7ac101 / fence-only visibility): event-only buffer and
//     texture readbacks checked immediately after waiting on the Event (via
//     event.synchronize() and event.is_completed()) WITHOUT first calling
//     stream.synchronize(), plus a stream-sync control.
//   - Isolation checks: cross-stream GPU gate wait ensures fixes do not use a
//     global device sync, and blocked user callbacks on HIP/CUDA ensure transfer
//     completion does not wait on unrelated application callbacks.

#include "ut/ut.hpp"
#include "test_device.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string_view>
#include <thread>
#include <vector>

#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] bool approx_eq(float a, float b) noexcept {
    return std::abs(a - b) <= 1.0e-5f;
}

[[nodiscard]] bool approx_eq4(float4 a, float4 b) noexcept {
    return approx_eq(a.x, b.x) && approx_eq(a.y, b.y) &&
           approx_eq(a.z, b.z) && approx_eq(a.w, b.w);
}

void run_tests(Device &device, luisa::string_view backend_name, bool run_perf) noexcept {
    auto stream = device.create_stream();

    auto write_pattern_kernel = device.compile<1>([](BufferFloat buf, UInt round) noexcept {
        auto i = dispatch_id().x;
        buf.write(i, cast<float>(i) * 1001.0f + cast<float>(round) * 17.0f);
    });

    auto write_const_kernel = device.compile<1>([](BufferFloat buf, Float base) noexcept {
        auto i = dispatch_id().x;
        buf.write(i, base + cast<float>(i));
    });

    auto heavy_compute_kernel = device.compile<1>([](BufferFloat buf, UInt iter) noexcept {
        auto i = dispatch_id().x;
        auto x = cast<float>(i) + cast<float>(iter) * 3.0f;
        for (uint k = 0u; k < 32u; ++k) {
            x = x * 1.0001f + 0.5f;
            x = x - 0.5f;
        }
        buf.write(i, cast<float>(i) * 2.0f + cast<float>(iter) + x * 0.0f);
    });

    auto write_tex_kernel = device.compile<2>([](ImageFloat img, UInt round) noexcept {
        auto p = dispatch_id().xy();
        img.write(p, make_float4(
                         cast<float>(p.x) + cast<float>(round),
                         cast<float>(p.y) + cast<float>(round) * 2.0f,
                         cast<float>(p.x + p.y),
                         1.0f));
    });

    // Helper: check Case A (same CommandList + separate submissions) on a given stream.
    auto verify_case_a_core = [&](Stream &s, luisa::string_view tag) {
        constexpr uint32_t N = 4096u;
        auto z = device.create_buffer<float>(N);
        auto x = device.create_buffer<float>(N);
        luisa::vector<float> z_data(N);
        luisa::vector<float> x_data(N);

        // Case A1: Issue #143 exact pattern in a SINGLE CommandList
        for (uint32_t round = 0u; round < 8u; ++round) {
            std::fill(z_data.begin(), z_data.end(), 123.0f);
            std::fill(x_data.begin(), x_data.end(), -1.0f);
            CommandList list;
            list << z.copy_from(z_data.data())
                 << write_pattern_kernel(z, round).dispatch(N)
                 << z.copy_to(z_data.data())
                 << z.copy_from(z_data.data())
                 << x.copy_from(z)
                 << x.copy_to(x_data.data());
            s << list.commit() << synchronize();

            bool ok = true;
            for (uint32_t i = 0u; i < N; ++i) {
                float expected = static_cast<float>(i) * 1001.0f + static_cast<float>(round) * 17.0f;
                if (!approx_eq(z_data[i], expected) || !approx_eq(x_data[i], expected)) {
                    LUISA_WARNING("[{}] Case A1 mismatch round={} i={}: z={} x={} expected={}",
                                  tag, round, i, z_data[i], x_data[i], expected);
                    ok = false;
                    break;
                }
            }
            expect(ok) << tag << " Case A1 (same CommandList D2H->H2D) round " << round;
        }

        // Case A2: Issue #143 pattern across SEPARATE submissions without host sync
        for (uint32_t round = 0u; round < 8u; ++round) {
            std::fill(z_data.begin(), z_data.end(), 123.0f);
            std::fill(x_data.begin(), x_data.end(), -1.0f);
            s << z.copy_from(z_data.data())
              << write_pattern_kernel(z, round).dispatch(N)
              << z.copy_to(z_data.data())
              << commit();
            s << z.copy_from(z_data.data())
              << x.copy_from(z)
              << x.copy_to(x_data.data())
              << synchronize();

            bool ok = true;
            for (uint32_t i = 0u; i < N; ++i) {
                float expected = static_cast<float>(i) * 1001.0f + static_cast<float>(round) * 17.0f;
                if (!approx_eq(z_data[i], expected) || !approx_eq(x_data[i], expected)) {
                    LUISA_WARNING("[{}] Case A2 mismatch round={} i={}: z={} x={} expected={}",
                                  tag, round, i, z_data[i], x_data[i], expected);
                    ok = false;
                    break;
                }
            }
            expect(ok) << tag << " Case A2 (separate submissions D2H->H2D) round " << round;
        }
    };

    "case_a_issue143_same_and_separate"_test = [&] {
        verify_case_a_core(stream, backend_name);
    };

    "case_a3_cross_buffer_same_host_memory"_test = [&] {
        constexpr uint32_t N = 2048u;
        auto buf_a = device.create_buffer<float>(N);
        auto buf_b = device.create_buffer<float>(N);
        luisa::vector<float> shared_host(N, 77.0f);
        luisa::vector<float> out_host(N, -5.0f);

        // Same CommandList with DISTINCT device buffers sharing host memory
        for (uint32_t round = 0u; round < 4u; ++round) {
            std::fill(shared_host.begin(), shared_host.end(), 77.0f);
            std::fill(out_host.begin(), out_host.end(), -5.0f);
            CommandList list;
            list << write_pattern_kernel(buf_a, round).dispatch(N)
                 << buf_a.copy_to(shared_host.data())
                 << buf_b.copy_from(shared_host.data())
                 << buf_b.copy_to(out_host.data());
            stream << list.commit() << synchronize();

            bool ok = true;
            for (uint32_t i = 0u; i < N; ++i) {
                float expected = static_cast<float>(i) * 1001.0f + static_cast<float>(round) * 17.0f;
                if (!approx_eq(shared_host[i], expected) || !approx_eq(out_host[i], expected)) {
                    LUISA_WARNING("[{}] Case A3 same_list mismatch round={} i={}: shared={} out={} expected={}",
                                  backend_name, round, i, shared_host[i], out_host[i], expected);
                    ok = false;
                    break;
                }
            }
            expect(ok) << "Case A3 same CommandList cross-buffer";
        }

        // Separate submissions with DISTINCT device buffers sharing host memory
        for (uint32_t round = 0u; round < 4u; ++round) {
            std::fill(shared_host.begin(), shared_host.end(), 77.0f);
            std::fill(out_host.begin(), out_host.end(), -5.0f);
            stream << write_pattern_kernel(buf_a, round + 10u).dispatch(N)
                   << buf_a.copy_to(shared_host.data())
                   << commit();
            stream << buf_b.copy_from(shared_host.data())
                   << buf_b.copy_to(out_host.data())
                   << synchronize();

            bool ok = true;
            for (uint32_t i = 0u; i < N; ++i) {
                float expected = static_cast<float>(i) * 1001.0f + static_cast<float>(round + 10u) * 17.0f;
                if (!approx_eq(shared_host[i], expected) || !approx_eq(out_host[i], expected)) {
                    LUISA_WARNING("[{}] Case A3 separate mismatch round={} i={}: shared={} out={} expected={}",
                                  backend_name, round, i, shared_host[i], out_host[i], expected);
                    ok = false;
                    break;
                }
            }
            expect(ok) << "Case A3 separate submissions cross-buffer";
        }
    };

    "case_a4_partial_overlap_and_waw_downloads"_test = [&] {
        constexpr uint32_t N = 1024u;
        constexpr uint32_t Half = N / 2u;
        constexpr uint32_t Quarter = N / 4u;
        auto buf_a = device.create_buffer<float>(Half);
        auto buf_b = device.create_buffer<float>(Half);
        luisa::vector<float> host(N);
        luisa::vector<float> verify_b(Half);

        auto check_partial = [&](bool same_list) {
            std::fill(host.begin(), host.end(), 77.0f);
            std::fill(verify_b.begin(), verify_b.end(), -1.0f);
            if (same_list) {
                CommandList list;
                list << write_const_kernel(buf_a, 1000.0f).dispatch(Half)
                     << buf_a.copy_to(host.data())
                     << buf_b.copy_from(host.data() + Quarter)
                     << buf_b.copy_to(verify_b.data());
                stream << list.commit() << synchronize();
            } else {
                stream << write_const_kernel(buf_a, 1000.0f).dispatch(Half)
                       << buf_a.copy_to(host.data())
                       << commit();
                stream << buf_b.copy_from(host.data() + Quarter)
                       << buf_b.copy_to(verify_b.data())
                       << synchronize();
            }
            bool ok = true;
            for (uint32_t k = 0u; k < Half; ++k) {
                float expected = k < Quarter ? (1000.0f + static_cast<float>(Quarter + k)) : 77.0f;
                if (!approx_eq(verify_b[k], expected)) {
                    LUISA_WARNING("Case A4 partial same_list={} k={}: got={} expected={}",
                                  same_list, k, verify_b[k], expected);
                    ok = false;
                    break;
                }
            }
            expect(ok) << "Case A4 partial overlap same_list=" << same_list;
        };
        check_partial(true);
        check_partial(false);

        // Same-list WAW downloads to same host pointer across distinct device buffers
        auto waw_a = device.create_buffer<float>(Half);
        auto waw_b = device.create_buffer<float>(Half);
        luisa::vector<float> init_b(Half, 222.0f);
        luisa::vector<float> waw_host(Half, -3.0f);
        stream << waw_b.copy_from(init_b.data()) << synchronize();
        {
            CommandList list;
            list << write_const_kernel(waw_a, 111.0f).dispatch(Half)
                 << waw_a.copy_to(waw_host.data())
                 << waw_b.copy_to(waw_host.data());
            stream << list.commit() << synchronize();
        }
        bool waw_ok = true;
        for (uint32_t i = 0u; i < Half; ++i) {
            if (!approx_eq(waw_host[i], 222.0f)) {
                waw_ok = false;
                break;
            }
        }
        expect(waw_ok) << "Case A4 same-list WAW download ordering";
    };

    "case_a5_texture_and_cross_type_d2h_h2d"_test = [&] {
        constexpr uint2 tex_size = make_uint2(64u, 64u);
        constexpr uint32_t texels = tex_size.x * tex_size.y;
        auto tex_a = device.create_image<float>(PixelStorage::FLOAT4, tex_size);
        auto tex_b = device.create_image<float>(PixelStorage::FLOAT4, tex_size);
        auto buf_from_tex = device.create_buffer<float4>(texels);
        luisa::vector<float4> host_tex(texels);
        luisa::vector<float4> verify_tex(texels);
        luisa::vector<float4> verify_buf(texels);

        auto run_tex_round = [&](bool same_list, uint32_t round) {
            std::fill(host_tex.begin(), host_tex.end(), make_float4(-9.0f));
            std::fill(verify_tex.begin(), verify_tex.end(), make_float4(-8.0f));
            std::fill(verify_buf.begin(), verify_buf.end(), make_float4(-7.0f));
            if (same_list) {
                CommandList list;
                list << write_tex_kernel(tex_a, round).dispatch(tex_size)
                     << tex_a.copy_to(host_tex.data())
                     << tex_b.copy_from(host_tex.data())
                     << buf_from_tex.copy_from(host_tex.data())
                     << tex_b.copy_to(verify_tex.data())
                     << buf_from_tex.copy_to(verify_buf.data());
                stream << list.commit() << synchronize();
            } else {
                stream << write_tex_kernel(tex_a, round).dispatch(tex_size)
                       << tex_a.copy_to(host_tex.data())
                       << commit();
                stream << tex_b.copy_from(host_tex.data())
                       << buf_from_tex.copy_from(host_tex.data())
                       << tex_b.copy_to(verify_tex.data())
                       << buf_from_tex.copy_to(verify_buf.data())
                       << synchronize();
            }
            bool ok = true;
            for (uint32_t y = 0u; y < tex_size.y && ok; ++y) {
                for (uint32_t x = 0u; x < tex_size.x; ++x) {
                    auto idx = y * tex_size.x + x;
                    auto expected = make_float4(
                        static_cast<float>(x) + static_cast<float>(round),
                        static_cast<float>(y) + static_cast<float>(round) * 2.0f,
                        static_cast<float>(x + y),
                        1.0f);
                    if (!approx_eq4(host_tex[idx], expected) ||
                        !approx_eq4(verify_tex[idx], expected) ||
                        !approx_eq4(verify_buf[idx], expected)) {
                        LUISA_WARNING("[{}] Case A5 tex mismatch same_list={} ({},{}) host=({},{},{},{}) tex=({},{},{},{}) buf=({},{},{},{})",
                                      backend_name, same_list, x, y,
                                      host_tex[idx].x, host_tex[idx].y, host_tex[idx].z, host_tex[idx].w,
                                      verify_tex[idx].x, verify_tex[idx].y, verify_tex[idx].z, verify_tex[idx].w,
                                      verify_buf[idx].x, verify_buf[idx].y, verify_buf[idx].z, verify_buf[idx].w);
                        ok = false;
                        break;
                    }
                }
            }
            expect(ok) << "Case A5 texture/cross-type same_list=" << same_list;
        };
        run_tex_round(true, 1u);
        run_tex_round(false, 2u);
    };

    // Helper: Case B (event-only buffer & texture readback without stream.synchronize()) + stream-sync control
    auto verify_case_b_core = [&](Stream &s, luisa::string_view tag) {
        auto ev = device.create_timeline_event();
        uint64_t fence = 1u;

        // Buffer readback: large buffer + 40-download batch (> HIP max_deferred_download_batch=32)
        constexpr uint32_t large_n = 262144u;
        constexpr uint32_t small_n = 64u;
        constexpr uint32_t batch_count = 40u;
        auto large_buf = device.create_buffer<float>(large_n);
        auto small_buf = device.create_buffer<float>(small_n);
        luisa::vector<float> large_host(large_n, -999.0f);
        luisa::vector<luisa::vector<float>> small_hosts(batch_count, luisa::vector<float>(small_n, -999.0f));

        for (uint32_t iter = 0u; iter < 16u; ++iter) {
            std::fill(large_host.begin(), large_host.end(), -999.0f);
            for (auto &vec : small_hosts) {
                std::fill(vec.begin(), vec.end(), -999.0f);
            }
            CommandList list;
            list << heavy_compute_kernel(large_buf, iter).dispatch(large_n)
                 << heavy_compute_kernel(small_buf, iter).dispatch(small_n)
                 << large_buf.copy_to(large_host.data());
            for (uint32_t b = 0u; b < batch_count; ++b) {
                list << small_buf.copy_to(small_hosts[b].data());
            }
            s << list.commit() << ev.signal(fence);

            if ((iter & 1u) == 0u) {
                ev.synchronize(fence);
            } else {
                while (!ev.is_completed(fence)) {
                    std::this_thread::yield();
                }
            }
            // IMMEDIATELY check BEFORE any stream.synchronize()!
            bool large_ok = true;
            for (uint32_t i = 0u; i < large_n; ++i) {
                float expected = static_cast<float>(i) * 2.0f + static_cast<float>(iter);
                if (!approx_eq(large_host[i], expected)) {
                    LUISA_WARNING("[{}] Case B buffer iter={} i={}: got={} expected={}",
                                  tag, iter, i, large_host[i], expected);
                    large_ok = false;
                    break;
                }
            }
            bool small_ok = true;
            for (uint32_t b = 0u; b < batch_count && small_ok; ++b) {
                for (uint32_t i = 0u; i < small_n; ++i) {
                    float expected = static_cast<float>(i) * 2.0f + static_cast<float>(iter);
                    if (!approx_eq(small_hosts[b][i], expected)) {
                        LUISA_WARNING("[{}] Case B small batch={} iter={} i={}: got={} expected={}",
                                      tag, b, iter, i, small_hosts[b][i], expected);
                        small_ok = false;
                        break;
                    }
                }
            }
            expect(large_ok) << tag << " Case B1 event-only large buffer iter=" << iter;
            expect(small_ok) << tag << " Case B1 event-only batch buffer iter=" << iter;
            ++fence;
        }

        // Event-only texture readback (NO stream.synchronize() before verify!)
        constexpr uint2 img_size = make_uint2(128u, 128u);
        constexpr uint32_t img_texels = img_size.x * img_size.y;
        auto tex = device.create_image<float>(PixelStorage::FLOAT4, img_size);
        luisa::vector<float4> host_img(img_texels, make_float4(-999.0f));
        for (uint32_t iter = 0u; iter < 16u; ++iter) {
            std::fill(host_img.begin(), host_img.end(), make_float4(-999.0f));
            s << write_tex_kernel(tex, iter).dispatch(img_size)
              << tex.copy_to(host_img.data())
              << ev.signal(fence);
            if ((iter & 1u) == 0u) {
                ev.synchronize(fence);
            } else {
                while (!ev.is_completed(fence)) {
                    std::this_thread::yield();
                }
            }
            bool tex_ok = true;
            for (uint32_t y = 0u; y < img_size.y && tex_ok; ++y) {
                for (uint32_t x = 0u; x < img_size.x; ++x) {
                    auto idx = y * img_size.x + x;
                    auto expected = make_float4(
                        static_cast<float>(x) + static_cast<float>(iter),
                        static_cast<float>(y) + static_cast<float>(iter) * 2.0f,
                        static_cast<float>(x + y),
                        1.0f);
                    if (!approx_eq4(host_img[idx], expected)) {
                        LUISA_WARNING("[{}] Case B texture iter={} ({},{}): got=({},{},{},{}) expected=({},{},{},{})",
                                      tag, iter, x, y,
                                      host_img[idx].x, host_img[idx].y, host_img[idx].z, host_img[idx].w,
                                      expected.x, expected.y, expected.z, expected.w);
                        tex_ok = false;
                        break;
                    }
                }
            }
            expect(tex_ok) << tag << " Case B2 event-only texture iter=" << iter;
            ++fence;
        }

        // Also check auto-counting Event::signal()/synchronize()/is_completed() without stream.synchronize().
        {
            auto counting_ev = device.create_event();
            std::fill(large_host.begin(), large_host.end(), -999.0f);
            std::fill(host_img.begin(), host_img.end(), make_float4(-999.0f));
            s << heavy_compute_kernel(large_buf, 77u).dispatch(large_n)
              << large_buf.copy_to(large_host.data())
              << write_tex_kernel(tex, 77u).dispatch(img_size)
              << tex.copy_to(host_img.data())
              << counting_ev.signal();
            counting_ev.synchronize();
            bool ev_ok = approx_eq(large_host[0], 77.0f) &&
                         approx_eq(large_host[large_n - 1u], static_cast<float>(large_n - 1u) * 2.0f + 77.0f) &&
                         approx_eq4(host_img[0], make_float4(77.0f, 154.0f, 0.0f, 1.0f));
            expect(ev_ok) << tag << " Case B Event auto-fence readback";
        }

        // Control: stream.synchronize()
        std::fill(large_host.begin(), large_host.end(), -999.0f);
        std::fill(host_img.begin(), host_img.end(), make_float4(-999.0f));
        s << heavy_compute_kernel(large_buf, 99u).dispatch(large_n)
          << large_buf.copy_to(large_host.data())
          << write_tex_kernel(tex, 99u).dispatch(img_size)
          << tex.copy_to(host_img.data())
          << synchronize();
        bool ctrl_ok = true;
        for (uint32_t i = 0u; i < large_n; ++i) {
            if (!approx_eq(large_host[i], static_cast<float>(i) * 2.0f + 99.0f)) {
                ctrl_ok = false;
                break;
            }
        }
        for (uint32_t i = 0u; i < img_texels && ctrl_ok; ++i) {
            uint32_t x = i % img_size.x;
            uint32_t y = i / img_size.x;
            auto expected = make_float4(static_cast<float>(x) + 99.0f,
                                        static_cast<float>(y) + 198.0f,
                                        static_cast<float>(x + y), 1.0f);
            if (!approx_eq4(host_img[i], expected)) {
                ctrl_ok = false;
                break;
            }
        }
        expect(ctrl_ok) << tag << " Case B3 stream-sync control";
        s << synchronize();
    };

    "case_b_event_only_readback_and_control"_test = [&] {
        verify_case_b_core(stream, backend_name);
    };

    // Isolation check 1: another stream busy/blocked -> verify no global device sync.
    if (backend_name != "simd") {
        "isolation_no_global_device_sync"_test = [&] {
            auto stream_other = device.create_stream();
            if (backend_name == "hip" || backend_name == "cuda") {
                auto other_buf = device.create_buffer<float>(1u);
                auto other_ev = device.create_timeline_event();
                auto busy_shader = device.compile<1>([](BufferFloat out, UInt iters) noexcept {
                    Float acc = 1.0f;
                    for (auto k : dynamic_range(iters)) {
                        static_cast<void>(k);
                        acc = acc * 1.0000001f + 0.0000001f;
                    }
                    out.write(0u, acc);
                });
                uint32_t busy_iters = 2'000'000u;
                for (;;) {
                    auto t0 = std::chrono::steady_clock::now();
                    stream_other << busy_shader(other_buf, busy_iters).dispatch(1u)
                                 << synchronize();
                    auto ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t0)
                                  .count();
                    if (ms >= 200.0 || busy_iters >= 128'000'000u) {
                        break;
                    }
                    busy_iters *= 2u;
                }

                constexpr uint32_t iso_n = 2048u;
                auto z = device.create_buffer<float>(iso_n);
                auto x = device.create_buffer<float>(iso_n);
                auto b_buf = device.create_buffer<float>(iso_n);
                luisa::vector<float> z_data(iso_n, 123.0f);
                luisa::vector<float> x_data(iso_n, -1.0f);
                luisa::vector<float> b_host(iso_n, -999.0f);
                auto ev = device.create_timeline_event();

                stream_other << busy_shader(other_buf, busy_iters).dispatch(1u)
                             << other_ev.signal(1u);

                {
                    CommandList list;
                    list << z.copy_from(z_data.data())
                         << write_pattern_kernel(z, 9u).dispatch(iso_n)
                         << z.copy_to(z_data.data())
                         << z.copy_from(z_data.data())
                         << x.copy_from(z)
                         << x.copy_to(x_data.data());
                    stream << list.commit() << synchronize();
                }
                {
                    CommandList list;
                    list << heavy_compute_kernel(b_buf, 7u).dispatch(iso_n)
                         << b_buf.copy_to(b_host.data());
                    stream << list.commit() << ev.signal(1u);
                    ev.synchronize(1u);
                }
                const bool other_still_running = !other_ev.is_completed(1u);
                bool fast_a_ok = true;
                bool fast_b_ok = true;
                for (uint32_t i = 0u; i < iso_n; ++i) {
                    float exp_a = static_cast<float>(i) * 1001.0f + 9.0f * 17.0f;
                    if (!approx_eq(z_data[i], exp_a) || !approx_eq(x_data[i], exp_a)) {
                        fast_a_ok = false;
                        break;
                    }
                    float exp_b = static_cast<float>(i) * 2.0f + 7.0f;
                    if (!approx_eq(b_host[i], exp_b)) {
                        fast_b_ok = false;
                        break;
                    }
                }
                expect(fast_a_ok) << "no_global_sync concurrent Case A1";
                expect(fast_b_ok) << "no_global_sync concurrent Case B1";
                expect(other_still_running)
                    << "stream transfer sync must not perform a global device sync";

                verify_case_a_core(stream, "no_global_sync");
                verify_case_b_core(stream, "no_global_sync");

                stream_other << synchronize();
                stream << synchronize();
            } else {
                auto gate_event = device.create_timeline_event();
                stream_other << gate_event.wait(1u);

                verify_case_a_core(stream, "no_global_sync");
                verify_case_b_core(stream, "no_global_sync");

                stream << gate_event.signal(1u);
                stream_other << synchronize();
                stream << synchronize();
            }
        };
    }

    // Isolation check 2a: blocked user callback on ANOTHER stream -> neither Case A nor Case B
    // (nor stream.synchronize()) on `stream` may wait on the other stream's callback.
    if (backend_name != "simd") {
        "isolation_no_wait_on_other_stream_callback"_test = [&] {
            auto stream_other = device.create_stream();
            std::atomic_bool cb_started{false};
            std::atomic_bool cb_release{false};
            CommandList blocker;
            blocker.add_callback([&]() noexcept {
                cb_started.store(true, std::memory_order_release);
                while (!cb_release.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }
            });
            stream_other << blocker.commit();
            while (!cb_started.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            verify_case_a_core(stream, "other_cb");
            verify_case_b_core(stream, "other_cb");

            cb_release.store(true, std::memory_order_release);
            stream_other << synchronize();
            stream << synchronize();
        };
    }

    // Isolation check 2b: blocked user callback on the SAME stream (HIP/CUDA) -> verify Case A
    // transfer ordering and Case B event-only readbacks complete via Event without waiting on
    // unrelated application callbacks on `_callback_thread` (never calling stream.synchronize()).
    if (backend_name == "hip" || backend_name == "cuda") {
        "isolation_no_wait_on_same_stream_user_callback"_test = [&] {
            constexpr uint32_t N = 2048u;
            constexpr uint32_t small_n = 64u;
            constexpr uint32_t batch_count = 40u;
            constexpr uint2 img_size = make_uint2(64u, 64u);
            constexpr uint32_t img_texels = img_size.x * img_size.y;

            auto z = device.create_buffer<float>(N);
            auto x = device.create_buffer<float>(N);
            auto large_buf = device.create_buffer<float>(65536u);
            auto small_buf = device.create_buffer<float>(small_n);
            auto tex = device.create_image<float>(PixelStorage::FLOAT4, img_size);
            auto ev = device.create_timeline_event();
            uint64_t fence = 1u;

            luisa::vector<float> z_data(N);
            luisa::vector<float> x_data(N);
            luisa::vector<float> large_host(65536u);
            luisa::vector<luisa::vector<float>> small_hosts(batch_count, luisa::vector<float>(small_n));
            luisa::vector<float4> host_img(img_texels);

            std::atomic_bool cb_started{false};
            std::atomic_bool cb_release{false};
            CommandList blocker;
            blocker.add_callback([&]() noexcept {
                cb_started.store(true, std::memory_order_release);
                while (!cb_release.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }
            });
            stream << blocker.commit();
            while (!cb_started.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            // Case A1 (same CommandList) waited via Event only
            std::fill(z_data.begin(), z_data.end(), 123.0f);
            std::fill(x_data.begin(), x_data.end(), -1.0f);
            {
                CommandList list;
                list << z.copy_from(z_data.data())
                     << write_pattern_kernel(z, 3u).dispatch(N)
                     << z.copy_to(z_data.data())
                     << z.copy_from(z_data.data())
                     << x.copy_from(z)
                     << x.copy_to(x_data.data());
                stream << list.commit() << ev.signal(fence);
            }
            ev.synchronize(fence++);
            bool a1_ok = true;
            for (uint32_t i = 0u; i < N; ++i) {
                float expected = static_cast<float>(i) * 1001.0f + 51.0f;
                if (!approx_eq(z_data[i], expected) || !approx_eq(x_data[i], expected)) {
                    LUISA_WARNING("[same_cb] Case A1 mismatch i={}: z={} x={} expected={}",
                                  i, z_data[i], x_data[i], expected);
                    a1_ok = false;
                    break;
                }
            }
            expect(a1_ok) << "cb_isolated Case A1 same CommandList";

            // Case A2 (separate submissions) waited via Event only
            std::fill(z_data.begin(), z_data.end(), 123.0f);
            std::fill(x_data.begin(), x_data.end(), -1.0f);
            stream << z.copy_from(z_data.data())
                   << write_pattern_kernel(z, 5u).dispatch(N)
                   << z.copy_to(z_data.data())
                   << commit();
            stream << z.copy_from(z_data.data())
                   << x.copy_from(z)
                   << x.copy_to(x_data.data())
                   << ev.signal(fence);
            ev.synchronize(fence++);
            bool a2_ok = true;
            for (uint32_t i = 0u; i < N; ++i) {
                float expected = static_cast<float>(i) * 1001.0f + 85.0f;
                if (!approx_eq(z_data[i], expected) || !approx_eq(x_data[i], expected)) {
                    LUISA_WARNING("[same_cb] Case A2 mismatch i={}: z={} x={} expected={}",
                                  i, z_data[i], x_data[i], expected);
                    a2_ok = false;
                    break;
                }
            }
            expect(a2_ok) << "cb_isolated Case A2 separate submissions";

            // Case B (event-only buffer + batch + texture readback) waited via Event only
            std::fill(large_host.begin(), large_host.end(), -999.0f);
            for (auto &vec : small_hosts) { std::fill(vec.begin(), vec.end(), -999.0f); }
            std::fill(host_img.begin(), host_img.end(), make_float4(-999.0f));
            {
                CommandList list;
                list << heavy_compute_kernel(large_buf, 11u).dispatch(65536u)
                     << heavy_compute_kernel(small_buf, 11u).dispatch(small_n)
                     << large_buf.copy_to(large_host.data());
                for (uint32_t b = 0u; b < batch_count; ++b) {
                    list << small_buf.copy_to(small_hosts[b].data());
                }
                list << write_tex_kernel(tex, 11u).dispatch(img_size)
                     << tex.copy_to(host_img.data());
                stream << list.commit() << ev.signal(fence);
            }
            while (!ev.is_completed(fence)) {
                std::this_thread::yield();
            }
            ++fence;

            bool b_buf_ok = true;
            for (uint32_t i = 0u; i < 65536u; ++i) {
                float expected = static_cast<float>(i) * 2.0f + 11.0f;
                if (!approx_eq(large_host[i], expected)) {
                    LUISA_WARNING("[same_cb] Case B buffer mismatch i={}: got={} expected={}",
                                  i, large_host[i], expected);
                    b_buf_ok = false;
                    break;
                }
            }
            for (uint32_t b = 0u; b < batch_count && b_buf_ok; ++b) {
                for (uint32_t i = 0u; i < small_n; ++i) {
                    float expected = static_cast<float>(i) * 2.0f + 11.0f;
                    if (!approx_eq(small_hosts[b][i], expected)) {
                        LUISA_WARNING("[same_cb] Case B batch={} mismatch i={}: got={} expected={}",
                                      b, i, small_hosts[b][i], expected);
                        b_buf_ok = false;
                        break;
                    }
                }
            }
            bool b_tex_ok = approx_eq4(host_img[0], make_float4(11.0f, 22.0f, 0.0f, 1.0f)) &&
                            approx_eq4(host_img[img_texels - 1u],
                                       make_float4(63.0f + 11.0f, 63.0f + 22.0f, 126.0f, 1.0f));
            if (!b_tex_ok) {
                LUISA_WARNING("[same_cb] Case B tex mismatch got[0]=({},{},{},{})",
                              host_img[0].x, host_img[0].y, host_img[0].z, host_img[0].w);
            }
            expect(b_buf_ok) << "cb_isolated Case B buffer/batch event-only readback";
            expect(b_tex_ok) << "cb_isolated Case B texture event-only readback";

            cb_release.store(true, std::memory_order_release);
            stream << synchronize();
        };
    }

    if (run_perf) {
        constexpr uint32_t elem_count = 1024u; // 4KB per download
        constexpr uint32_t iters = 200u;
        auto buf = device.create_buffer<float>(elem_count);
        auto buf2 = device.create_buffer<float>(elem_count);
        luisa::vector<float> host_a(elem_count, 1.0f);
        luisa::vector<float> host_b(elem_count, 2.0f);
        auto ev = device.create_timeline_event();

        // Warmup
        for (uint32_t i = 0u; i < 10u; ++i) {
            stream << write_const_kernel(buf, static_cast<float>(i)).dispatch(elem_count)
                   << buf.copy_to(host_a.data())
                   << synchronize();
        }

        // (a) Stream-sync fast path: small compute + 4x 4KB D2H + stream.synchronize()
        auto t0 = std::chrono::steady_clock::now();
        for (uint32_t i = 0u; i < iters; ++i) {
            CommandList list;
            list << write_const_kernel(buf, static_cast<float>(i)).dispatch(elem_count)
                 << buf.copy_to(host_a.data())
                 << buf.copy_to(host_a.data())
                 << buf.copy_to(host_a.data())
                 << buf.copy_to(host_a.data());
            stream << list.commit() << synchronize();
        }
        auto t1 = std::chrono::steady_clock::now();

        // (b) Event-only download path: small compute + 4x 4KB D2H + ev.signal + ev.synchronize
        uint64_t fence = 1u;
        for (uint32_t i = 0u; i < iters; ++i) {
            CommandList list;
            list << write_const_kernel(buf, static_cast<float>(i)).dispatch(elem_count)
                 << buf.copy_to(host_a.data())
                 << buf.copy_to(host_a.data())
                 << buf.copy_to(host_a.data())
                 << buf.copy_to(host_a.data());
            stream << list.commit() << ev.signal(fence);
            ev.synchronize(fence++);
        }
        auto t2 = std::chrono::steady_clock::now();

        // (c) Disjoint pipelined D2H (host_a) + H2D (host_b) without per-iter sync
        for (uint32_t i = 0u; i < iters; ++i) {
            CommandList list;
            list << buf.copy_to(host_a.data())
                 << buf.copy_to(host_a.data())
                 << buf.copy_to(host_a.data())
                 << buf.copy_to(host_a.data())
                 << buf2.copy_from(host_b.data())
                 << buf2.copy_from(host_b.data())
                 << buf2.copy_from(host_b.data())
                 << buf2.copy_from(host_b.data());
            stream << list.commit();
        }
        stream << synchronize();
        auto t3 = std::chrono::steady_clock::now();

        auto stream_sync_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        auto event_only_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        auto disjoint_pipeline_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        std::printf("PERF_SANITY backend=%.*s iters=%u stream_sync_ms=%.3f event_only_ms=%.3f disjoint_pipeline_ms=%.3f\n",
                    static_cast<int>(backend_name.size()), backend_name.data(), iters,
                    stream_sync_ms, event_only_ms, disjoint_pipeline_ms);
        std::fflush(stdout);
    }
}

}// namespace

int main(int argc, char *argv[]) {
    bool enable_validation = false;
    bool run_perf = false;
    std::vector<char *> filtered;
    filtered.reserve(static_cast<size_t>(argc));
    for (int i = 0; i < argc; ++i) {
        std::string_view arg{argv[i] ? argv[i] : ""};
        if (arg == "--validation") {
            enable_validation = true;
        } else if (arg == "--perf") {
            run_perf = true;
        } else {
            filtered.push_back(argv[i]);
        }
    }
    int filtered_argc = static_cast<int>(filtered.size());
    char **filtered_argv = filtered.data();

    auto dc = luisa::test::create_device_from_ut(
        filtered_argc, filtered_argv, nullptr, enable_validation);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(filtered_argc, const_cast<const char **>(filtered_argv));

    auto requested_backend = luisa::string_view{filtered_argv[1]};
    run_tests(dc->device, requested_backend, run_perf);
}
