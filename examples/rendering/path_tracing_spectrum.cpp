#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string_view>
#include <luisa/backends/ext/dx_hdr_ext.hpp>
#include <stb/stb_image_write.h>

#include "common/reference_compare.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

#include "cornell_box.h"

#include "spectrum_data.h"

#include "srgb_to_fourier_embedded.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

using namespace luisa;
using namespace luisa::compute;

struct Onb {
    float3 tangent;
    float3 binormal;
    float3 normal;
};

LUISA_STRUCT(Onb, tangent, binormal, normal) {
    [[nodiscard]] Float3 to_world(Var<float3> v) const noexcept {
        return v.x * tangent + v.y * binormal + v.z * normal;
    }
};

struct complex {
    float real, imag;
};

using Complex = Var<complex>;

LUISA_STRUCT(complex, real, imag) {
    [[nodiscard]] inline Complex conjugate() const noexcept {
        return def<complex>(real, -imag);
    }
    [[nodiscard]] inline Float abs_square() const noexcept {
        return real * real + imag * imag;
    }
};

[[nodiscard]] inline Complex operator-(Complex z) {
    return def<complex>(-z.real, -z.imag);
}
/*!	This operator implements complex addition.*/
[[nodiscard]] inline Complex operator+(Complex lhs, Complex rhs) {
    return def<complex>(lhs.real + rhs.real, lhs.imag + rhs.imag);
}
/*!	This operator implements complex subtraction.*/
[[nodiscard]] inline Complex operator-(Complex lhs, Complex rhs) {
    return def<complex>(lhs.real - rhs.real, lhs.imag - rhs.imag);
}
/*!	This operator implements complex multiplication.*/
[[nodiscard]] inline Complex operator*(Complex lhs, Complex rhs) {
    return def<complex>(lhs.real * rhs.real - lhs.imag * rhs.imag, lhs.real * rhs.imag + lhs.imag * rhs.real);
}
/*!	This operator implements mixed real complex addition.*/
[[nodiscard]] inline Complex operator+(Float lhs, Complex rhs) {
    return def<complex>(lhs + rhs.real, rhs.imag);
}
/*!	This operator implements  mixed real complex multiplication.*/
[[nodiscard]] inline Complex operator*(Float lhs, Complex rhs) {
    return def<complex>(lhs * rhs.real, lhs * rhs.imag);
}
/*!	This operator implements mixed real complex addition.*/
[[nodiscard]] inline Complex operator+(float lhs, Complex rhs) {
    return def<complex>(lhs + rhs.real, rhs.imag);
}
/*!	This operator implements  mixed real complex multiplication.*/
[[nodiscard]] inline Complex operator*(float lhs, Complex rhs) {
    return def<complex>(lhs * rhs.real, lhs * rhs.imag);
}
[[nodiscard]] inline Complex operator/(Complex numerator, Complex denominator) {
    auto factor = 1.0f / denominator->abs_square();
    return def<complex>((numerator.real * denominator.real + numerator.imag * denominator.imag) * factor, (-numerator.real * denominator.imag + numerator.imag * denominator.real) * factor);
}

void trigonometricToExponentialMomentsReal3(std::array<Complex, 3> &pOutExponentialMoment, Float3 pTrigonometricMoment) {
    auto zerothMomentPhase = pi * pTrigonometricMoment.x - pi_over_two;
    pOutExponentialMoment[0] = def<complex>(cos(zerothMomentPhase), sin(zerothMomentPhase));
    pOutExponentialMoment[0] = (1.0f / (pi * 4.0f)) * pOutExponentialMoment[0];
    pOutExponentialMoment[1] = pTrigonometricMoment.y * def<complex>(0.0f, 2 * pi) * pOutExponentialMoment[0];
    pOutExponentialMoment[2] = pTrigonometricMoment.z * def<complex>(0.0f, 2 * pi) * pOutExponentialMoment[0] + pTrigonometricMoment.y * def<complex>(0.0f, pi) * pOutExponentialMoment[1];
    pOutExponentialMoment[0] = 2.0f * pOutExponentialMoment[0];
}

void levinsonsAlgorithm3(std::array<Complex, 3> &pOutSolution, std::array<Complex, 3> &pFirstColumn) {
    pOutSolution[0] = def<complex>(1.0f / (pFirstColumn[0].real), 0.0f);
    auto dotProduct = pOutSolution[0].real * pFirstColumn[1];
    auto factor = 1.0f / (1.0f - dotProduct->abs_square());
    auto flippedSolution1 = def<complex>(pOutSolution[0].real, 0.0f);
    pOutSolution[0] = def<complex>(factor * pOutSolution[0].real, 0.0f);
    pOutSolution[1] = factor * (-flippedSolution1.real * dotProduct);
    dotProduct = pOutSolution[0].real * pFirstColumn[2] + pOutSolution[1] * pFirstColumn[1];
    factor = 1.0f / (1.0f - dotProduct->abs_square());
    flippedSolution1 = pOutSolution[1]->conjugate();
    auto flippedSolution2 = def<complex>(pOutSolution[0].real, 0.0f);
    pOutSolution[0] = def<complex>(factor * pOutSolution[0].real, 0.0f);
    pOutSolution[1] = factor * (pOutSolution[1] - flippedSolution1 * dotProduct);
    pOutSolution[2] = factor * (-flippedSolution2.real * dotProduct);
}

void computeAutocorrelation3(std::array<Complex, 3> &pOutAutocorrelation, std::array<Complex, 3> &pSignal) {
    pOutAutocorrelation[0] = pSignal[0] * pSignal[0]->conjugate() + pSignal[1] * pSignal[1]->conjugate() + pSignal[2] * pSignal[2]->conjugate();
    pOutAutocorrelation[1] = pSignal[0] * pSignal[1]->conjugate() + pSignal[1] * pSignal[2]->conjugate();
    pOutAutocorrelation[2] = pSignal[0] * pSignal[2]->conjugate();
}

Float3 computeImaginaryCorrelation3(std::array<Complex, 3> &pLHS, std::array<Complex, 3> &pRHS) {
    Float3 pOutCorrelation;
    pOutCorrelation.x = pLHS[0].real * pRHS[0].imag + pLHS[0].imag * pRHS[0].real + pLHS[1].real * pRHS[1].imag + pLHS[1].imag * pRHS[1].real + pLHS[2].real * pRHS[2].imag + pLHS[2].imag * pRHS[2].real;
    pOutCorrelation.y = pLHS[1].real * pRHS[0].imag + pLHS[1].imag * pRHS[0].real + pLHS[2].real * pRHS[1].imag + pLHS[2].imag * pRHS[1].real;
    pOutCorrelation.z = pLHS[2].real * pRHS[0].imag + pLHS[2].imag * pRHS[0].real;
    return pOutCorrelation;
}

Float3 prepareReflectanceSpectrumLagrangeReal3(Float3 pTrigonometricMoment) {
    std::array<Complex, 3> pExponentialMoment;
    trigonometricToExponentialMomentsReal3(pExponentialMoment, pTrigonometricMoment);
    std::array<Complex, 3> pEvaluationPolynomial;
    levinsonsAlgorithm3(pEvaluationPolynomial, pExponentialMoment);
    pEvaluationPolynomial[0] = (2 * pi) * pEvaluationPolynomial[0];
    pEvaluationPolynomial[1] = (2 * pi) * pEvaluationPolynomial[1];
    pEvaluationPolynomial[2] = (2 * pi) * pEvaluationPolynomial[2];
    std::array<Complex, 3> pAutocorrelation;
    computeAutocorrelation3(pAutocorrelation, pEvaluationPolynomial);
    pExponentialMoment[0] = 0.5f * pExponentialMoment[0];
    auto pOutLagrangeMultiplier = computeImaginaryCorrelation3(pAutocorrelation, pExponentialMoment);
    auto normalizationFactor = 1.0f / (pi * pEvaluationPolynomial[0].real);
    pOutLagrangeMultiplier *= normalizationFactor;
    return pOutLagrangeMultiplier;
}

Float evaluateReflectanceSpectrumLagrangeReal3(Float phase, Float3 pLagrangeMultiplier) {
    auto conjCirclePoint = def<complex>(cos(-phase), sin(-phase));
    auto conjCirclePoint2 = conjCirclePoint * conjCirclePoint;
    auto result = pLagrangeMultiplier.y * conjCirclePoint.real + pLagrangeMultiplier.z * conjCirclePoint2.real;
    return atan(2.0f * result + pLagrangeMultiplier.x) * inv_pi + 0.5f;
}

/**
 * Compute inverse CDF (quantiles) from unnormalized PDF.
 *
 * @param x_span Sorted span of x values (size m).
 * @param y_span PDF values corresponding to x (size m), may include zeros at ends.
 * @param ps probabilities in (0,1) for which to compute quantiles.
 * @param iterations Maximum Newton-Raphson iterations per quantile.
 * @param tol Convergence tolerance for Newton-Raphson.
 * @return Vector of quantiles corresponding to probabilities ps.
 */
template<class T, class G>
luisa::vector<T> generate_quantiles(
    luisa::span<T const> xs,
    luisa::span<T const> ys,
    G &&ps,
    int iterations,
    T tol) {
    size_t m = xs.size();

    // Compute integration step sizes (left differences)
    luisa::vector<T> dx(m);
    dx[0] = 0.0;
    for (size_t i = 1; i < m; ++i) {
        dx[i] = xs[i] - xs[i - 1];
    }

    // Compute unnormalized CDF
    luisa::vector<T> cdf(m);
    cdf[0] = ys[0] * dx[0];
    for (size_t i = 1; i < m; ++i) {
        cdf[i] = cdf[i - 1] + ys[i] * dx[i];
    }
    auto &total = cdf.back();
    for (auto &v : cdf) v /= total;// Normalize to [0,1]

    // Determine valid support where PDF > 0
    size_t start = 0;
    while (start < m && ys[start] <= 0) ++start;
    size_t end = m - 1;
    while (end > 0 && ys[end] <= 0) --end;
    auto &xmin = xs[start];
    auto &xmax = xs[end];

    // Prepare output
    luisa::vector<T> quantiles;

    // Inverse CDF for each probability in ps
    for (auto &&p : std::forward<G>(ps)) {
        // Initial guess via linear interpolation on CDF
        auto it = std::lower_bound(cdf.begin(), cdf.end(), p);
        T x0;
        if (it == cdf.begin()) {
            x0 = xs[0];
        } else if (it == cdf.end()) {
            x0 = xs[m - 1];
        } else {
            size_t j = it - cdf.begin();
            auto &c1 = cdf[j - 1];
            auto &c2 = cdf[j];
            auto t = (p - c1) / (c2 - c1);
            x0 = xs[j - 1] + t * (xs[j] - xs[j - 1]);
        }

        // Newton-Raphson refinement
        for (int iters = 0; iters < iterations; ++iters) {
            // Locate interval for x0
            auto up = std::upper_bound(xs.begin(), xs.begin() + m, x0);
            size_t j = (up == xs.begin()) ? 0 : (up - xs.begin() - 1);
            size_t k = luisa::min(j + 1, m - 1);

            auto xj = xs[j], xk = xs[k];
            auto cj = cdf[j], ck = cdf[k];
            auto yj = ys[j], yk = ys[k];
            auto t = (xk == xj ? 0 : (x0 - xj) / (xk - xj));
            auto cval = cj + t * (ck - cj);
            auto pdf = yj + t * (yk - yj);

            if (pdf <= 0) break;
            auto xnew = x0 - (cval - p) / pdf;
            if (luisa::abs(xnew - x0) < tol) {
                x0 = xnew;
                break;
            }
            x0 = xnew;
        }

        // Clamp to valid support
        quantiles.emplace_back(luisa::clamp(x0, xmin, xmax));
    }

    return quantiles;
}

int main(int argc, char *argv[]) {

    log_level_verbose();

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    bool force_offline = false;
    std::optional<std::filesystem::path> compare_path;
    for (int i = 2; i < argc; i++) {
        if (std::string_view{argv[i]} == "--offline") {
            force_offline = true;
        } else if ((std::string_view{argv[i]} == "--compare" || std::string_view{argv[i]} == "-c") && i + 1 < argc) {
            compare_path = std::filesystem::path{argv[++i]};
            force_offline = true;
        }
    }
    Device device = context.create_device(argv[1]);

    // load the Cornell Box scene
    tinyobj::ObjReaderConfig obj_reader_config;
    obj_reader_config.triangulate = true;
    obj_reader_config.vertex_color = false;
    tinyobj::ObjReader obj_reader;
    if (!obj_reader.ParseFromString(obj_string, "", obj_reader_config)) {
        luisa::string_view error_message = "unknown error.";
        if (auto &&e = obj_reader.Error(); !e.empty()) { error_message = e; }
        LUISA_ERROR_WITH_LOCATION("Failed to load OBJ file: {}", error_message);
    }
    if (auto &&e = obj_reader.Warning(); !e.empty()) {
        LUISA_WARNING_WITH_LOCATION("{}", e);
    }

    auto &&p = obj_reader.GetAttrib().vertices;
    luisa::vector<float3> vertices;
    vertices.reserve(p.size() / 3u);
    for (uint i = 0u; i < p.size(); i += 3u) {
        vertices.emplace_back(make_float3(
            p[i + 0u], p[i + 1u], p[i + 2u]));
    }
    LUISA_INFO(
        "Loaded mesh with {} shape(s) and {} vertices.",
        obj_reader.GetShapes().size(), vertices.size());

    BindlessArray heap = device.create_bindless_array();
    Stream stream = device.create_stream(force_offline ? StreamTag::COMPUTE : StreamTag::GRAPHICS);
    stream.set_log_callback([](string_view s) {
        fmt::println("{}", s);
    });
    Buffer<float3> vertex_buffer = device.create_buffer<float3>(vertices.size());
    stream << vertex_buffer.copy_from(luisa::span{vertices});
    luisa::vector<Mesh> meshes;
    luisa::vector<Buffer<Triangle>> triangle_buffers;
    for (auto &&shape : obj_reader.GetShapes()) {
        uint index = static_cast<uint>(meshes.size());
        auto const &t = shape.mesh.indices;
        uint triangle_count = t.size() / 3u;
        LUISA_INFO(
            "Processing shape '{}' at index {} with {} triangle(s).",
            shape.name, index, triangle_count);
        luisa::vector<uint> indices;
        indices.reserve(t.size());
        for (tinyobj::index_t i : t) { indices.emplace_back(i.vertex_index); }
        Buffer<Triangle> &triangle_buffer = triangle_buffers.emplace_back(device.create_buffer<Triangle>(triangle_count));
        Mesh &mesh = meshes.emplace_back(device.create_mesh(vertex_buffer, triangle_buffer));
        heap.emplace_on_update(index, triangle_buffer);
        stream << triangle_buffer.copy_from(luisa::span{indices})
               << mesh.build();
    }

    Accel accel = device.create_accel({});
    for (Mesh &m : meshes) {
        accel.emplace_back(m, make_float4x4(1.0f));
    }
    stream << heap.update()
           << accel.build()
           << synchronize();

    constexpr const float wavelength_min = 360;
    constexpr const float wavelength_max = 830;

    BindlessArray lut_heap = device.create_bindless_array();
    luisa::function<uint()> lut_indexer = [index = 0]() mutable { return index++; };

    uint SRGB_TO_FOURIER_EVEN = lut_indexer();
    Volume<float> srgb_to_fourier_even;
    constexpr auto fourier_cmin = -inv_pi;
    constexpr auto fourier_cmax = inv_pi;
    {
        auto *lut_ptr = reinterpret_cast<const char *>(luisa_compute_SRGBToFourierEvenPacked);
        struct {
            size_t version, x, y, z;
        } header;
        std::memcpy(&header, lut_ptr, sizeof(header));
        assert(header.version == 0);
        auto lut_resolution = make_uint3(header.x, header.y, header.z);
        auto buffer_size = sizeof(uint) * lut_resolution.x * lut_resolution.y * lut_resolution.z;
        auto lut_data = lut_ptr + sizeof(header);
        srgb_to_fourier_even = device.create_volume<float>(PixelStorage::R10G10B10A2, lut_resolution);
        stream << lut_heap.emplace_on_update(SRGB_TO_FOURIER_EVEN, srgb_to_fourier_even, Sampler::linear_linear_mirror()).update() << srgb_to_fourier_even.copy_from(luisa::span{lut_data, static_cast<size_t>(buffer_size)})
               << synchronize();
    }

    uint CIE_XYZ_CDFINV = lut_indexer();
    Image<float> cie_xyz_cdfinv;
    {
        uint lut_resolution = 2048;
        luisa::vector<float> ps;
        ps.reserve(lut_resolution);
        for (size_t i = 0; i < lut_resolution; i++) {
            ps.emplace_back(float(i) / (lut_resolution - 1));
        }
        luisa::vector<float4> lut_data(lut_resolution);
        for (size_t c = 0; c < 3; c++) {
            luisa::vector<float> xs, ys;
            for (size_t i = 0; i <= size_t(wavelength_max - wavelength_min); i++) {
                xs.push_back(test::spectrum::CIE_xyz_1931_2deg[i].first);
                ys.push_back(test::spectrum::CIE_xyz_1931_2deg[i].second[c]);
            }
            auto result = generate_quantiles<float>(xs, ys, ps, 50, 10e-12f);
            for (size_t i = 0; i < lut_resolution; i++) {
                lut_data[i][c] = result[i];
            }
        }
        cie_xyz_cdfinv = device.create_image<float>(PixelStorage::FLOAT4, make_uint2(lut_resolution, 1));
        stream << lut_heap.emplace_on_update(CIE_XYZ_CDFINV, cie_xyz_cdfinv, Sampler::linear_linear_mirror()).update() << cie_xyz_cdfinv.copy_from(luisa::span{lut_data})
               << synchronize();
    }

    uint ILLUM_D65 = lut_indexer();
    Image<float> illum_d65;
    {
        uint step = 10;
        uint lut_resolution = uint(wavelength_max - wavelength_min) / step + 1;
        luisa::vector<float> lut_data(lut_resolution);
        auto &start = test::spectrum::CIE_std_illum_D65[0];
        for (size_t i = 0; i < lut_resolution; i++) {
            lut_data[i] = (1.0f / 98.8900106203f) * test::spectrum::CIE_std_illum_D65[size_t(wavelength_min) - start.first + i * step].second;
        }
        illum_d65 = device.create_image<float>(PixelStorage::FLOAT1, make_uint2(lut_resolution, 1));
        stream << lut_heap.emplace_on_update(ILLUM_D65, illum_d65, Sampler::linear_linear_mirror()).update() << illum_d65.copy_from(luisa::span{lut_data})
               << synchronize();
    }

    uint BMESE_PHASE = lut_indexer();
    Image<float> bmese_phase;
    {
        uint step = 5;
        uint lut_resolution = uint(wavelength_max - wavelength_min) / step + 1;
        luisa::vector<float> lut_data(lut_resolution);
        for (size_t i = 0; i < lut_resolution; i++) {
            lut_data[i] = test::spectrum::BMESE_wavelength_to_phase[i].second;
        }
        bmese_phase = device.create_image<float>(PixelStorage::FLOAT1, make_uint2(lut_resolution, 1));
        stream << lut_heap.emplace_on_update(BMESE_PHASE, bmese_phase, Sampler::linear_linear_mirror()).update() << bmese_phase.copy_from(luisa::span{lut_data})
               << synchronize();
    }

    // Constant materials{
    //     make_float3(0.725f, 0.710f, 0.680f),// floor
    //     make_float3(0.725f, 0.710f, 0.680f),// ceiling
    //     make_float3(0.725f, 0.710f, 0.680f),// back wall
    //     make_float3(0.140f, 0.450f, 0.091f),// right wall
    //     make_float3(0.630f, 0.065f, 0.050f),// left wall
    //     make_float3(0.725f, 0.710f, 0.680f),// short box
    //     make_float3(0.725f, 0.710f, 0.680f),// tall box
    //     make_float3(0.000f, 0.000f, 0.000f),// light
    // };
    float3 materials_array[] = {
        make_float3(0.725f, 0.710f, 0.680f),// floor
        make_float3(0.725f, 0.710f, 0.680f),// ceiling
        make_float3(0.725f, 0.710f, 0.680f),// back wall
        make_float3(0.140f, 0.450f, 0.091f),// right wall (green)
        make_float3(0.630f, 0.065f, 0.050f),// left wall (red)
        make_float3(0.725f, 0.710f, 0.680f),// short box
        make_float3(0.725f, 0.710f, 0.680f),// tall box
        make_float3(0.000f, 0.000f, 0.000f),// light (emissive, not used directly)
    };
    auto materials = device.create_buffer<float3>(8);
    stream << materials.copy_from(luisa::span{materials_array, std::size(materials_array)});
    Callable lsrgb_to_lagrange = [&](Float3 x) noexcept {
        auto lut = lut_heap->tex3d(SRGB_TO_FOURIER_EVEN);
        auto size = make_float3(lut.size());
        auto packed = lut.sample(x * ((size - 1.0f) / size) + 0.5f / size).xyz();
        auto trigonometric_moment = make_float3(packed.x, packed.yz() * (fourier_cmax - fourier_cmin) + fourier_cmin);
        return prepareReflectanceSpectrumLagrangeReal3(trigonometric_moment);
    };

    Callable spectrum_reflectance = [&](Float lambda, Float3 lagrange) noexcept {
        auto lut = lut_heap->tex2d(BMESE_PHASE);
        auto size = cast<float>(lut.size().x);
        auto phase = lut.sample(make_float2((lambda - wavelength_min) / (wavelength_max - wavelength_min) * ((size - 1.0f) / size) + 0.5f / size, 0.5f)).x;
        return evaluateReflectanceSpectrumLagrangeReal3(phase, lagrange);
    };

    Callable d6_normalized = [&](Float lambda) noexcept {
        auto lut = lut_heap->tex2d(ILLUM_D65);
        auto size = cast<float>(lut.size().x);
        auto radiance = lut.sample(make_float2((lambda - wavelength_min) / (wavelength_max - wavelength_min) * ((size - 1.0f) / size) + 0.5f / size, 0.5f)).x;
        return radiance;
    };

    Callable sample_xyz = [&](Float dx) noexcept {
        auto lut = lut_heap->tex2d(CIE_XYZ_CDFINV);
        auto size = cast<float>(lut.size().x);
        return lut.sample(make_float2(dx * ((size - 1.0f) / size) + 0.5f / size, 0.5f)).xyz();
    };

    Callable linear_to_srgb = [&](Var<float3> x) noexcept {
        return saturate(select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
                               12.92f * x,
                               x <= 0.00031308f));
    };

    Callable tea = [](UInt v0, UInt v1) noexcept {
        set_name("tea");
        UInt s0 = def(0u);
        for (uint n = 0u; n < 4u; n++) {
            s0 += 0x9e3779b9u;
            v0 += ((v1 << 4) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5u) + 0xc8013ea4u);
            v1 += ((v0 << 4) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5u) + 0x7e95761eu);
        }
        return v0;
    };

    Kernel2D make_sampler_kernel = [&](ImageUInt seed_image) noexcept {
        set_name("make_sampler_kernel");
        UInt2 p = dispatch_id().xy();
        UInt state = tea(p.x, p.y);
        seed_image.write(p, make_uint4(state));
    };

    Callable lcg = [](UInt &state) noexcept {
        set_name("lcg");
        constexpr uint lcg_a = 1664525u;
        constexpr uint lcg_c = 1013904223u;
        state = lcg_a * state + lcg_c;
        return cast<float>(state & 0x00ffffffu) *
               (1.0f / static_cast<float>(0x01000000u));
    };

    Callable make_onb = [](const Float3 &normal) noexcept {
        set_name("make_onb");
        Float3 binormal = normalize(ite(
            abs(normal.x) > abs(normal.z),
            make_float3(-normal.y, normal.x, 0.0f),
            make_float3(0.0f, -normal.z, normal.y)));
        Float3 tangent = normalize(cross(binormal, normal));
        return def<Onb>(tangent, binormal, normal);
    };

    Callable generate_ray = [](Float2 p) noexcept {
        set_name("generate_ray");
        static constexpr float fov = radians(27.8f);
        static constexpr float3 origin = make_float3(-0.01f, 0.995f, 5.0f);
        Float3 pixel = origin + make_float3(p * tan(0.5f * fov), -1.0f);
        Float3 direction = normalize(pixel - origin);
        return make_ray(origin, direction);
    };

    Callable cosine_sample_hemisphere = [](Float2 u) noexcept {
        set_name("cosine_sample_hemisphere");
        Float r = sqrt(u.x);
        Float phi = 2.0f * constants::pi * u.y;
        return make_float3(r * cos(phi), r * sin(phi), sqrt(1.0f - u.x));
    };

    Callable balanced_heuristic = [](Float pdf_a, Float pdf_b) noexcept {
        set_name("balanced_heuristic");
        return pdf_a / max(pdf_a + pdf_b, 1e-4f);
    };

    auto spp_per_dispatch = device.backend_name() == "metal" || device.backend_name() == "cpu" || device.backend_name() == "fallback" ? 1u : 64u;
    bool infinite_render = !force_offline;
    uint total_spp = force_offline ? 256u : 0u;

    Kernel2D raytracing_kernel = [&](ImageFloat image, ImageUInt seed_image, AccelVar accel, UInt2 resolution) noexcept {
        set_name("raytracing_kernel");
        set_block_size(16u, 16u, 1u);
        UInt2 coord = dispatch_id().xy();
        Float frame_size = min(resolution.x, resolution.y).cast<float>();
        UInt state = seed_image.read(coord).x;
        Float rx = lcg(state);
        Float ry = lcg(state);
        Float2 pixel = (make_float2(coord) + make_float2(rx, ry)) / frame_size * 2.0f - 1.0f;
        Float3 radiance = def(make_float3(0.0f));
        Bool use_spectrum = (block_id().x / 4 + block_id().y / 4) % 2 == 1;

        $for (i, spp_per_dispatch) {
            Var<Ray> ray = generate_ray(pixel * make_float2(1.0f, -1.0f));

            Float3 lambda = sample_xyz(lcg(state));
            Float3 light_emission = def(make_float3(17.0f, 12.0f, 4.0f));
            $if (use_spectrum) {
                auto max_emission = max(max(light_emission.x, light_emission.y), light_emission.z);
                auto lagrange = lsrgb_to_lagrange(light_emission / max_emission);
                auto color = make_float3(spectrum_reflectance(lambda.x, lagrange), spectrum_reflectance(lambda.y, lagrange), spectrum_reflectance(lambda.z, lagrange));
                light_emission = max_emission * color * make_float3(d6_normalized(lambda.x), d6_normalized(lambda.y), d6_normalized(lambda.z));
            };

            Float3 beta = def(make_float3(1.0f));
            Float pdf_bsdf = def(0.0f);
            constexpr float3 light_position = make_float3(-0.24f, 1.98f, 0.16f);
            constexpr float3 light_u = make_float3(-0.24f, 1.98f, -0.22f) - light_position;
            constexpr float3 light_v = make_float3(0.23f, 1.98f, 0.16f) - light_position;
            Float light_area = length(cross(light_u, light_v));
            Float3 light_normal = normalize(cross(light_u, light_v));
            $for (depth, 10u) {
                // trace
                Var<TriangleHit> hit = accel.intersect(ray, {});
                reorder_shader_execution();
                $if (hit->miss()) { $break; };
                Var<Triangle> triangle = heap->buffer<Triangle>(hit.inst).read(hit.prim);
                Float3 p0 = vertex_buffer->read(triangle.i0);
                Float3 p1 = vertex_buffer->read(triangle.i1);
                Float3 p2 = vertex_buffer->read(triangle.i2);
                Float3 p = triangle_interpolate(hit.bary, p0, p1, p2);
                Float3 n = normalize(cross(p1 - p0, p2 - p0));
                Float cos_wo = dot(-ray->direction(), n);
                $if (cos_wo < 1e-4f) { $break; };

                // hit light
                $if (hit.inst == static_cast<uint>(meshes.size() - 1u)) {
                    $if (depth == 0u) {
                        radiance += light_emission;
                    }
                    $else {
                        Float pdf_light = length_squared(p - ray->origin()) / (light_area * cos_wo);
                        Float mis_weight = balanced_heuristic(pdf_bsdf, pdf_light);
                        radiance += mis_weight * beta * light_emission;
                    };
                    $break;
                };

                // sample light
                Float ux_light = lcg(state);
                Float uy_light = lcg(state);
                Float3 p_light = light_position + ux_light * light_u + uy_light * light_v;
                Float3 pp = offset_ray_origin(p, n);
                Float3 pp_light = offset_ray_origin(p_light, light_normal);
                Float d_light = distance(pp, pp_light);
                Float3 wi_light = normalize(pp_light - pp);
                Var<Ray> shadow_ray = make_ray(offset_ray_origin(pp, n), wi_light, 0.f, d_light);
                Bool occluded = accel.intersect_any(shadow_ray, {});
                Float cos_wi_light = dot(wi_light, n);
                Float cos_light = -dot(light_normal, wi_light);
                Float3 albedo = materials->read(hit.inst);
                $if (use_spectrum) {
                    auto lagrange = lsrgb_to_lagrange(albedo);
                    albedo = make_float3(spectrum_reflectance(lambda.x, lagrange), spectrum_reflectance(lambda.y, lagrange), spectrum_reflectance(lambda.z, lagrange));
                };

                $if (!occluded & cos_wi_light > 1e-4f & cos_light > 1e-4f) {
                    Float pdf_light = (d_light * d_light) / (light_area * cos_light);
                    Float pdf_bsdf = cos_wi_light * inv_pi;
                    Float mis_weight = balanced_heuristic(pdf_light, pdf_bsdf);
                    Float3 bsdf = albedo * inv_pi * cos_wi_light;
                    radiance += beta * bsdf * mis_weight * light_emission / max(pdf_light, 1e-4f);
                };

                // sample BSDF
                Var<Onb> onb = make_onb(n);
                Float ux = lcg(state);
                Float uy = lcg(state);
                Float3 wi_local = cosine_sample_hemisphere(make_float2(ux, uy));
                Float cos_wi = abs(wi_local.z);
                Float3 new_direction = onb->to_world(wi_local);
                ray = make_ray(pp, new_direction);
                pdf_bsdf = cos_wi * inv_pi;
                beta *= albedo;// * cos_wi * inv_pi / pdf_bsdf => * 1.f

                // rr
                Float l = dot(make_float3(0.212671f, 0.715160f, 0.072169f), beta);
                $if (use_spectrum) {
                    l = (beta.x + beta.y + beta.z) * (1.0f / 3.0f);
                };
                $if (l == 0.0f) { $break; };
                Float q = max(l, 0.05f);
                Float r = lcg(state);
                $if (r >= q) { $break; };
                beta *= 1.0f / q;
            };
        };
        radiance /= static_cast<float>(spp_per_dispatch);
        seed_image.write(coord, make_uint4(state));
        $if (any(dsl::isnan(radiance))) { radiance = make_float3(0.0f); };

        constexpr const auto XYZtoRGB_Rec709 = transpose(float3x3{
            {3.2409699419045213f, -1.5373831775700935f, -0.4986107602930033f},
            {-0.9692436362808798f, 1.8759675015077206f, 0.0415550574071756f},
            {0.0556300796969936f, -0.2039769588889765f, 1.0569715142428784f}});

        $if (use_spectrum) {
            radiance = XYZtoRGB_Rec709 * radiance;
        };

        image.write(dispatch_id().xy(), make_float4(clamp(radiance, 0.0f, 30.0f), 1.0f));
    };

    Kernel2D accumulate_kernel = [&](ImageFloat accum_image, ImageFloat curr_image) noexcept {
        set_name("accumulate_kernel");
        UInt2 p = dispatch_id().xy();
        Float4 accum = accum_image.read(p);
        Float3 curr = curr_image.read(p).xyz();
        accum_image.write(p, accum + make_float4(curr, 1.f));
    };

    Kernel2D clear_kernel = [](ImageFloat image) noexcept {
        set_name("clear_kernel");
        image.write(dispatch_id().xy(), make_float4(0.f));
    };

    Kernel2D hdr2ldr_kernel = [&](ImageFloat hdr_image, ImageFloat ldr_image, Float scale) noexcept {
        set_name("hdr2ldr_kernel");
        UInt2 coord = dispatch_id().xy();
        Float4 hdr = hdr_image.read(coord);
        Float3 ldr = linear_to_srgb(clamp(hdr.xyz() / hdr.w * scale, 0.f, 1.f));
        ldr_image.write(coord, make_float4(ldr, 1.f));
    };

    ShaderOption o{.enable_debug_info = false};
    auto raytracing_shader = device.compile(raytracing_kernel, ShaderOption{.name = "path_tracing"});
    auto clear_shader = device.compile(clear_kernel, o);
    auto hdr2ldr_shader = device.compile(hdr2ldr_kernel, o);
    auto accumulate_shader = device.compile(accumulate_kernel, o);
    auto make_sampler_shader = device.compile(make_sampler_kernel, o);

    static constexpr uint2 resolution = make_uint2(1024u);
    Image<float> framebuffer = device.create_image<float>(PixelStorage::HALF4, resolution);
    Image<float> accum_image = device.create_image<float>(PixelStorage::FLOAT4, resolution);
    luisa::vector<std::array<uint8_t, 4u>> host_image(resolution.x * resolution.y);

    Image<uint> seed_image = device.create_image<uint>(PixelStorage::INT1, resolution);
    stream << clear_shader(accum_image).dispatch(resolution)
           << make_sampler_shader(seed_image).dispatch(resolution);

    std::unique_ptr<Window> window;
    std::optional<Swapchain> swap_chain;
    if (!force_offline) {
        window = std::make_unique<Window>("path tracing", resolution);
        swap_chain.emplace(device.create_swapchain(
            stream,
            SwapchainOption{
                .display = window->native_display(),
                .window = window->native_handle(),
                .size = make_uint2(resolution),
                .wants_hdr = false,
                .wants_vsync = false,
                .back_buffer_count = 8,
            }));
    }

    Image<float> ldr_image = device.create_image<float>(
        (!force_offline && swap_chain.has_value()) ? swap_chain->backend_storage() : PixelStorage::BYTE4,
        resolution);
    double last_time = 0.0;
    uint frame_count = 0u;
    Clock clock;

    while (infinite_render || frame_count < total_spp) {
        stream << raytracing_shader(framebuffer, seed_image, accel, resolution)
                      .dispatch(resolution)
               << accumulate_shader(accum_image, framebuffer)
                      .dispatch(resolution)
               << hdr2ldr_shader(accum_image, ldr_image, 2.f).dispatch(resolution);
        if (!force_offline && swap_chain.has_value()) {
            stream << swap_chain->present(ldr_image)
                   << synchronize();
            window->poll_events();
            if (window->should_close()) { break; }
        }
        double dt = clock.toc() - last_time;
        LUISA_INFO("dt = {:.2f}ms ({:.2f} spp/s)", dt, spp_per_dispatch / dt * 1000);
        last_time = clock.toc();
        frame_count += spp_per_dispatch;
    }
    stream
        << ldr_image.copy_to(luisa::span{host_image})
        << synchronize();
    LUISA_INFO("FPS: {}", frame_count / clock.toc() * 1000);
    stbi_write_png("test_path_tracing_spectrum.png", resolution.x, resolution.y, 4, host_image.data(), 0);
    if (force_offline) {
        if (compare_path) {
            auto result = luisa::ref::compare_with_reference_file(
                reinterpret_cast<const uint8_t *>(host_image.data()),
                resolution.x, resolution.y, 4,
                *compare_path);
            LUISA_INFO("Reference comparison: {} ({})", result.passed ? "PASSED" : "FAILED", result.message);
            if (!result.passed) { return 1; }
        }
    }
    return 0;
}
