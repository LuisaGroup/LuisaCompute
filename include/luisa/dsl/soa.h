#pragma once

#include <luisa/dsl/var.h>
#include <luisa/dsl/atomic.h>
#include <luisa/dsl/builtin.h>
#include <luisa/dsl/func.h>
#include <luisa/runtime/shader.h>

namespace luisa::compute {

template<typename T>
class SOAView;

template<typename T>
class SOA;

constexpr auto soa_cache_line_size = 32u;

template<typename T>
[[nodiscard]] inline auto align_to_soa_cache_line(T size) noexcept {
    return (size + (soa_cache_line_size - 1u)) / soa_cache_line_size * soa_cache_line_size;
}

namespace detail {

struct SOAExprBase {
private:
    Expr<Buffer<uint>> _buffer;
    Expr<uint> _soa_offset;
    Expr<uint> _soa_size;
    Expr<uint> _element_offset;

public:
    SOAExprBase(Expr<Buffer<uint>> buffer,
                Expr<uint> soa_offset,
                Expr<uint> soa_size,
                Expr<uint> element_offset) noexcept
        : _buffer{buffer},
          _soa_offset{soa_offset},
          _soa_size{soa_size},
          _element_offset{element_offset} {}
    [[nodiscard]] auto buffer() const noexcept { return _buffer; }
    [[nodiscard]] auto soa_offset() const noexcept { return _soa_offset; }
    [[nodiscard]] auto soa_size() const noexcept { return _soa_size; }
    [[nodiscard]] auto element_offset() const noexcept { return _element_offset; }
};

}// namespace detail

template<typename T>
struct Expr<SOA<T>> : public detail::SOAExprBase {

private:
    static constexpr auto element_stride = SOAView<T>::element_stride;

public:
    Expr(Expr<Buffer<uint>> buffer,
         Expr<uint> soa_offset,
         Expr<uint> soa_size,
         Expr<uint> elem_offset) noexcept
        : detail::SOAExprBase{buffer, soa_offset, soa_size, elem_offset} {}

    Expr(SOAView<T> soa) noexcept
        : Expr{soa.buffer(), soa.soa_offset(), soa.soa_size(), soa.element_offset()} {}

    Expr(const SOA<T> &soa) noexcept
        : Expr{soa.view()} {}

    template<typename I>
    [[nodiscard]] auto read(I &&index) const noexcept {
        if constexpr (element_stride == 1u) {
            auto i = soa_offset() + std::forward<I>(index) + element_offset();
            auto x = buffer().read(i);
            if constexpr (sizeof(T) == sizeof(uint)) {
                return x.template as<T>();
            } else if constexpr (sizeof(T) * 2u == sizeof(uint)) {// 16bit
                if constexpr (is_scalar_v<T>) {
                    auto u = x.template as<Vector<T, 2u>>();
                    return u.x;
                } else {
                    static_assert(std::is_same_v<T, bool2>);
                    return x.template as<bool4>().xy();
                }
            } else if constexpr (sizeof(T) * 4u == sizeof(uint)) {// 8bit
                static_assert(is_scalar_v<T>);
                auto u = x.template as<Vector<T, 4u>>();
                return u.x;
            } else {// unreachable
                static_assert(sizeof(T) == sizeof(uint));
            }
        } else if constexpr (element_stride == 2u) {
            auto i = soa_offset() + (std::forward<I>(index) + element_offset()) * 2u;
            auto u = dsl::make_uint2(buffer().read(i),
                                     buffer().read(i + 1u));
            return u.template as<T>();
        } else {
            static_assert(element_stride == 4u);
            auto i = soa_offset() + (std::forward<I>(index) + element_offset()) * 4u;
            auto u = dsl::make_uint4(buffer().read(i),
                                     buffer().read(i + 1u),
                                     buffer().read(i + 2u),
                                     buffer().read(i + 3u));
            return u.template as<T>();
        }
    }

    template<typename I>
    void write(I &&index, Expr<T> value) const noexcept {
        if constexpr (element_stride == 1u) {
            auto i = soa_offset() + std::forward<I>(index) + element_offset();
            if constexpr (sizeof(T) == sizeof(uint)) {
                buffer().write(i, value.template as<uint>());
            } else if constexpr (sizeof(T) * 2u == sizeof(uint)) {// 16bit
                if constexpr (is_scalar_v<T>) {
                    auto u = def<Vector<T, 2u>>();
                    u.x = value;
                    buffer().write(i, u.template as<uint>());
                } else {
                    static_assert(std::is_same_v<T, bool2>);
                    auto u = make_bool4(value, make_bool2());
                    buffer().write(i, u.template as<uint>());
                }
            } else if constexpr (sizeof(T) * 4u == sizeof(uint)) {
                static_assert(is_scalar_v<T>);
                auto u = def<Vector<T, 4u>>();
                u.x = value;
                buffer().write(i, u.template as<uint>());
            } else {// unreachable
                static_assert(sizeof(T) == sizeof(uint));
            }
        } else if constexpr (element_stride == 2u) {
            auto i = soa_offset() + (std::forward<I>(index) + element_offset()) * 2u;
            auto u = value.template as<Vector<uint, 2>>();
            buffer().write(i, u.x);
            buffer().write(i + 1u, u.y);
        } else {
            static_assert(element_stride == 4u);
            auto i = soa_offset() + (std::forward<I>(index) + element_offset()) * 4u;
            auto u = value.template as<Vector<uint, 4>>();
            buffer().write(i, u.x);
            buffer().write(i + 1u, u.y);
            buffer().write(i + 2u, u.z);
            buffer().write(i + 3u, u.w);
        }
    }

    [[nodiscard]] auto operator->() const noexcept { return this; }
};

template<typename T>
    requires(sizeof(T) >= sizeof(uint))// if T is smaller than uint, it is not splittable
struct Expr<SOA<Vector<T, 2>>> : public detail::SOAExprBase {

public:
    Expr<SOA<T>> x;
    Expr<SOA<T>> y;

public:
    Expr(Expr<Buffer<uint>> buffer,
         Expr<uint> soa_offset,
         Expr<uint> soa_size,
         Expr<uint> elem_offset) noexcept
        : detail::SOAExprBase{buffer, soa_offset, soa_size, elem_offset},
          x{buffer, soa_offset, soa_size, elem_offset},
          y{buffer, soa_offset + SOA<T>::compute_soa_size(soa_size), soa_size, elem_offset} {}

    Expr(SOAView<Vector<T, 2>> soa) noexcept
        : Expr{soa.buffer(), soa.soa_offset(), soa.soa_size(), soa.element_offset()} {}

    Expr(const SOA<Vector<T, 2>> &soa) noexcept
        : Expr{soa.view()} {}

    template<typename I>
    [[nodiscard]] auto read(I &&index) const noexcept {
        auto i = def(std::forward<I>(index));
        auto x = this->x.read(i);
        auto y = this->y.read(i);
        return def<Vector<T, 2>>(x, y);
    }

    template<typename I>
    void write(I &&index, Expr<Vector<T, 2>> value) const noexcept {
        auto i = def(std::forward<I>(index));
        x.write(i, value.x);
        y.write(i, value.y);
    }

    [[nodiscard]] auto operator->() const noexcept { return this; }
};

template<typename T>
    requires(sizeof(T) >= sizeof(uint))// if T is smaller than uint, it is not splittable
struct Expr<SOA<Vector<T, 3>>> : public detail::SOAExprBase {

public:
    Expr<SOA<T>> x;
    Expr<SOA<T>> y;
    Expr<SOA<T>> z;

public:
    Expr(Expr<Buffer<uint>> buffer,
         Expr<uint> soa_offset,
         Expr<uint> soa_size,
         Expr<uint> elem_offset) noexcept
        : detail::SOAExprBase{buffer, soa_offset, soa_size, elem_offset},
          x{buffer, soa_offset, soa_size, elem_offset},
          y{buffer, soa_offset + SOA<T>::compute_soa_size(soa_size), soa_size, elem_offset},
          z{buffer, soa_offset + SOA<T>::compute_soa_size(soa_size) * 2u, soa_size, elem_offset} {}

    Expr(SOAView<Vector<T, 3>> soa) noexcept
        : Expr{soa.buffer(), soa.soa_offset(), soa.soa_size(), soa.element_offset()} {}

    Expr(const SOA<Vector<T, 3>> &soa) noexcept
        : Expr{soa.view()} {}

    template<typename I>
    [[nodiscard]] auto read(I &&index) const noexcept {
        auto i = def(std::forward<I>(index));
        auto x = this->x.read(i);
        auto y = this->y.read(i);
        auto z = this->z.read(i);
        return def<Vector<T, 3>>(x, y, z);
    }

    template<typename I>
    void write(I &&index, Expr<Vector<T, 3>> value) const noexcept {
        auto i = def(std::forward<I>(index));
        x.write(i, value.x);
        y.write(i, value.y);
        z.write(i, value.z);
    }

    [[nodiscard]] auto operator->() const noexcept { return this; }
};

template<typename T>
    requires(sizeof(T) >= sizeof(uint))// if T is smaller than uint, it is not splittable
struct Expr<SOA<Vector<T, 4>>> : public detail::SOAExprBase {

public:
    Expr<SOA<T>> x;
    Expr<SOA<T>> y;
    Expr<SOA<T>> z;
    Expr<SOA<T>> w;

public:
    Expr(Expr<Buffer<uint>> buffer,
         Expr<uint> soa_offset,
         Expr<uint> soa_size,
         Expr<uint> elem_offset) noexcept
        : detail::SOAExprBase{buffer, soa_offset, soa_size, elem_offset},
          x{buffer, soa_offset, soa_size, elem_offset},
          y{buffer, soa_offset + SOA<T>::compute_soa_size(soa_size), soa_size, elem_offset},
          z{buffer, soa_offset + SOA<T>::compute_soa_size(soa_size) * 2u, soa_size, elem_offset},
          w{buffer, soa_offset + SOA<T>::compute_soa_size(soa_size) * 3u, soa_size, elem_offset} {}

    Expr(SOAView<Vector<T, 4>> soa) noexcept
        : Expr{soa.buffer(), soa.soa_offset(), soa.soa_size(), soa.element_offset()} {}

    Expr(const SOA<Vector<T, 4>> &soa) noexcept
        : Expr{soa.view()} {}

    template<typename I>
    [[nodiscard]] auto read(I &&index) const noexcept {
        auto i = def(std::forward<I>(index));
        auto x = this->x.read(i);
        auto y = this->y.read(i);
        auto z = this->z.read(i);
        auto w = this->w.read(i);
        return def<Vector<T, 4>>(x, y, z, w);
    }

    template<typename I>
    void write(I &&index, Expr<Vector<T, 4>> value) const noexcept {
        auto i = def(std::forward<I>(index));
        x.write(i, value.x);
        y.write(i, value.y);
        z.write(i, value.z);
        w.write(i, value.w);
    }

    [[nodiscard]] auto operator->() const noexcept { return this; }
};

template<size_t N>
struct Expr<SOA<Matrix<N>>> : public detail::SOAExprBase {

private:
    using Column = Vector<float, N>;
    Expr<SOA<Column>> _cols[N];

private:
    template<size_t... i>
    Expr(Expr<Buffer<uint>> buffer,
         Expr<uint> soa_offset,
         Expr<uint> soa_size,
         Expr<uint> elem_offset,
         std::index_sequence<i...>) noexcept
        : detail::SOAExprBase{buffer, soa_offset, soa_size, elem_offset},
          _cols{Expr<SOA<Column>>{buffer, soa_offset + SOA<Column>::compute_soa_size(soa_size) * static_cast<uint>(i),
                                  soa_size, elem_offset}...} {}

public:
    Expr(SOAView<Matrix<N>> soa) noexcept
        : Expr{soa.buffer(), soa.soa_offset(), soa.soa_size(), soa.element_offset()} {}

    Expr(const SOA<Matrix<N>> &soa) noexcept
        : Expr{soa.view()} {}

    Expr(Expr<Buffer<uint>> buffer,
         Expr<uint> soa_offset,
         Expr<uint> soa_size,
         Expr<uint> elem_offset) noexcept
        : Expr{buffer, soa_offset, soa_size, elem_offset, std::make_index_sequence<N>{}} {}

    template<typename I>
    [[nodiscard]] auto read(I &&index) const noexcept {
        auto m = def<Matrix<N>>();
        auto i = def(std::forward<I>(index));
        for (auto c = 0u; c < N; c++) {
            m[c] = this->_cols[c].read(i);
        }
        return m;
    }

    template<typename I>
    void write(I &&index, Expr<Matrix<N>> value) const noexcept {
        auto i = def(std::forward<I>(index));
        for (auto c = 0u; c < N; c++) {
            this->_cols[c].write(i, value[c]);
        }
    }

    [[nodiscard]] auto operator[](size_t i) const noexcept { return _cols[i]; }
    [[nodiscard]] auto operator->() const noexcept { return this; }
};

template<typename T, size_t N>
    requires(sizeof(T) >= sizeof(uint))// if T is smaller than uint, we do not split it
struct Expr<SOA<std::array<T, N>>> : public detail::SOAExprBase {

private:
    using Array = std::array<T, N>;
    Expr<SOA<T>> _elems[N];

private:
    template<size_t... i>
    Expr(Expr<Buffer<uint>> buffer,
         Expr<uint> soa_offset,
         Expr<uint> soa_size,
         Expr<uint> elem_offset,
         std::index_sequence<i...>) noexcept
        : detail::SOAExprBase{buffer, soa_offset, soa_size, elem_offset},
          _elems{Expr<SOA<T>>{buffer, soa_offset + SOA<T>::compute_soa_size(soa_size) * static_cast<uint>(i),
                              soa_size, elem_offset}...} {}

public:
    Expr(SOAView<Array> soa) noexcept
        : Expr{soa.buffer(), soa.soa_offset(), soa.soa_size(), soa.element_offset()} {}

    Expr(const SOA<Array> &soa) noexcept
        : Expr{soa.view()} {}

    Expr(Expr<Buffer<uint>> buffer,
         Expr<uint> soa_offset,
         Expr<uint> soa_size,
         Expr<uint> elem_offset) noexcept
        : Expr{buffer, soa_offset, soa_size, elem_offset,
               std::make_index_sequence<N>{}} {}

    template<typename I>
    [[nodiscard]] auto read(I &&index) const noexcept {
        auto a = def<Array>();
        auto i = def(std::forward<I>(index));
        for (auto c = 0u; c < N; c++) {
            a[c] = this->_elems[c].read(i);
        }
        return a;
    }

    template<typename I>
    void write(I &&index, Expr<Array> value) const noexcept {
        auto i = def(std::forward<I>(index));
        for (auto c = 0u; c < N; c++) {
            this->_elems[c].write(i, value[c]);
        }
    }
    [[nodiscard]] auto operator[](size_t i) const noexcept { return _elems[i]; }
    [[nodiscard]] auto operator->() const noexcept { return this; }
};

template<typename T, size_t N>
struct Expr<SOA<T[N]>> : public Expr<SOA<std::array<T, N>>> {
    using Expr<SOA<std::array<T, N>>>::Expr;
};

template<typename T>
struct Expr<SOAView<T>> : public Expr<SOA<T>> {
    using Expr<SOA<T>>::Expr;
};

template<typename T>
Expr(SOAView<T>) -> Expr<SOAView<T>>;

template<typename T>
Expr(const SOA<T> &) -> Expr<SOA<T>>;

namespace detail {

template<typename T>
void callable_encode_soa(CallableInvoke &invoke, Expr<T> soa) {
    invoke << soa.buffer()
           << soa.soa_offset()
           << soa.soa_size()
           << soa.element_offset();
}

}// namespace detail

template<typename T>
struct Var<SOA<T>> : public Expr<SOA<T>> {

private:
    using Base = Expr<SOA<T>>;

    Var(Expr<Buffer<uint>> buffer,
        Expr<uint> soa_offset,
        Expr<uint> soa_size,
        Expr<uint> elem_offset) noexcept
        : Base{buffer, soa_offset, soa_size, elem_offset} {}

    // make the call sequential
    Var(Expr<Buffer<uint>> buffer,
        Expr<uint> soa_offset,
        Expr<uint> soa_size) noexcept
        : Var{buffer, soa_offset, soa_size,
              Var<uint>{detail::ArgumentCreation{}}} {}

    Var(Expr<Buffer<uint>> buffer,
        Expr<uint> soa_offset) noexcept
        : Var{buffer, soa_offset,
              Var<uint>{detail::ArgumentCreation{}}} {}

    Var(Expr<Buffer<uint>> buffer) noexcept
        : Var{buffer,
              Var<uint>{detail::ArgumentCreation{}}} {}

public:
    Var(detail::ArgumentCreation) noexcept
        : Var{Var<Buffer<uint>>{detail::ArgumentCreation{}}} {}
    [[nodiscard]] explicit operator Expr<SOAView<T>>() const noexcept { return Base{*this}; }
    [[nodiscard]] explicit operator Var<SOAView<T>>() const noexcept { return Expr<SOAView<T>>{*this}; }
};

template<typename T>
struct Var<SOAView<T>> : public Var<SOA<T>> {
    using Var<SOA<T>>::Var;
};

namespace detail {

template<typename T>
struct shader_argument_encode_count<SOA<T>> {
    static constexpr uint value = 4u;
};

template<typename T>
struct shader_argument_encode_count<SOAView<T>>
    : public shader_argument_encode_count<SOA<T>> {};

template<typename T>
ShaderInvokeBase &ShaderInvokeBase::operator<<(SOAView<T> soa) noexcept {
    return *this << soa.buffer()
                 << soa.soa_offset()
                 << soa.soa_size()
                 << soa.element_offset();
}

template<typename T>
ShaderInvokeBase &ShaderInvokeBase::operator<<(const SOA<T> &soa) noexcept {
    return *this << soa.view();
}

LC_DSL_API void error_soa_subview_out_of_range() noexcept;
LC_DSL_API void error_soa_view_exceeds_uint_max() noexcept;
LC_DSL_API void error_soa_index_out_of_range() noexcept;

template<typename T>
class SOAViewBase {

private:
    using View = SOAView<T>;

private:
    BufferView<uint> _buffer;
    uint _soa_offset{};
    uint _soa_size{};
    uint _elem_offset{};
    uint _elem_size{};

public:
    SOAViewBase() noexcept = default;
    SOAViewBase(BufferView<uint> buffer,
                size_t soa_offset,
                size_t soa_size,
                size_t elem_offset,
                size_t elem_size) noexcept
        : _buffer{buffer},
          _soa_offset{static_cast<uint>(soa_offset)},
          _soa_size{static_cast<uint>(soa_size)},
          _elem_offset{static_cast<uint>(elem_offset)},
          _elem_size{static_cast<uint>(elem_size)} {}
    [[nodiscard]] auto buffer() const noexcept { return _buffer; }
    [[nodiscard]] auto soa_offset() const noexcept { return _soa_offset; }
    [[nodiscard]] auto soa_size() const noexcept { return _soa_size; }
    [[nodiscard]] auto element_offset() const noexcept { return _elem_offset; }
    [[nodiscard]] auto element_size() const noexcept { return _elem_size; }
    [[nodiscard]] auto operator->() const noexcept { return Expr<View>{*reinterpret_cast<const View *>(this)}; }
    [[nodiscard]] auto subview(size_t offset, size_t size) const noexcept {
        if (!(offset + size <= this->element_size())) [[unlikely]] {
            error_soa_subview_out_of_range();
        }
        return View{this->buffer(),
                    this->soa_offset(),
                    this->soa_size(),
                    this->element_offset() + offset,
                    size};
    }
};

}// namespace detail

template<typename T>
class SOAView : public detail::SOAViewBase<T> {

public:
    static constexpr auto element_stride =
        static_cast<uint>((sizeof(T) + sizeof(uint) - 1u) / sizeof(uint));

public:
    [[nodiscard]] static auto compute_soa_size(auto n) noexcept {
        return align_to_soa_cache_line(n * element_stride);
    }

public:
    SOAView() noexcept = default;
    SOAView(BufferView<uint> buffer,
            size_t soa_offset, size_t soa_size,
            size_t elem_offset, size_t elem_size) noexcept
        : detail::SOAViewBase<T> { buffer, soa_offset, soa_size, elem_offset, elem_size }
    {
        auto buffer_end = this->buffer().offset() + soa_offset +
                          (elem_offset + elem_size) * element_stride;
        if (!(buffer_end <= std::numeric_limits<uint>::max())) [[unlikely]] {
            detail::error_soa_view_exceeds_uint_max();
        }
    }
};

template<typename T>
    requires(sizeof(T) >= sizeof(uint))
class SOAView<Vector<T, 2>> : public detail::SOAViewBase<Vector<T, 2>> {

public:
    SOAView<T> x;
    SOAView<T> y;

public:
    [[nodiscard]] static auto compute_soa_size(auto n) noexcept {
        return SOAView<T>::compute_soa_size(n) * 2u;
    }

public:
    SOAView() noexcept = default;
    SOAView(BufferView<uint> buffer,
            size_t soa_offset, size_t soa_size,
            size_t elem_offset, size_t elem_size) noexcept
        : detail::SOAViewBase<Vector<T, 2>>{buffer, soa_offset, soa_size, elem_offset, elem_size},
          x{buffer, soa_offset + SOAView<T>::compute_soa_size(soa_size) * 0u, soa_size, elem_offset, elem_size},
          y{buffer, soa_offset + SOAView<T>::compute_soa_size(soa_size) * 1u, soa_size, elem_offset, elem_size} {}

public:
    [[nodiscard]] auto operator[](size_t i) const noexcept {
        if (i == 0u) {
            return this->x;
        } else if (i == 1u) {
            return this->y;
        } else {
            detail::error_soa_index_out_of_range();
        }
    }
};

template<typename T>
    requires(sizeof(T) >= sizeof(uint))
class SOAView<Vector<T, 3>> : public detail::SOAViewBase<Vector<T, 3>> {

public:
    SOAView<T> x;
    SOAView<T> y;
    SOAView<T> z;

public:
    [[nodiscard]] static auto compute_soa_size(auto n) noexcept {
        return SOAView<T>::compute_soa_size(n) * 3u;
    }

public:
    SOAView() noexcept = default;
    SOAView(BufferView<uint> buffer,
            size_t soa_offset, size_t soa_size,
            size_t elem_offset, size_t elem_size) noexcept
        : detail::SOAViewBase<Vector<T, 3>>{buffer, soa_offset, soa_size, elem_offset, elem_size},
          x{buffer, soa_offset + SOAView<T>::compute_soa_size(soa_size) * 0u, soa_size, elem_offset, elem_size},
          y{buffer, soa_offset + SOAView<T>::compute_soa_size(soa_size) * 1u, soa_size, elem_offset, elem_size},
          z{buffer, soa_offset + SOAView<T>::compute_soa_size(soa_size) * 2u, soa_size, elem_offset, elem_size} {}

public:
    [[nodiscard]] auto operator[](size_t i) const noexcept {
        if (i == 0u) {
            return this->x;
        } else if (i == 1u) {
            return this->y;
        } else if (i == 2u) {
            return this->z;
        } else {
            detail::error_soa_index_out_of_range();
        }
    }
};

template<typename T>
    requires(sizeof(T) >= sizeof(uint))
class SOAView<Vector<T, 4>> : public detail::SOAViewBase<Vector<T, 4>> {

public:
    SOAView<T> x;
    SOAView<T> y;
    SOAView<T> z;
    SOAView<T> w;

public:
    [[nodiscard]] static auto compute_soa_size(auto n) noexcept {
        return SOAView<T>::compute_soa_size(n) * 4u;
    }

public:
    SOAView() noexcept = default;
    SOAView(BufferView<uint> buffer,
            size_t soa_offset, size_t soa_size,
            size_t elem_offset, size_t elem_size) noexcept
        : detail::SOAViewBase<Vector<T, 4>>{buffer, soa_offset, soa_size, elem_offset, elem_size},
          x{buffer, soa_offset + SOAView<T>::compute_soa_size(soa_size) * 0u, soa_size, elem_offset, elem_size},
          y{buffer, soa_offset + SOAView<T>::compute_soa_size(soa_size) * 1u, soa_size, elem_offset, elem_size},
          z{buffer, soa_offset + SOAView<T>::compute_soa_size(soa_size) * 2u, soa_size, elem_offset, elem_size},
          w{buffer, soa_offset + SOAView<T>::compute_soa_size(soa_size) * 3u, soa_size, elem_offset, elem_size} {}

public:
    [[nodiscard]] auto operator[](size_t i) const noexcept {
        if (i == 0u) {
            return this->x;
        } else if (i == 1u) {
            return this->y;
        } else if (i == 2u) {
            return this->z;
        } else if (i == 3u) {
            return this->w;
        } else {
            detail::error_soa_index_out_of_range();
        }
    }
};

template<size_t N>
class SOAView<Matrix<N>> : public detail::SOAViewBase<Matrix<N>> {

private:
    using Column = Vector<float, N>;
    SOAView<Column> _cols[N];

public:
    [[nodiscard]] static auto compute_soa_size(auto n) noexcept {
        return SOAView<Column>::compute_soa_size(n) * static_cast<uint>(N);
    }

public:
    SOAView() noexcept = default;
    SOAView(BufferView<uint> buffer,
            size_t soa_offset, size_t soa_size,
            size_t elem_offset, size_t elem_size) noexcept
        : detail::SOAViewBase<Matrix<N>>{buffer, soa_offset, soa_size, elem_offset, elem_size},
          _cols{} {
        for (auto i = 0u; i < N; i++) {
            _cols[i] = SOAView<Column>{
                buffer,
                soa_offset + SOAView<Column>::compute_soa_size(soa_size) * i,
                soa_size, elem_offset, elem_size};
        }
    }

public:
    [[nodiscard]] auto operator[](size_t i) const noexcept { return _cols[i]; }
};

template<typename T, size_t N>
class SOAView<std::array<T, N>> : public detail::SOAViewBase<std::array<T, N>> {

private:
    SOAView<T> _elems[N];

public:
    [[nodiscard]] static auto compute_soa_size(auto n) noexcept {
        return SOAView<T>::compute_soa_size(n) * static_cast<uint>(N);
    }

public:
    SOAView() noexcept = default;
    SOAView(BufferView<uint> buffer,
            size_t soa_offset, size_t soa_size,
            size_t elem_offset, size_t elem_size) noexcept
        : detail::SOAViewBase<std::array<T, N>>{buffer, soa_offset, soa_size, elem_offset, elem_size},
          _elems{} {
        for (auto i = 0u; i < N; i++) {
            _elems[i] = SOAView<T>{
                buffer,
                soa_offset + SOAView<T>::compute_soa_size(soa_size) * i,
                soa_size, elem_offset, elem_size};
        }
    }

public:
    [[nodiscard]] auto operator[](size_t i) const noexcept { return _elems[i]; }
};

template<typename T, size_t N>
class SOAView<T[N]> : public SOAView<std::array<T, N>> {
    using Base = SOAView<std::array<T, N>>;
    using Base::Base;
};

template<typename T>
class SOA : public SOAView<T> {

private:
    Buffer<uint> _buffer;

private:
    SOA(Buffer<uint> buffer, size_t size) noexcept
        : SOAView<T>{buffer.view(), 0u, size, 0u, size},
          _buffer{std::move(buffer)} {}

public:
    SOA() noexcept = default;
    SOA(Device &device, size_t elem_count) noexcept
        : SOA{device.create_buffer<uint>(SOAView<T>::compute_soa_size(elem_count)), elem_count} {}
    [[nodiscard]] auto view() const noexcept { return SOAView<T>{*this}; }
};

template<typename T>
SOAView(const SOA<T> &) -> SOAView<T>;

template<typename T>
using SOAVar = Var<SOA<T>>;

}// namespace luisa::compute
