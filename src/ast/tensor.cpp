#include <luisa/ast/tensor.h>

#include <luisa/ast/function_builder.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>

#include <cstring>
#include <utility>

namespace luisa::compute {

namespace {

// Expression sub-classes can only be constructed while a FunctionBuilder is on
// the builder stack (Expression::Expression reads FunctionBuilder::current()).
// Deserialization must therefore materialize LiteralExpr under a short-lived
// builder guard; the freshly allocated nodes are NOT registered in the
// builder's expression pool, so they stay owned by the tensor statement.
template<typename F>
decltype(auto) with_builder(F &&f) noexcept {
    detail::FunctionBuilder builder;
    detail::FunctionBuilder::FunctionStackGuard guard{&builder};
    return std::forward<F>(f)();
}

}// namespace

namespace {

// ---------------------------------------------------------------------------
// Compact binary helpers.  All integers are written little-endian (raw host
// byte order); strings are length-prefixed (u32); vectors are count-prefixed.
// Every read is bounds-checked and returns false on malformed input.
// ---------------------------------------------------------------------------

void write_bytes(luisa::vector<char> &buf, const void *src, size_t n) noexcept {
    auto old = buf.size();
    buf.resize(old + n);
    if (n != 0u) [[likely]] { std::memcpy(buf.data() + old, src, n); }
}

bool read_bytes(char const *&p, char const *end, void *dst, size_t n) noexcept {
    if (p == nullptr || end < p) [[unlikely]] { return false; }
    if (static_cast<size_t>(end - p) < n) [[unlikely]] { return false; }
    if (n != 0u) [[likely]] { std::memcpy(dst, p, n); }
    p += n;
    return true;
}

template<typename T>
void write_scalar(luisa::vector<char> &buf, T v) noexcept {
    write_bytes(buf, &v, sizeof(T));
}

template<typename T>
bool read_scalar(char const *&p, char const *end, T &v) noexcept {
    return read_bytes(p, end, &v, sizeof(T));
}

void write_u8(luisa::vector<char> &buf, uint8_t v) noexcept { write_scalar(buf, v); }
void write_u16(luisa::vector<char> &buf, uint16_t v) noexcept { write_scalar(buf, v); }
void write_u32(luisa::vector<char> &buf, uint32_t v) noexcept { write_scalar(buf, v); }
void write_i32(luisa::vector<char> &buf, int32_t v) noexcept { write_scalar(buf, v); }
void write_i64(luisa::vector<char> &buf, int64_t v) noexcept { write_scalar(buf, v); }

bool read_u8(char const *&p, char const *end, uint8_t &v) noexcept { return read_scalar(p, end, v); }
bool read_u16(char const *&p, char const *end, uint16_t &v) noexcept { return read_scalar(p, end, v); }
bool read_u32(char const *&p, char const *end, uint32_t &v) noexcept { return read_scalar(p, end, v); }
bool read_i32(char const *&p, char const *end, int32_t &v) noexcept { return read_scalar(p, end, v); }
bool read_i64(char const *&p, char const *end, int64_t &v) noexcept { return read_scalar(p, end, v); }

void write_string(luisa::vector<char> &buf, luisa::string_view s) noexcept {
    write_u32(buf, static_cast<uint32_t>(s.size()));
    write_bytes(buf, s.data(), s.size());
}

bool read_string(char const *&p, char const *end, luisa::string &s) noexcept {
    uint32_t n;
    if (!read_u32(p, end, n)) [[unlikely]] { return false; }
    if (static_cast<size_t>(end - p) < static_cast<size_t>(n)) [[unlikely]] { return false; }
    s.assign(p, n);
    p += n;
    return true;
}

void write_i32_vector(luisa::vector<char> &buf, luisa::span<const int32_t> v) noexcept {
    write_u32(buf, static_cast<uint32_t>(v.size()));
    write_bytes(buf, v.data(), v.size() * sizeof(int32_t));
}

bool read_i32_vector(char const *&p, char const *end, luisa::fixed_vector<int32_t, 4> &v) noexcept {
    uint32_t n;
    if (!read_u32(p, end, n)) [[unlikely]] { return false; }
    if (static_cast<size_t>(end - p) < static_cast<size_t>(n) * sizeof(int32_t)) [[unlikely]] { return false; }
    v.resize(n);
    return read_bytes(p, end, v.data(), n * sizeof(int32_t));
}

// --- literal payload: type description + variant index + raw value bytes ----

void write_literal(luisa::vector<char> &buf, const LiteralExpr *lit) noexcept {
    write_string(buf, lit->type() == nullptr ? luisa::string_view{} : lit->type()->description());
    auto var = lit->value().to_variant();
    write_u16(buf, static_cast<uint16_t>(var.index()));
    luisa::visit([&](auto &&val) noexcept { write_bytes(buf, &val, sizeof(val)); }, var);
}

template<size_t I>
bool read_literal_alternative(detail::LiteralValue &v, char const *&p, char const *end) noexcept {
    using T = luisa::variant_alternative_t<I, detail::LiteralValueVariant>;
    if (static_cast<size_t>(end - p) < sizeof(T)) [[unlikely]] { return false; }
    T val;
    std::memcpy(&val, p, sizeof(T));
    p += sizeof(T);
    v = detail::LiteralValue{val};
    return true;
}

template<size_t... I>
bool read_literal_impl(detail::LiteralValue &v, size_t idx, char const *&p, char const *end,
                       std::index_sequence<I...>) noexcept {
    return ((idx == I ? read_literal_alternative<I>(v, p, end) : false) || ...);
}

bool read_literal(char const *&p, char const *end, const LiteralExpr *&out) noexcept {
    luisa::string desc;
    if (!read_string(p, end, desc)) [[unlikely]] { return false; }
    const Type *type = desc.empty() ? nullptr : Type::from(desc);
    uint16_t idx;
    if (!read_u16(p, end, idx)) [[unlikely]] { return false; }
    detail::LiteralValue value;
    constexpr auto n_alts = luisa::variant_size_v<detail::LiteralValueVariant>;
    if (idx >= n_alts) [[unlikely]] { return false; }
    if (!read_literal_impl(value, idx, p, end, std::make_index_sequence<n_alts>{})) [[unlikely]] { return false; }
    out = with_builder([&] { return new LiteralExpr(type, std::move(value)); });
    return true;
}

}// namespace

// ---------------------------------------------------------------------------
// TensorScope / TensorElementType helpers
// ---------------------------------------------------------------------------

const char *scope_name(TensorScope scope) noexcept {
    switch (scope) {
        case TensorScope::Global: return "global";
        case TensorScope::Shared: return "shared";
        case TensorScope::Fragment: return "fragment";
    }
    return "?";
}

const char *tensor_element_type_name(TensorElementType e) noexcept {
    switch (e) {
        case TensorElementType::F16: return "half";
        case TensorElementType::F32: return "float";
        case TensorElementType::I32: return "int";
        case TensorElementType::I8: return "int8";
        case TensorElementType::FP8: return "fp8";
        case TensorElementType::I4: return "int4";
        case TensorElementType::FP4: return "fp4";
    }
    return "?";
}

// ---------------------------------------------------------------------------
// TensorExpr
// ---------------------------------------------------------------------------

  TensorExpr::TensorExpr(int32_t rank,
                         TensorElementType dtype,
                         TensorScope scope,
                         luisa::fixed_vector<int32_t, 4> &&dims,
                         luisa::fixed_vector<int32_t, 4> &&offset,
                         luisa::fixed_vector<int32_t, 4> &&extent,
                         const RefExpr *handle,
                         luisa::string_view name) noexcept
        : _rank{rank}, _dtype{dtype}, _scope{scope},
          _dims{std::move(dims)}, _offset{std::move(offset)},
          _extent{std::move(extent)}, _handle{handle}, _name{name} {
    if (_offset.empty() && !_dims.empty()) [[likely]] { _offset.assign(_dims.size(), 0); }
    if (_extent.empty()) [[likely]] { _extent = _dims; }// whole-tensor view
}

luisa::string TensorExpr::describe() const {
    auto join = [](luisa::span<const int32_t> v) noexcept {
        luisa::string s;
        for (size_t i = 0u; i < v.size(); ++i) {
            if (i != 0u) { s += ","; }
            s += luisa::format("{}", v[i]);
        }
        return s;
    };
    return luisa::format("{}<{}>({})@({})",
                         scope_name(_scope),
                         tensor_element_type_name(_dtype),
                         join(_dims), join(_offset));
}

size_t TensorExpr::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    write_i32(output_buffer, _rank);
    write_u32(output_buffer, static_cast<uint32_t>(_dtype));
    write_u32(output_buffer, static_cast<uint32_t>(_scope));
    write_i32_vector(output_buffer, _dims);
    write_i32_vector(output_buffer, _offset);
    write_i32_vector(output_buffer, _extent);
    // _handle: RefExpr* — a pointer, non-serializable (R3 runtime variable).
    return output_buffer.size() - start;
}

bool TensorExpr::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!read_i32(input_ptr, end_ptr, _rank)) [[unlikely]] { return false; }
    uint32_t dtype;
    if (!read_u32(input_ptr, end_ptr, dtype)) [[unlikely]] { return false; }
    if (dtype > static_cast<uint32_t>(TensorElementType::FP4)) [[unlikely]] { return false; }
    _dtype = static_cast<TensorElementType>(dtype);
    uint32_t scope;
    if (!read_u32(input_ptr, end_ptr, scope)) [[unlikely]] { return false; }
    if (scope > static_cast<uint32_t>(TensorScope::Fragment)) [[unlikely]] { return false; }
    _scope = static_cast<TensorScope>(scope);
    if (!read_i32_vector(input_ptr, end_ptr, _dims)) [[unlikely]] { return false; }
    if (!read_i32_vector(input_ptr, end_ptr, _offset)) [[unlikely]] { return false; }
    if (!read_i32_vector(input_ptr, end_ptr, _extent)) [[unlikely]] { return false; }
    _handle = nullptr;// pointer, non-serializable
    return true;
}

// ---------------------------------------------------------------------------
// TensorStmt (base)
// ---------------------------------------------------------------------------

TensorStmt::TensorStmt(TileOpKind op) noexcept : _op{op} {}

TensorStmt::TensorStmt(TileOpKind op, TensorExpr *output, luisa::vector<TensorExpr *> inputs) noexcept
    : _op{op}, _output{output}, _inputs{std::move(inputs)} {}

TensorStmt::~TensorStmt() { _clear_owned(); }

void TensorStmt::_clear_owned() noexcept {
    delete _output;
    _output = nullptr;
    for (auto *t : _inputs) { delete t; }
    _inputs.clear();
}

void TensorStmt::set_annotation(luisa::string key, int64_t value) noexcept {
    for (auto &[k, v] : _annotations) {
        if (k == key) {
            v = value;
            return;
        }
    }
    _annotations.emplace_back(std::move(key), value);
}

const int64_t *TensorStmt::annotation(luisa::string_view key) const noexcept {
    for (auto &[k, v] : _annotations) {
        if (k == key) { return &v; }
    }
    return nullptr;
}

size_t TensorStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    write_u32(output_buffer, static_cast<uint32_t>(_op));
    // output tensor (may be absent)
    write_u8(output_buffer, _output == nullptr ? 0u : 1u);
    if (_output != nullptr) [[likely]] { _output->serialize(output_buffer); }
    // input argument tensors
    write_u32(output_buffer, static_cast<uint32_t>(_inputs.size()));
    for (auto *t : _inputs) { t->serialize(output_buffer); }
    // annotations (host-side meta)
    write_u32(output_buffer, static_cast<uint32_t>(_annotations.size()));
    for (auto &[k, v] : _annotations) {
        write_string(output_buffer, k);
        write_i64(output_buffer, v);
    }
    return output_buffer.size() - start;
}

bool TensorStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    uint32_t op;
    if (!read_u32(input_ptr, end_ptr, op)) [[unlikely]] { return false; }
    if (op != static_cast<uint32_t>(_op)) [[unlikely]] { return false; }// kind mismatch
    _clear_owned();
    // output tensor
    uint8_t has_output;
    if (!read_u8(input_ptr, end_ptr, has_output)) [[unlikely]] { return false; }
    if (has_output != 0u) [[likely]] {
        auto *out = new TensorExpr;
        if (!out->deserialize(input_ptr, end_ptr)) [[unlikely]] {
            delete out;
            return false;
        }
        _output = out;
    }
    // input argument tensors
    uint32_t n_inputs;
    if (!read_u32(input_ptr, end_ptr, n_inputs)) [[unlikely]] { return false; }
    // every tensor takes at least 24 bytes (6 x u32/i32), so a count that
    // exceeds the remaining buffer is malformed.
    if (static_cast<size_t>(end_ptr - input_ptr) < static_cast<size_t>(n_inputs) * 24u) [[unlikely]] { return false; }
    for (uint32_t i = 0u; i < n_inputs; ++i) {
        auto *in = new TensorExpr;
        if (!in->deserialize(input_ptr, end_ptr)) [[unlikely]] {
            delete in;
            return false;
        }
        _inputs.emplace_back(in);
    }
    // annotations
    uint32_t n_ann;
    if (!read_u32(input_ptr, end_ptr, n_ann)) [[unlikely]] { return false; }
    if (static_cast<size_t>(end_ptr - input_ptr) < static_cast<size_t>(n_ann) * 12u) [[unlikely]] { return false; }
    _annotations.clear();
    for (uint32_t i = 0u; i < n_ann; ++i) {
        luisa::string k;
        int64_t v;
        if (!read_string(input_ptr, end_ptr, k)) [[unlikely]] { return false; }
        if (!read_i64(input_ptr, end_ptr, v)) [[unlikely]] { return false; }
        _annotations.emplace_back(std::move(k), v);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Derived statements
// ---------------------------------------------------------------------------

// --- Gemm ---
size_t GemmStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_i32(output_buffer, _trans_a);
    write_i32(output_buffer, _trans_b);
    write_u32(output_buffer, static_cast<uint32_t>(_policy));
    write_i32(output_buffer, _clear_accum);
    write_i32(output_buffer, _k_pack);
    return output_buffer.size() - start;
}
bool GemmStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    uint32_t policy;
    if (!read_i32(input_ptr, end_ptr, _trans_a) ||
        !read_i32(input_ptr, end_ptr, _trans_b) ||
        !read_u32(input_ptr, end_ptr, policy)) [[unlikely]] { return false; }
    if (policy > static_cast<uint32_t>(GemmWarpPolicy::FullCol)) [[unlikely]] { return false; }
    _policy = static_cast<GemmWarpPolicy>(policy);
    return read_i32(input_ptr, end_ptr, _clear_accum) &&
           read_i32(input_ptr, end_ptr, _k_pack);
}

// --- Clear ---
size_t ClearStmt::serialize(luisa::vector<char> &output_buffer) {
    return TensorStmt::serialize(output_buffer);
}
bool ClearStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    return TensorStmt::deserialize(input_ptr, end_ptr);
}

// --- Copy ---
size_t CopyStmt::serialize(luisa::vector<char> &output_buffer) {
    return TensorStmt::serialize(output_buffer);
}
bool CopyStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    return TensorStmt::deserialize(input_ptr, end_ptr);
}

// --- ReduceSum ---
size_t ReduceSumStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_u32(output_buffer, _dim);
    return output_buffer.size() - start;
}
bool ReduceSumStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    return read_u32(input_ptr, end_ptr, _dim);
}

// --- TilePrint ---
size_t TilePrintStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_string(output_buffer, _msg);
    return output_buffer.size() - start;
}
bool TilePrintStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    return read_string(input_ptr, end_ptr, _msg);
}

// --- Alloc ---
AllocStmt::AllocStmt(luisa::fixed_vector<int32_t, 4> dims, TensorElementType dtype, TensorScope scope,
                       const RefExpr *handle, luisa::string_view name) noexcept
      : TensorStmt{TileOpKind::ALLOC,
                   new TensorExpr{static_cast<int32_t>(dims.size()), dtype, scope,
                                  std::move(dims), {}, {}, handle, name},
                   {}} {}
size_t AllocStmt::serialize(luisa::vector<char> &output_buffer) {
    return TensorStmt::serialize(output_buffer);
}
bool AllocStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    return TensorStmt::deserialize(input_ptr, end_ptr);
}

// --- TileStore ---
TileStoreStmt::TileStoreStmt(int32_t op, TensorExpr *lhs, TensorExpr *rhs_tensor,
                             const LiteralExpr *rhs_literal, const RefExpr *rhs_ref) noexcept
    : TensorStmt{TileOpKind::STORE,
                 lhs,
                 rhs_tensor != nullptr ? luisa::vector<TensorExpr *>{rhs_tensor}
                                       : luisa::vector<TensorExpr *>{}},
      _op{op}, _rhs_literal{rhs_literal}, _rhs_ref{rhs_ref} {}
size_t TileStoreStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_i32(output_buffer, _op);
    write_u8(output_buffer, _rhs_literal == nullptr ? 0u : 1u);
    if (_rhs_literal != nullptr) [[likely]] { write_literal(output_buffer, _rhs_literal); }
    return output_buffer.size() - start;
}
bool TileStoreStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    if (!read_i32(input_ptr, end_ptr, _op)) [[unlikely]] { return false; }
    uint8_t has_literal;
    if (!read_u8(input_ptr, end_ptr, has_literal)) [[unlikely]] { return false; }
    _rhs_literal = nullptr;
    if (has_literal != 0u) [[likely]] {
        if (!read_literal(input_ptr, end_ptr, _rhs_literal)) [[unlikely]] { return false; }
    }
    _rhs_ref = nullptr;// R3 runtime scalar, non-serializable
    return true;
}

// --- TileBinary ---
TileBinaryStmt::TileBinaryStmt(BinaryOp op, TensorExpr *lhs, TensorExpr *rhs_tensor,
                               const LiteralExpr *rhs_literal, const RefExpr *rhs_ref) noexcept
    : TensorStmt{TileOpKind::BINARY,
                 nullptr,
                 rhs_tensor != nullptr ? luisa::vector<TensorExpr *>{lhs, rhs_tensor}
                                       : luisa::vector<TensorExpr *>{lhs}},
      _op{op}, _rhs_literal{rhs_literal}, _rhs_ref{rhs_ref} {}
size_t TileBinaryStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_u32(output_buffer, static_cast<uint32_t>(_op));
    write_u8(output_buffer, _rhs_literal == nullptr ? 0u : 1u);
    if (_rhs_literal != nullptr) [[likely]] { write_literal(output_buffer, _rhs_literal); }
    return output_buffer.size() - start;
}
bool TileBinaryStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    uint32_t op;
    if (!read_u32(input_ptr, end_ptr, op)) [[unlikely]] { return false; }
    if (op > static_cast<uint32_t>(BinaryOp::NOT_EQUAL)) [[unlikely]] { return false; }
    _op = static_cast<BinaryOp>(op);
    uint8_t has_literal;
    if (!read_u8(input_ptr, end_ptr, has_literal)) [[unlikely]] { return false; }
    _rhs_literal = nullptr;
    if (has_literal != 0u) [[likely]] {
        if (!read_literal(input_ptr, end_ptr, _rhs_literal)) [[unlikely]] { return false; }
    }
    _rhs_ref = nullptr;// R3 runtime scalar, non-serializable
    return true;
}

// --- Max ---
size_t MaxStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_literal(output_buffer, _b);
    return output_buffer.size() - start;
}
bool MaxStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    _b = nullptr;
    return read_literal(input_ptr, end_ptr, _b);
}

// --- Rsqrt ---
size_t RsqrtStmt::serialize(luisa::vector<char> &output_buffer) {
    return TensorStmt::serialize(output_buffer);
}
bool RsqrtStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    return TensorStmt::deserialize(input_ptr, end_ptr);
}

// --- CeilDiv ---
size_t CeilDivStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_i32(output_buffer, _a);
    write_i32(output_buffer, _b);
    return output_buffer.size() - start;
}
bool CeilDivStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    return read_i32(input_ptr, end_ptr, _a) &&
           read_i32(input_ptr, end_ptr, _b);
}

// --- Kernel1D ---
size_t Kernel1DStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_i32(output_buffer, _gx);
    write_i32(output_buffer, _threads);
    return output_buffer.size() - start;
}
bool Kernel1DStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    return read_i32(input_ptr, end_ptr, _gx) &&
           read_i32(input_ptr, end_ptr, _threads);
}

// --- Kernel2D ---
size_t Kernel2DStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_i32(output_buffer, _gx);
    write_i32(output_buffer, _gy);
    write_i32(output_buffer, _threads);
    return output_buffer.size() - start;
}
bool Kernel2DStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    return read_i32(input_ptr, end_ptr, _gx) &&
           read_i32(input_ptr, end_ptr, _gy) &&
           read_i32(input_ptr, end_ptr, _threads);
}

// --- Pipelined ---
size_t PipelinedStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_i32(output_buffer, _count);
    write_i32(output_buffer, _stages);
    return output_buffer.size() - start;
}
bool PipelinedStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    return read_i32(input_ptr, end_ptr, _count) &&
           read_i32(input_ptr, end_ptr, _stages);
}

// ---------------------------------------------------------------------------
// Gap-analysis statements (section 5b of <luisa/ast/tensor.h>)
// ---------------------------------------------------------------------------

// --- Fill ---
size_t FillStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_u8(output_buffer, _value_literal == nullptr ? 0u : 1u);
    if (_value_literal != nullptr) [[likely]] { write_literal(output_buffer, _value_literal); }
    return output_buffer.size() - start;
}
bool FillStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    uint8_t has_literal;
    if (!read_u8(input_ptr, end_ptr, has_literal)) [[unlikely]] { return false; }
    _value_literal = nullptr;
    if (has_literal != 0u) [[likely]] {
        if (!read_literal(input_ptr, end_ptr, _value_literal)) [[unlikely]] { return false; }
    }
    _value_ref = nullptr;// R3 runtime scalar, non-serializable
    return true;
}

// --- Transpose ---
size_t TransposeStmt::serialize(luisa::vector<char> &output_buffer) {
    return TensorStmt::serialize(output_buffer);
}
bool TransposeStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    return TensorStmt::deserialize(input_ptr, end_ptr);
}

// --- Im2Col ---
size_t Im2ColStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_i32(output_buffer, _kernel);
    write_i32(output_buffer, _stride);
    write_i32(output_buffer, _dilation);
    write_i32(output_buffer, _pad);
    write_i32(output_buffer, _eviction_policy);
    return output_buffer.size() - start;
}
bool Im2ColStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    _nhw_step = nullptr;// R3 runtime step, non-serializable
    _c_step = nullptr;  // R3 runtime step, non-serializable
    return read_i32(input_ptr, end_ptr, _kernel) &&
           read_i32(input_ptr, end_ptr, _stride) &&
           read_i32(input_ptr, end_ptr, _dilation) &&
           read_i32(input_ptr, end_ptr, _pad) &&
           read_i32(input_ptr, end_ptr, _eviction_policy);
}

// --- AsyncCopy ---
size_t AsyncCopyStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_i32(output_buffer, _coalesced_width);
    return output_buffer.size() - start;
}
bool AsyncCopyStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    return read_i32(input_ptr, end_ptr, _coalesced_width);
}

// --- CopyCluster ---
size_t CopyClusterStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_i32(output_buffer, _dst_block);
    write_i32(output_buffer, _cluster_mask);
    write_i32(output_buffer, _coalesced_width);
    return output_buffer.size() - start;
}
bool CopyClusterStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    return read_i32(input_ptr, end_ptr, _dst_block) &&
           read_i32(input_ptr, end_ptr, _cluster_mask) &&
           read_i32(input_ptr, end_ptr, _coalesced_width);
}

// --- TmaCopy ---
size_t TmaCopyStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_i32(output_buffer, _leader_scope_threads);
    write_i32(output_buffer, _eviction_policy);
    return output_buffer.size() - start;
}
bool TmaCopyStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    return read_i32(input_ptr, end_ptr, _leader_scope_threads) &&
           read_i32(input_ptr, end_ptr, _eviction_policy);
}

// --- TmaGather4 ---
size_t TmaGather4Stmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_i32_vector(output_buffer, _rows);
    write_i32(output_buffer, _eviction_policy);
    return output_buffer.size() - start;
}
bool TmaGather4Stmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    _col = nullptr;// R3 runtime column, non-serializable
    if (!read_i32_vector(input_ptr, end_ptr, _rows)) [[unlikely]] { return false; }
    if (_rows.size() != 4u) [[unlikely]] { return false; }// gather4 requires exactly 4 rows
    return read_i32(input_ptr, end_ptr, _eviction_policy);
}

// --- TmaScatter4 ---
size_t TmaScatter4Stmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_i32_vector(output_buffer, _rows);
    write_i32(output_buffer, _eviction_policy);
    return output_buffer.size() - start;
}
bool TmaScatter4Stmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    _col = nullptr;// R3 runtime column, non-serializable
    if (!read_i32_vector(input_ptr, end_ptr, _rows)) [[unlikely]] { return false; }
    if (_rows.size() != 4u) [[unlikely]] { return false; }// scatter4 requires exactly 4 rows
    return read_i32(input_ptr, end_ptr, _eviction_policy);
}

// --- Reshape ---
size_t ReshapeStmt::serialize(luisa::vector<char> &output_buffer) {
    return TensorStmt::serialize(output_buffer);
}
bool ReshapeStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    return TensorStmt::deserialize(input_ptr, end_ptr);
}

// --- View ---
size_t ViewStmt::serialize(luisa::vector<char> &output_buffer) {
    return TensorStmt::serialize(output_buffer);
}
bool ViewStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    return TensorStmt::deserialize(input_ptr, end_ptr);
}

// --- Reduce ---
size_t ReduceStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_u32(output_buffer, static_cast<uint32_t>(_op));
    write_u32(output_buffer, _dim);
    write_i32(output_buffer, _clear);
    write_i32(output_buffer, _batch);
    write_i32(output_buffer, _nan_propagate);
    return output_buffer.size() - start;
}
bool ReduceStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    uint32_t op;
    if (!read_u32(input_ptr, end_ptr, op)) [[unlikely]] { return false; }
    if (op > static_cast<uint32_t>(TileReduceOp::BIT_XOR)) [[unlikely]] { return false; }
    _op = static_cast<TileReduceOp>(op);
    return read_u32(input_ptr, end_ptr, _dim) &&
           read_i32(input_ptr, end_ptr, _clear) &&
           read_i32(input_ptr, end_ptr, _batch) &&
           read_i32(input_ptr, end_ptr, _nan_propagate);
}

// --- FinalizeReducer ---
size_t FinalizeReducerStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_i32(output_buffer, _batch);
    return output_buffer.size() - start;
}
bool FinalizeReducerStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    return read_i32(input_ptr, end_ptr, _batch);
}

// --- WarpReduce ---
size_t WarpReduceStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_u32(output_buffer, static_cast<uint32_t>(_op));
    return output_buffer.size() - start;
}
bool WarpReduceStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    uint32_t op;
    if (!read_u32(input_ptr, end_ptr, op)) [[unlikely]] { return false; }
    if (op > static_cast<uint32_t>(TileWarpReduceOp::BIT_OR)) [[unlikely]] { return false; }
    _op = static_cast<TileWarpReduceOp>(op);
    return true;
}

// --- CumSum ---
size_t CumSumStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_u32(output_buffer, _dim);
    write_i32(output_buffer, _reverse);
    return output_buffer.size() - start;
}
bool CumSumStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    return read_u32(input_ptr, end_ptr, _dim) &&
           read_i32(input_ptr, end_ptr, _reverse);
}

// --- CumMax ---
size_t CumMaxStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_u32(output_buffer, _dim);
    write_i32(output_buffer, _reverse);
    return output_buffer.size() - start;
}
bool CumMaxStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    return read_u32(input_ptr, end_ptr, _dim) &&
           read_i32(input_ptr, end_ptr, _reverse);
}

// --- WgmmaGemm ---
size_t WgmmaGemmStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_i32(output_buffer, _trans_a);
    write_i32(output_buffer, _trans_b);
    write_u32(output_buffer, static_cast<uint32_t>(_policy));
    write_i32(output_buffer, _clear_accum);
    return output_buffer.size() - start;
}
bool WgmmaGemmStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    uint32_t policy;
    if (!read_i32(input_ptr, end_ptr, _trans_a) ||
        !read_i32(input_ptr, end_ptr, _trans_b) ||
        !read_u32(input_ptr, end_ptr, policy)) [[unlikely]] { return false; }
    if (policy > static_cast<uint32_t>(GemmWarpPolicy::FullCol)) [[unlikely]] { return false; }
    _policy = static_cast<GemmWarpPolicy>(policy);
    return read_i32(input_ptr, end_ptr, _clear_accum);
}

// --- Tcgen05Gemm ---
size_t Tcgen05GemmStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_i32(output_buffer, _trans_a);
    write_i32(output_buffer, _trans_b);
    write_u32(output_buffer, static_cast<uint32_t>(_policy));
    write_i32(output_buffer, _clear_accum);
    return output_buffer.size() - start;
}
bool Tcgen05GemmStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    uint32_t policy;
    if (!read_i32(input_ptr, end_ptr, _trans_a) ||
        !read_i32(input_ptr, end_ptr, _trans_b) ||
        !read_u32(input_ptr, end_ptr, policy)) [[unlikely]] { return false; }
    if (policy > static_cast<uint32_t>(GemmWarpPolicy::FullCol)) [[unlikely]] { return false; }
    _policy = static_cast<GemmWarpPolicy>(policy);
    return read_i32(input_ptr, end_ptr, _clear_accum);
}

// --- Tcgen05GemmBlockscaled ---
size_t Tcgen05GemmBlockscaledStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_i32(output_buffer, _trans_a);
    write_i32(output_buffer, _trans_b);
    write_i32(output_buffer, _clear_accum);
    write_i32(output_buffer, _k_start);
    write_i32(output_buffer, _sf_a_granularity_k);
    write_i32(output_buffer, _sf_b_granularity_k);
    return output_buffer.size() - start;
}
bool Tcgen05GemmBlockscaledStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    return read_i32(input_ptr, end_ptr, _trans_a) &&
           read_i32(input_ptr, end_ptr, _trans_b) &&
           read_i32(input_ptr, end_ptr, _clear_accum) &&
           read_i32(input_ptr, end_ptr, _k_start) &&
           read_i32(input_ptr, end_ptr, _sf_a_granularity_k) &&
           read_i32(input_ptr, end_ptr, _sf_b_granularity_k);
}

// --- GemmSp / WgmmaGemmSp / Tcgen05GemmSp ---
template<typename SpStmt>
size_t serialize_gemm_sp(SpStmt *stmt, luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    stmt->TensorStmt::serialize(output_buffer);
    write_i32(output_buffer, stmt->trans_a());
    write_i32(output_buffer, stmt->trans_e());
    write_i32(output_buffer, stmt->trans_b());
    write_u32(output_buffer, static_cast<uint32_t>(stmt->policy()));
    write_i32(output_buffer, stmt->clear_accum());
    return output_buffer.size() - start;
}

size_t GemmSpStmt::serialize(luisa::vector<char> &output_buffer) {
    return serialize_gemm_sp(this, output_buffer);
}
bool GemmSpStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    int32_t trans_a, trans_e, trans_b;
    uint32_t policy;
    if (!read_i32(input_ptr, end_ptr, trans_a) ||
        !read_i32(input_ptr, end_ptr, trans_e) ||
        !read_i32(input_ptr, end_ptr, trans_b) ||
        !read_u32(input_ptr, end_ptr, policy)) [[unlikely]] { return false; }
    if (policy > static_cast<uint32_t>(GemmWarpPolicy::FullCol)) [[unlikely]] { return false; }
    _trans_a = trans_a;
    _trans_e = trans_e;
    _trans_b = trans_b;
    _policy = static_cast<GemmWarpPolicy>(policy);
    return read_i32(input_ptr, end_ptr, _clear_accum);
}

size_t WgmmaGemmSpStmt::serialize(luisa::vector<char> &output_buffer) {
    return serialize_gemm_sp(this, output_buffer);
}
bool WgmmaGemmSpStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    int32_t trans_a, trans_e, trans_b;
    uint32_t policy;
    if (!read_i32(input_ptr, end_ptr, trans_a) ||
        !read_i32(input_ptr, end_ptr, trans_e) ||
        !read_i32(input_ptr, end_ptr, trans_b) ||
        !read_u32(input_ptr, end_ptr, policy)) [[unlikely]] { return false; }
    if (policy > static_cast<uint32_t>(GemmWarpPolicy::FullCol)) [[unlikely]] { return false; }
    _trans_a = trans_a;
    _trans_e = trans_e;
    _trans_b = trans_b;
    _policy = static_cast<GemmWarpPolicy>(policy);
    return read_i32(input_ptr, end_ptr, _clear_accum);
}

size_t Tcgen05GemmSpStmt::serialize(luisa::vector<char> &output_buffer) {
    return serialize_gemm_sp(this, output_buffer);
}
bool Tcgen05GemmSpStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    int32_t trans_a, trans_e, trans_b;
    uint32_t policy;
    if (!read_i32(input_ptr, end_ptr, trans_a) ||
        !read_i32(input_ptr, end_ptr, trans_e) ||
        !read_i32(input_ptr, end_ptr, trans_b) ||
        !read_u32(input_ptr, end_ptr, policy)) [[unlikely]] { return false; }
    if (policy > static_cast<uint32_t>(GemmWarpPolicy::FullCol)) [[unlikely]] { return false; }
    _trans_a = trans_a;
    _trans_e = trans_e;
    _trans_b = trans_b;
    _policy = static_cast<GemmWarpPolicy>(policy);
    return read_i32(input_ptr, end_ptr, _clear_accum);
}

// --- Atomic ---
size_t AtomicStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_u32(output_buffer, static_cast<uint32_t>(_op));
    write_u32(output_buffer, static_cast<uint32_t>(_memory_order));
    write_i32(output_buffer, _return_prev);
    write_i32(output_buffer, _use_tma);
    write_u8(output_buffer, _value_literal == nullptr ? 0u : 1u);
    if (_value_literal != nullptr) [[likely]] { write_literal(output_buffer, _value_literal); }
    return output_buffer.size() - start;
}
bool AtomicStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    uint32_t op, memory_order;
    if (!read_u32(input_ptr, end_ptr, op) ||
        !read_u32(input_ptr, end_ptr, memory_order)) [[unlikely]] { return false; }
    if (op > static_cast<uint32_t>(TileAtomicOp::STORE)) [[unlikely]] { return false; }
    if (memory_order > static_cast<uint32_t>(TileMemoryOrder::SEQ_CST)) [[unlikely]] { return false; }
    _op = static_cast<TileAtomicOp>(op);
    _memory_order = static_cast<TileMemoryOrder>(memory_order);
    if (!read_i32(input_ptr, end_ptr, _return_prev) ||
        !read_i32(input_ptr, end_ptr, _use_tma)) [[unlikely]] { return false; }
    uint8_t has_literal;
    if (!read_u8(input_ptr, end_ptr, has_literal)) [[unlikely]] { return false; }
    _value_literal = nullptr;
    if (has_literal != 0u) [[likely]] {
        if (!read_literal(input_ptr, end_ptr, _value_literal)) [[unlikely]] { return false; }
    }
    _value_ref = nullptr;// R3 runtime scalar, non-serializable
    return true;
}

// --- Clamp ---
size_t ClampStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_u8(output_buffer, _lo_literal == nullptr ? 0u : 1u);
    if (_lo_literal != nullptr) [[likely]] { write_literal(output_buffer, _lo_literal); }
    write_u8(output_buffer, _hi_literal == nullptr ? 0u : 1u);
    if (_hi_literal != nullptr) [[likely]] { write_literal(output_buffer, _hi_literal); }
    return output_buffer.size() - start;
}
bool ClampStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    uint8_t has_lo, has_hi;
    if (!read_u8(input_ptr, end_ptr, has_lo)) [[unlikely]] { return false; }
    _lo_literal = nullptr;
    if (has_lo != 0u) [[likely]] {
        if (!read_literal(input_ptr, end_ptr, _lo_literal)) [[unlikely]] { return false; }
    }
    if (!read_u8(input_ptr, end_ptr, has_hi)) [[unlikely]] { return false; }
    _hi_literal = nullptr;
    if (has_hi != 0u) [[likely]] {
        if (!read_literal(input_ptr, end_ptr, _hi_literal)) [[unlikely]] { return false; }
    }
    _lo_ref = nullptr;// R3 runtime scalars, non-serializable
    _hi_ref = nullptr;
    return true;
}

// --- Dp4a ---
size_t Dp4aStmt::serialize(luisa::vector<char> &output_buffer) {
    return TensorStmt::serialize(output_buffer);
}
bool Dp4aStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    return TensorStmt::deserialize(input_ptr, end_ptr);
}

// --- LoopBreak ---
size_t LoopBreakStmt::serialize(luisa::vector<char> &output_buffer) {
    return TensorStmt::serialize(output_buffer);
}
bool LoopBreakStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    return TensorStmt::deserialize(input_ptr, end_ptr);
}

// --- AnyOf ---
size_t AnyOfStmt::serialize(luisa::vector<char> &output_buffer) {
    return TensorStmt::serialize(output_buffer);
}
bool AnyOfStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    return TensorStmt::deserialize(input_ptr, end_ptr);
}

// --- AllOf ---
size_t AllOfStmt::serialize(luisa::vector<char> &output_buffer) {
    return TensorStmt::serialize(output_buffer);
}
bool AllOfStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    return TensorStmt::deserialize(input_ptr, end_ptr);
}

// --- Sync ---
size_t SyncStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_u32(output_buffer, static_cast<uint32_t>(_op));
    write_i32(output_buffer, _mask);
    write_i32(output_buffer, _barrier_id);
    write_i32(output_buffer, _arrive_count);
    return output_buffer.size() - start;
}
bool SyncStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    uint32_t op;
    if (!read_u32(input_ptr, end_ptr, op)) [[unlikely]] { return false; }
    if (op > static_cast<uint32_t>(TileSyncOp::GLOBAL)) [[unlikely]] { return false; }
    _op = static_cast<TileSyncOp>(op);
    return read_i32(input_ptr, end_ptr, _mask) &&
           read_i32(input_ptr, end_ptr, _barrier_id) &&
           read_i32(input_ptr, end_ptr, _arrive_count);
}

// --- Barrier ---
size_t BarrierStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_u32(output_buffer, static_cast<uint32_t>(_op));
    write_i32(output_buffer, _parity);
    write_i32(output_buffer, _barrier_id);
    write_i32(output_buffer, _thread_count);
    return output_buffer.size() - start;
}
bool BarrierStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    uint32_t op;
    if (!read_u32(input_ptr, end_ptr, op)) [[unlikely]] { return false; }
    if (op > static_cast<uint32_t>(TileBarrierOp::NAMED_ARRIVE)) [[unlikely]] { return false; }
    _op = static_cast<TileBarrierOp>(op);
    return read_i32(input_ptr, end_ptr, _parity) &&
           read_i32(input_ptr, end_ptr, _barrier_id) &&
           read_i32(input_ptr, end_ptr, _thread_count);
}

// --- MBarrier ---
size_t MBarrierStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_u32(output_buffer, static_cast<uint32_t>(_op));
    write_i32(output_buffer, _tx);
    write_i32(output_buffer, _parity);
    write_i32(output_buffer, _cta_id);
    return output_buffer.size() - start;
}
bool MBarrierStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    uint32_t op;
    if (!read_u32(input_ptr, end_ptr, op)) [[unlikely]] { return false; }
    if (op > static_cast<uint32_t>(TileMBarrierOp::WAIT_PARITY)) [[unlikely]] { return false; }
    _op = static_cast<TileMBarrierOp>(op);
    return read_i32(input_ptr, end_ptr, _tx) &&
           read_i32(input_ptr, end_ptr, _parity) &&
           read_i32(input_ptr, end_ptr, _cta_id);
}

// --- WarpVote ---
size_t WarpVoteStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_u32(output_buffer, static_cast<uint32_t>(_op));
    write_u8(output_buffer, _pred_literal == nullptr ? 0u : 1u);
    if (_pred_literal != nullptr) [[likely]] { write_literal(output_buffer, _pred_literal); }
    return output_buffer.size() - start;
}
bool WarpVoteStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    uint32_t op;
    if (!read_u32(input_ptr, end_ptr, op)) [[unlikely]] { return false; }
    if (op > static_cast<uint32_t>(TileWarpVoteOp::MATCH_ALL_SYNC)) [[unlikely]] { return false; }
    _op = static_cast<TileWarpVoteOp>(op);
    uint8_t has_literal;
    if (!read_u8(input_ptr, end_ptr, has_literal)) [[unlikely]] { return false; }
    _pred_literal = nullptr;
    if (has_literal != 0u) [[likely]] {
        if (!read_literal(input_ptr, end_ptr, _pred_literal)) [[unlikely]] { return false; }
    }
    _mask = nullptr;    // R3 warp mask, non-serializable
    _pred_ref = nullptr;// R3 runtime predicate, non-serializable
    return true;
}

// --- Shuffle ---
size_t ShuffleStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_u32(output_buffer, static_cast<uint32_t>(_op));
    write_i32(output_buffer, _width);
    write_i32(output_buffer, _delta);
    write_i32(output_buffer, _thread_extent);
    write_u8(output_buffer, _value_literal == nullptr ? 0u : 1u);
    if (_value_literal != nullptr) [[likely]] { write_literal(output_buffer, _value_literal); }
    return output_buffer.size() - start;
}
bool ShuffleStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    uint32_t op;
    if (!read_u32(input_ptr, end_ptr, op)) [[unlikely]] { return false; }
    if (op > static_cast<uint32_t>(TileShuffleOp::ELECT)) [[unlikely]] { return false; }
    _op = static_cast<TileShuffleOp>(op);
    if (!read_i32(input_ptr, end_ptr, _width) ||
        !read_i32(input_ptr, end_ptr, _delta) ||
        !read_i32(input_ptr, end_ptr, _thread_extent)) [[unlikely]] { return false; }
    uint8_t has_literal;
    if (!read_u8(input_ptr, end_ptr, has_literal)) [[unlikely]] { return false; }
    _value_literal = nullptr;
    if (has_literal != 0u) [[likely]] {
        if (!read_literal(input_ptr, end_ptr, _value_literal)) [[unlikely]] { return false; }
    }
    _mask = nullptr;    // R3 warp mask, non-serializable
    _src_lane = nullptr;// R3 source lane, non-serializable
    _value_ref = nullptr;// R3 runtime value, non-serializable
    return true;
}

// --- SyncThreadsVote ---
size_t SyncThreadsVoteStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_u32(output_buffer, static_cast<uint32_t>(_op));
    write_u8(output_buffer, _pred_literal == nullptr ? 0u : 1u);
    if (_pred_literal != nullptr) [[likely]] { write_literal(output_buffer, _pred_literal); }
    return output_buffer.size() - start;
}
bool SyncThreadsVoteStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    uint32_t op;
    if (!read_u32(input_ptr, end_ptr, op)) [[unlikely]] { return false; }
    if (op > static_cast<uint32_t>(TileSyncThreadsVoteOp::OR)) [[unlikely]] { return false; }
    _op = static_cast<TileSyncThreadsVoteOp>(op);
    uint8_t has_literal;
    if (!read_u8(input_ptr, end_ptr, has_literal)) [[unlikely]] { return false; }
    _pred_literal = nullptr;
    if (has_literal != 0u) [[likely]] {
        if (!read_literal(input_ptr, end_ptr, _pred_literal)) [[unlikely]] { return false; }
    }
    _pred_ref = nullptr;// R3 runtime predicate, non-serializable
    return true;
}

// --- FastRcp ---
size_t FastRcpStmt::serialize(luisa::vector<char> &output_buffer) {
    return TensorStmt::serialize(output_buffer);
}
bool FastRcpStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    return TensorStmt::deserialize(input_ptr, end_ptr);
}

// --- IeeeMath ---
size_t IeeeMathStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_u32(output_buffer, static_cast<uint32_t>(_op));
    write_i32(output_buffer, _rounding_mode);
    return output_buffer.size() - start;
}
bool IeeeMathStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    uint32_t op;
    if (!read_u32(input_ptr, end_ptr, op)) [[unlikely]] { return false; }
    if (op > static_cast<uint32_t>(TileIeeeOp::FDIV)) [[unlikely]] { return false; }
    _op = static_cast<TileIeeeOp>(op);
    return read_i32(input_ptr, end_ptr, _rounding_mode);
}

// --- PackedMath ---
size_t PackedMathStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_u32(output_buffer, static_cast<uint32_t>(_op));
    return output_buffer.size() - start;
}
bool PackedMathStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    uint32_t op;
    if (!read_u32(input_ptr, end_ptr, op)) [[unlikely]] { return false; }
    if (op > static_cast<uint32_t>(TilePackedOp::ABS2)) [[unlikely]] { return false; }
    _op = static_cast<TilePackedOp>(op);
    return true;
}

// --- FastMath ---
size_t FastMathStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_u32(output_buffer, static_cast<uint32_t>(_op));
    return output_buffer.size() - start;
}
bool FastMathStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    uint32_t op;
    if (!read_u32(input_ptr, end_ptr, op)) [[unlikely]] { return false; }
    if (op > static_cast<uint32_t>(TileFastMathOp::TAN)) [[unlikely]] { return false; }
    _op = static_cast<TileFastMathOp>(op);
    return true;
}

// --- AllocSpecial ---
size_t AllocSpecialStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_u32(output_buffer, static_cast<uint32_t>(_kind));
    write_i32(output_buffer, _count);
    write_i32(output_buffer, _reducer_op);
    write_i32(output_buffer, _desc_kind);
    write_u8(output_buffer, _init == nullptr ? 0u : 1u);
    if (_init != nullptr) [[likely]] { write_literal(output_buffer, _init); }
    return output_buffer.size() - start;
}
bool AllocSpecialStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    uint32_t kind;
    if (!read_u32(input_ptr, end_ptr, kind)) [[unlikely]] { return false; }
    if (kind > static_cast<uint32_t>(TileAllocKind::CLUSTER_BARRIER)) [[unlikely]] { return false; }
    _kind = static_cast<TileAllocKind>(kind);
    if (!read_i32(input_ptr, end_ptr, _count) ||
        !read_i32(input_ptr, end_ptr, _reducer_op) ||
        !read_i32(input_ptr, end_ptr, _desc_kind)) [[unlikely]] { return false; }
    uint8_t has_init;
    if (!read_u8(input_ptr, end_ptr, has_init)) [[unlikely]] { return false; }
    _init = nullptr;
    if (has_init != 0u) [[likely]] {
        if (!read_literal(input_ptr, end_ptr, _init)) [[unlikely]] { return false; }
    }
    return true;
}

// --- LoopAnnotation ---
size_t LoopAnnotationStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_u32(output_buffer, static_cast<uint32_t>(_kind));
    write_i32(output_buffer, _extent);
    write_i32(output_buffer, _coalesced_width);
    return output_buffer.size() - start;
}
bool LoopAnnotationStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    uint32_t kind;
    if (!read_u32(input_ptr, end_ptr, kind)) [[unlikely]] { return false; }
    if (kind > static_cast<uint32_t>(TileLoopAnnotKind::VECTORIZED)) [[unlikely]] { return false; }
    _kind = static_cast<TileLoopAnnotKind>(kind);
    return read_i32(input_ptr, end_ptr, _extent) &&
           read_i32(input_ptr, end_ptr, _coalesced_width);
}

// --- Annotate ---
size_t AnnotateStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_u32(output_buffer, static_cast<uint32_t>(_kind));
    write_i32(output_buffer, _panel_size);
    write_i32(output_buffer, _order);
    write_i32(output_buffer, _enable);
    write_i32(output_buffer, _value);
    write_bytes(output_buffer, &_hit_ratio, sizeof(float));
    write_u8(output_buffer, _safe_value == nullptr ? 0u : 1u);
    if (_safe_value != nullptr) [[likely]] { write_literal(output_buffer, _safe_value); }
    return output_buffer.size() - start;
}
bool AnnotateStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    uint32_t kind;
    if (!read_u32(input_ptr, end_ptr, kind)) [[unlikely]] { return false; }
    if (kind > static_cast<uint32_t>(TileAnnotKind::MIN_BLOCKS_PER_SM)) [[unlikely]] { return false; }
    _kind = static_cast<TileAnnotKind>(kind);
    if (!read_i32(input_ptr, end_ptr, _panel_size) ||
        !read_i32(input_ptr, end_ptr, _order) ||
        !read_i32(input_ptr, end_ptr, _enable) ||
        !read_i32(input_ptr, end_ptr, _value)) [[unlikely]] { return false; }
    if (!read_bytes(input_ptr, end_ptr, &_hit_ratio, sizeof(float))) [[unlikely]] { return false; }
    uint8_t has_safe;
    if (!read_u8(input_ptr, end_ptr, has_safe)) [[unlikely]] { return false; }
    _safe_value = nullptr;
    if (has_safe != 0u) [[likely]] {
        if (!read_literal(input_ptr, end_ptr, _safe_value)) [[unlikely]] { return false; }
    }
    return true;
}

// --- Dynamic ---
size_t DynamicStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_string(output_buffer, _name);
    write_u32(output_buffer, static_cast<uint32_t>(_dtype));
    return output_buffer.size() - start;
}
bool DynamicStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    if (!read_string(input_ptr, end_ptr, _name)) [[unlikely]] { return false; }
    uint32_t dtype;
    if (!read_u32(input_ptr, end_ptr, dtype)) [[unlikely]] { return false; }
    if (dtype > static_cast<uint32_t>(TensorElementType::FP4)) [[unlikely]] { return false; }
    _dtype = static_cast<TensorElementType>(dtype);
    return true;
}

// --- Symbolic ---
size_t SymbolicStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_string(output_buffer, _name);
    write_u32(output_buffer, static_cast<uint32_t>(_dtype));
    return output_buffer.size() - start;
}
bool SymbolicStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    if (!read_string(input_ptr, end_ptr, _name)) [[unlikely]] { return false; }
    uint32_t dtype;
    if (!read_u32(input_ptr, end_ptr, dtype)) [[unlikely]] { return false; }
    if (dtype > static_cast<uint32_t>(TensorElementType::FP4)) [[unlikely]] { return false; }
    _dtype = static_cast<TensorElementType>(dtype);
    return true;
}

// --- Inline ---
size_t InlineStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_string(output_buffer, _message);
    return output_buffer.size() - start;
}
bool InlineStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    return read_string(input_ptr, end_ptr, _message);
}

// --- MetaClass ---
size_t MetaClassStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_string(output_buffer, _message);
    return output_buffer.size() - start;
}
bool MetaClassStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    return read_string(input_ptr, end_ptr, _message);
}

// --- AccessPtr ---
size_t AccessPtrStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_i32(output_buffer, _access_type);
    write_i32(output_buffer, _offset);
    write_i32(output_buffer, _extent);
    write_i32(output_buffer, _ignore_last_ndim);
    return output_buffer.size() - start;
}
bool AccessPtrStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    return read_i32(input_ptr, end_ptr, _access_type) &&
           read_i32(input_ptr, end_ptr, _offset) &&
           read_i32(input_ptr, end_ptr, _extent) &&
           read_i32(input_ptr, end_ptr, _ignore_last_ndim);
}

// --- IndexToCoordinates ---
size_t IndexToCoordinatesStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_u8(output_buffer, _index_literal == nullptr ? 0u : 1u);
    if (_index_literal != nullptr) [[likely]] { write_literal(output_buffer, _index_literal); }
    write_i32_vector(output_buffer, _shape);
    return output_buffer.size() - start;
}
bool IndexToCoordinatesStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    uint8_t has_literal;
    if (!read_u8(input_ptr, end_ptr, has_literal)) [[unlikely]] { return false; }
    _index_literal = nullptr;
    if (has_literal != 0u) [[likely]] {
        if (!read_literal(input_ptr, end_ptr, _index_literal)) [[unlikely]] { return false; }
    }
    _index_ref = nullptr;// R3 runtime index, non-serializable
    return read_i32_vector(input_ptr, end_ptr, _shape);
}

}// namespace luisa::compute
