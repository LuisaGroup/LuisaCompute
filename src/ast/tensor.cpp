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
// Deserialization must therefore materialize LiteralExpr / StringIDExpr under a
// short-lived builder guard; the freshly allocated nodes are NOT registered in
// the builder's expression pool, so they stay owned by the tensor statement.
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

void write_i64_vector(luisa::vector<char> &buf, luisa::span<const int64_t> v) noexcept {
    write_u32(buf, static_cast<uint32_t>(v.size()));
    write_bytes(buf, v.data(), v.size() * sizeof(int64_t));
}

bool read_i64_vector(char const *&p, char const *end, luisa::vector<int64_t> &v) noexcept {
    uint32_t n;
    if (!read_u32(p, end, n)) [[unlikely]] { return false; }
    if (static_cast<size_t>(end - p) < static_cast<size_t>(n) * sizeof(int64_t)) [[unlikely]] { return false; }
    v.resize(n);
    return read_bytes(p, end, v.data(), n * sizeof(int64_t));
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

void write_sid(luisa::vector<char> &buf, const StringIDExpr *sid) noexcept {
    write_string(buf, sid == nullptr ? luisa::string_view{} : sid->data());
}

bool read_sid(char const *&p, char const *end, const StringIDExpr *&out) noexcept {
    luisa::string s;
    if (!read_string(p, end, s)) [[unlikely]] { return false; }
    out = with_builder([&] { return new StringIDExpr(std::move(s)); });
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
    }
    return "?";
}

// ---------------------------------------------------------------------------
// TensorExpr
// ---------------------------------------------------------------------------

TensorExpr::TensorExpr(int32_t rank,
                       TensorElementType dtype,
                       TensorScope scope,
                       luisa::vector<int64_t> dims,
                       luisa::vector<int64_t> offset,
                       luisa::vector<int64_t> extent,
                       const RefExpr *handle) noexcept
    : _rank{rank}, _dtype{dtype}, _scope{scope},
      _dims{std::move(dims)}, _offset{std::move(offset)},
      _extent{std::move(extent)}, _handle{handle} {
    if (_offset.empty() && !_dims.empty()) [[likely]] { _offset.assign(_dims.size(), 0); }
    if (_extent.empty()) [[likely]] { _extent = _dims; }// whole-tensor view
}

luisa::string TensorExpr::describe() const {
    auto join = [](luisa::span<const int64_t> v) noexcept {
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
    write_i64_vector(output_buffer, _dims);
    write_i64_vector(output_buffer, _offset);
    write_i64_vector(output_buffer, _extent);
    // _handle: RefExpr* — a pointer, non-serializable (R3 runtime variable).
    return output_buffer.size() - start;
}

bool TensorExpr::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!read_i32(input_ptr, end_ptr, _rank)) [[unlikely]] { return false; }
    uint32_t dtype;
    if (!read_u32(input_ptr, end_ptr, dtype)) [[unlikely]] { return false; }
    if (dtype > static_cast<uint32_t>(TensorElementType::I32)) [[unlikely]] { return false; }
    _dtype = static_cast<TensorElementType>(dtype);
    uint32_t scope;
    if (!read_u32(input_ptr, end_ptr, scope)) [[unlikely]] { return false; }
    if (scope > static_cast<uint32_t>(TensorScope::Fragment)) [[unlikely]] { return false; }
    _scope = static_cast<TensorScope>(scope);
    if (!read_i64_vector(input_ptr, end_ptr, _dims)) [[unlikely]] { return false; }
    if (!read_i64_vector(input_ptr, end_ptr, _offset)) [[unlikely]] { return false; }
    if (!read_i64_vector(input_ptr, end_ptr, _extent)) [[unlikely]] { return false; }
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
    return output_buffer.size() - start;
}
bool GemmStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    return read_i32(input_ptr, end_ptr, _trans_a) &&
           read_i32(input_ptr, end_ptr, _trans_b);
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
    write_literal(output_buffer, _dim);
    return output_buffer.size() - start;
}
bool ReduceSumStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    delete _dim;
    _dim = nullptr;
    return read_literal(input_ptr, end_ptr, _dim);
}

// --- TilePrint ---
size_t TilePrintStmt::serialize(luisa::vector<char> &output_buffer) {
    auto start = output_buffer.size();
    TensorStmt::serialize(output_buffer);
    write_sid(output_buffer, _msg);
    return output_buffer.size() - start;
}
bool TilePrintStmt::deserialize(char const *&input_ptr, char const *end_ptr) {
    if (!TensorStmt::deserialize(input_ptr, end_ptr)) [[unlikely]] { return false; }
    delete _msg;
    _msg = nullptr;
    return read_sid(input_ptr, end_ptr, _msg);
}

// --- Alloc ---
AllocStmt::AllocStmt(luisa::vector<int64_t> dims, TensorElementType dtype, TensorScope scope,
                     const RefExpr *handle) noexcept
    : TensorStmt{TileOpKind::ALLOC,
                 new TensorExpr{static_cast<int32_t>(dims.size()), dtype, scope,
                                std::move(dims), {}, {}, handle},
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
    delete _rhs_literal;
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
    delete _rhs_literal;
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
    delete _b;
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

}// namespace luisa::compute
