#include <atomic>

template<class T>
inline T lc_cpu_custom_op(const KernelFnArgs* k_args, size_t i, T value) {
    if (i >= k_args->custom_ops_count) {
        fprintf(stderr, "Custom op out of bounds: %zu >= %zu\n", i, k_args->custom_ops_count);
        print_backtrace_hint();
        abort();
    }
    auto op = k_args->custom_ops[i];
    op.func(op.data, reinterpret_cast<uint8_t*>(&value));
    return value;
}
template<class T>
inline size_t lc_buffer_size(const BufferView &buffer) noexcept {
    return buffer.size / sizeof(T);
}
template<class T>
inline T lc_buffer_read(const BufferView &buffer, size_t i) noexcept {
    if (i >= lc_buffer_size<T>(buffer)) {
        fprintf(stderr, "Buffer read out of bounds: %zu >= %zu\n", i, lc_buffer_size<T>(buffer));
        print_backtrace_hint();
        abort();
    }
    return *(reinterpret_cast<const T *>(buffer.data) + i);
}
template<class T>
inline T *lc_buffer_ref(const BufferView &buffer, size_t i) noexcept {
    if (i >= lc_buffer_size<T>(buffer)) {
        fprintf(stderr, "Buffer ref out of bounds: %zu >= %zu\n", i, lc_buffer_size<T>(buffer));
        print_backtrace_hint();
        abort();
    }
    return (reinterpret_cast<T *>(buffer.data) + i);
}
template<class T>
inline void lc_buffer_write(const BufferView &buffer, size_t i, T value) noexcept {
    if (i >= lc_buffer_size<T>(buffer)) {
        fprintf(stderr, "Buffer read out of bounds: %zu >= %zu\n", i, lc_buffer_size<T>(buffer));
        print_backtrace_hint();
        abort();
    }
    *(reinterpret_cast<T *>(buffer.data) + i) = value;
}

inline BufferView lc_buffer_arg(const KernelFnArgs *k_args, size_t i) noexcept {
    if (i >= k_args->args_count) {
        fprintf(stderr, "Buffer argument out of bounds: %zu >= %zu\n", i, k_args->args_count);
        print_backtrace_hint();
        abort();
    }
    auto arg = k_args->args[i];
    if (arg.tag != KernelFnArg::Tag::Buffer) {
        fprintf(stderr, "Buffer argument type mismatch: %d != %d\n", arg.tag, KernelFnArg::Tag::Buffer);
        print_backtrace_hint();
        abort();
    }
    return arg.buffer._0;
}
inline BufferView lc_buffer_capture(const KernelFnArgs *k_args, size_t i) noexcept {
    if (i >= k_args->captured_count) {
        fprintf(stderr, "Buffer argument out of bounds: %zu >= %zu\n", i, k_args->captured_count);
        print_backtrace_hint();
        abort();
    }
    auto arg = k_args->captured[i];
    if (arg.tag != KernelFnArg::Tag::Buffer) {
        fprintf(stderr, "Buffer argument type mismatch: %d != %d\n", arg.tag, KernelFnArg::Tag::Buffer);
        print_backtrace_hint();
        abort();
    }
    return arg.buffer._0;
}
inline BindlessArray lc_bindless_arg(const KernelFnArgs *k_args, size_t i) noexcept {
    if (i >= k_args->args_count) {
        fprintf(stderr, "Bindless argument out of bounds: %zu >= %zu\n", i, k_args->args_count);
        print_backtrace_hint();
        abort();
    }
    auto arg = k_args->args[i];
    if (arg.tag != KernelFnArg::Tag::BindlessArray) {
        fprintf(stderr, "Bindless argument type mismatch: %d != %d\n", arg.tag, KernelFnArg::Tag::BindlessArray);
        print_backtrace_hint();
        abort();
    }
    return arg.bindless_array._0;
}
inline BindlessArray lc_bindless_capture(const KernelFnArgs *k_args, size_t i) noexcept {
    if (i >= k_args->captured_count) {
        fprintf(stderr, "Bindless argument out of bounds: %zu >= %zu\n", i, k_args->captured_count);
        print_backtrace_hint();
        abort();
    }
    auto arg = k_args->captured[i];
    if (arg.tag != KernelFnArg::Tag::BindlessArray) {
        fprintf(stderr, "Bindless argument type mismatch: %d != %d\n", arg.tag, KernelFnArg::Tag::BindlessArray);
        print_backtrace_hint();
        abort();
    }
    return arg.bindless_array._0;
}
BufferView lc_bindless_buffer(const BindlessArray&array, size_t buf_index) noexcept{
    if (buf_index >= array.buffers_count) {
        fprintf(stderr, "Bindless buffer index out of bounds: %zu >= %zu\n", buf_index, array.buffers_count);
        print_backtrace_hint();
        abort();
    }
    return array.buffers[buf_index];
}

inline uint64_t lc_bindless_buffer_type(const BindlessArray&array, size_t buf_index) noexcept {
    auto buf = lc_bindless_buffer(array, buf_index);
    return buf.ty;
}
template<class T>
inline T lc_bindless_buffer_read(const BindlessArray&array, size_t buf_index, size_t element) noexcept {
    auto buf = lc_bindless_buffer(array, buf_index);
    return lc_buffer_read<T>(buf, element);
}
template<class T>
inline T lc_bindless_buffer_size(const BindlessArray&array, size_t buf_index) noexcept {
    auto buf = lc_bindless_buffer(array, buf_index);
    return lc_buffer_size<T>(buf);
}
template<class T>
inline T lc_atomic_compare_exchange(T *ptr, T expected, T desired) noexcept {
    auto old = expected;
    reinterpret_cast<std::atomic<T> *>(ptr)->compare_exchange_weak(old, desired);
    return old;
}
template<class T>
inline T lc_atomic_exchange(T *ptr, T desired) noexcept {
    return reinterpret_cast<std::atomic<T> *>(ptr)->exchange(desired);
}
template<class T>
inline T lc_atomic_fetch_add(T *ptr, T value) noexcept {
    return reinterpret_cast<std::atomic<T> *>(ptr)->fetch_add(value);
}
template<class T>
inline T lc_atomic_fetch_sub(T *ptr, T value) noexcept {
    return reinterpret_cast<std::atomic<T> *>(ptr)->fetch_sub(value);
}

#ifndef __cpp_lib_atomic_float
inline float lc_atomic_fetch_add(float *ptr, float value) noexcept {
    while (true) {
        auto old = *ptr;
        if (lc_atomic_compare_exchange(ptr, old, old + value) == old) return old;
    }
}
inline float lc_atomic_fetch_sub(float *ptr, float value) noexcept {
    return lc_atomic_fetch_add(ptr, -value);
}
#endif

template<class T>
inline T lc_atomic_fetch_and(T *ptr, T value) noexcept {
    return reinterpret_cast<std::atomic<T> *>(ptr)->fetch_and(value);
}
template<class T>
inline T lc_atomic_fetch_or(T *ptr, T value) noexcept {
    return reinterpret_cast<std::atomic<T> *>(ptr)->fetch_or(value);
}
template<class T>
inline T lc_atomic_fetch_xor(T *ptr, T value) noexcept {
    return reinterpret_cast<std::atomic<T> *>(ptr)->fetch_xor(value);
}
template<class T>
inline T lc_atomic_fetch_min(T *ptr, T value) noexcept {
    while (true) {
        auto old = *ptr;
        if (old <= value) return old;
        if (lc_atomic_compare_exchange(ptr, old, value) == old) return old;
    }
}
template<class T>
inline T lc_atomic_fetch_max(T *ptr, T value) noexcept {
    while (true) {
        auto old = *ptr;
        if (old >= value) return old;
        if (lc_atomic_compare_exchange(ptr, old, value) == old) return old;
    }
}

inline Hit lc_trace_closest(const Accel &accel, const Ray & ray) {
    return accel.trace_closest(accel.handle, &ray);
}
inline bool lc_trace_any(const Accel &accel, const Ray & ray) {
    return accel.trace_any(accel.handle, &ray);
}
inline lc_float4x4 lc_accel_instance_transform(const Accel &accel, lc_uint inst_id) {
    auto m4 = accel.instance_transform(accel.handle, inst_id);
    return lc_bit_cast<lc_float4x4>(m4);
}
inline void set_instance_visibility(const Accel &accel, lc_uint inst_id, bool visible) {
    accel.set_instance_visibility(accel.handle, inst_id, visible);
}
inline void set_instance_transform(const Accel &accel, lc_uint inst_id, const lc_float4x4 &transform) {
    auto m4 = lc_bit_cast<Mat4>(transform);
    accel.set_instance_transform(accel.handle, inst_id, &m4);
}