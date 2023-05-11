template<class T>
inline T lc_cpu_custom_op(const KernelFnArgs *k_args, size_t i, T value) noexcept {
    if (i >= k_args->custom_ops_count) {
        lc_abort_and_print_sll(k_args->internal_data, "Custom op out of bounds: {} >= {}", i,
                               k_args->custom_ops_count);
    }
    auto op = k_args->custom_ops[i];
    op.func(op.data, reinterpret_cast<uint8_t *>(&value));
    return value;
}

template<class T>
inline size_t lc_buffer_size(const KernelFnArgs *k_args, const BufferView &buffer) noexcept {
    return buffer.size / sizeof(T);
}

template<class T>
inline T lc_buffer_read(const KernelFnArgs *k_args, const BufferView &buffer, size_t i) noexcept {
    if (i >= lc_buffer_size<T>(k_args, buffer)) {
        lc_abort_and_print_sll(k_args->internal_data, "Buffer read out of bounds: {} >= {}", i,
                               lc_buffer_size<T>(k_args, buffer));
    }
    return *(reinterpret_cast<const T *>(buffer.data) + i);
}

template<class T>
inline T *lc_buffer_ref(const KernelFnArgs *k_args, const BufferView &buffer, size_t i) noexcept {
    if (i >= lc_buffer_size<T>(k_args, buffer)) {
        lc_abort_and_print_sll(k_args->internal_data, "Buffer ref out of bounds: {} >= {}", i,
                               lc_buffer_size<T>(k_args, buffer));
    }
    return (reinterpret_cast<T *>(buffer.data) + i);
}

template<class T>
inline void lc_buffer_write(const KernelFnArgs *k_args, const BufferView &buffer, size_t i, T value) noexcept {
    if (i >= lc_buffer_size<T>(k_args, buffer)) {
        lc_abort_and_print_sll(k_args->internal_data, "Buffer read out of bounds: {} >= {}", i,
                               lc_buffer_size<T>(k_args, buffer));
    }
    *(reinterpret_cast<T *>(buffer.data) + i) = value;
}

inline BufferView lc_buffer_arg(const KernelFnArgs *k_args, size_t i) noexcept {
    if (i >= k_args->args_count) {
        lc_abort_and_print_sll(k_args->internal_data, "Buffer argument out of bounds: {} >= {}", i,
                               k_args->args_count);
    }
    auto arg = k_args->args[i];
    if (arg.tag != KernelFnArg::Tag::Buffer) {
        lc_abort_and_print_sll(k_args->internal_data, "Buffer argument type mismatch: {} != {}",
                               (unsigned int) arg.tag, (unsigned int) KernelFnArg::Tag::Buffer);
    }
    return arg.buffer._0;
}

inline BufferView lc_buffer_capture(const KernelFnArgs *k_args, size_t i) noexcept {
    if (i >= k_args->captured_count) {
        lc_abort_and_print_sll(k_args->internal_data, "Buffer argument out of bounds: {} >= {}", i,
                               k_args->captured_count);
    }
    auto arg = k_args->captured[i];
    if (arg.tag != KernelFnArg::Tag::Buffer) {
        lc_abort_and_print_sll(k_args->internal_data, "Buffer argument type mismatch: {} != {}",
                               (unsigned int) arg.tag, (unsigned int) KernelFnArg::Tag::Buffer);
    }
    return arg.buffer._0;
}

inline BindlessArray lc_bindless_arg(const KernelFnArgs *k_args, size_t i) noexcept {
    if (i >= k_args->args_count) {
        lc_abort_and_print_sll(k_args->internal_data, "Bindless argument out of bounds: {} >= {}", i,
                               k_args->args_count);
    }
    auto arg = k_args->args[i];
    if (arg.tag != KernelFnArg::Tag::BindlessArray) {
        lc_abort_and_print_sll(k_args->internal_data, "Bindless argument type mismatch: {} != {}",
                               (unsigned int) arg.tag, (unsigned int) KernelFnArg::Tag::BindlessArray);
    }
    return arg.bindless_array._0;
}

inline BindlessArray lc_bindless_capture(const KernelFnArgs *k_args, size_t i) noexcept {
    if (i >= k_args->captured_count) {
        lc_abort_and_print_sll(k_args->internal_data, "Bindless argument out of bounds: {} >= {}", i,
                               k_args->captured_count);
    }
    auto arg = k_args->captured[i];
    if (arg.tag != KernelFnArg::Tag::BindlessArray) {
        lc_abort_and_print_sll(k_args->internal_data, "Bindless argument type mismatch: {} != {}",
                               (unsigned int) arg.tag, (unsigned int) KernelFnArg::Tag::BindlessArray);
    }
    return arg.bindless_array._0;
}

inline BufferView
lc_bindless_buffer(const KernelFnArgs *k_args, const BindlessArray &array, size_t buf_index) noexcept {
    if (buf_index >= array.buffers_count) {
        lc_abort_and_print_sll(k_args->internal_data, "Bindless buffer index out of bounds: {} >= {}", buf_index,
                               array.buffers_count);
    }
    return array.buffers[buf_index];
}

inline uint64_t
lc_bindless_buffer_type(const KernelFnArgs *k_args, const BindlessArray &array, size_t buf_index) noexcept {
    auto buf = lc_bindless_buffer(k_args, array, buf_index);
    return buf.ty;
}

template<class T>
inline T lc_bindless_buffer_read(const KernelFnArgs *k_args, const BindlessArray &array, size_t buf_index,
                                 size_t element) noexcept {
    auto buf = lc_bindless_buffer(k_args, array, buf_index);
    return lc_buffer_read<T>(k_args, buf, element);
}

template<class T>
inline size_t lc_bindless_buffer_size(const KernelFnArgs *k_args, const BindlessArray &array, size_t buf_index) noexcept {
    auto buf = lc_bindless_buffer(k_args, array, buf_index);
    return lc_buffer_size<T>(k_args, buf);
}

template<class T>
inline T lc_atomic_compare_exchange(T *ptr, T expected, T desired) noexcept {
    auto old = expected;
    __atomic_compare_exchange_n(ptr, &old, desired, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    return old;
}

template<class T>
inline T lc_atomic_exchange(T *ptr, T desired) noexcept {
    return __atomic_exchange_n(ptr, desired, __ATOMIC_SEQ_CST);
}

template<class T>
inline T lc_atomic_fetch_add(T *ptr, T value) noexcept {
    return __atomic_fetch_add(ptr, value, __ATOMIC_SEQ_CST);
}

template<class T>
inline T lc_atomic_fetch_sub(T *ptr, T value) noexcept {
    return __atomic_fetch_sub(ptr, value, __ATOMIC_SEQ_CST);
}

//#ifndef __cpp_lib_atomic_float
//inline float lc_atomic_fetch_add(float *ptr, float value) noexcept {
//    while (true) {
//        auto old = *ptr;
//        if (lc_atomic_compare_exchange(ptr, old, old + value) == old) return old;
//    }
//}
//inline float lc_atomic_fetch_sub(float *ptr, float value) noexcept {
//    return lc_atomic_fetch_add(ptr, -value);
//}
//#endif

template<class T>
inline T lc_atomic_fetch_and(T *ptr, T value) noexcept {
    return __atomic_fetch_and(ptr, value, __ATOMIC_SEQ_CST);
}

template<class T>
inline T lc_atomic_fetch_or(T *ptr, T value) noexcept {
    return __atomic_fetch_or(ptr, value, __ATOMIC_SEQ_CST);
}

template<class T>
inline T lc_atomic_fetch_xor(T *ptr, T value) noexcept {
    return __atomic_fetch_xor(ptr, value, __ATOMIC_SEQ_CST);
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

inline Hit lc_trace_closest(const Accel &accel, const Ray &ray, uint8_t mask) noexcept {
    return accel.trace_closest(accel.handle, &ray, mask);
}

inline bool lc_trace_any(const Accel &accel, const Ray &ray, uint8_t mask) noexcept {
    return accel.trace_any(accel.handle, &ray, mask);
}

inline lc_float4x4 lc_accel_instance_transform(const Accel &accel, lc_uint inst_id) noexcept {
    auto m4 = accel.instance_transform(accel.handle, inst_id);
    return lc_bit_cast<lc_float4x4>(m4);
}

inline void set_instance_visibility(const Accel &accel, lc_uint inst_id, bool visible) noexcept {
    accel.set_instance_visibility(accel.handle, inst_id, visible);
}

inline void set_instance_transform(const Accel &accel, lc_uint inst_id, const lc_float4x4 &transform) noexcept {
    auto m4 = lc_bit_cast<Mat4>(transform);
    accel.set_instance_transform(accel.handle, inst_id, &m4);
}