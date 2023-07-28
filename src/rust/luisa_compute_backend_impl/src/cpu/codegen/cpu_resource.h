template<class T>
inline T lc_cpu_custom_op(const KernelFnArgs *k_args, size_t i, T value) noexcept {
#ifdef LUISA_DEBUG
    if (i >= k_args->custom_ops_count) {
        lc_abort_and_print_sll(k_args->internal_data, "Custom op out of bounds: {} >= {}", i,
                               k_args->custom_ops_count);
    }
#endif
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
#ifdef LUISA_DEBUG
    if (i >= lc_buffer_size<T>(k_args, buffer)) {
        lc_abort_and_print_sll(k_args->internal_data, "Buffer read out of bounds: {} >= {}", i,
                               lc_buffer_size<T>(k_args, buffer));
    }
#endif
    return *(reinterpret_cast<const T *>(buffer.data) + i);
}

template<class T>
inline T *lc_buffer_ref(const KernelFnArgs *k_args, const BufferView &buffer, size_t i) noexcept {
#ifdef LUISA_DEBUG
    if (i >= lc_buffer_size<T>(k_args, buffer)) {
        lc_abort_and_print_sll(k_args->internal_data, "Buffer ref out of bounds: {} >= {}", i,
                               lc_buffer_size<T>(k_args, buffer));
    }
#endif
    return (reinterpret_cast<T *>(buffer.data) + i);
}

template<class T>
inline void lc_buffer_write(const KernelFnArgs *k_args, const BufferView &buffer, size_t i, T value) noexcept {
#ifdef LUISA_DEBUG
    if (i >= lc_buffer_size<T>(k_args, buffer)) {
        lc_abort_and_print_sll(k_args->internal_data, "Buffer read out of bounds: {} >= {}", i,
                               lc_buffer_size<T>(k_args, buffer));
    }
#endif
    *(reinterpret_cast<T *>(buffer.data) + i) = value;
}

inline BufferView lc_buffer_arg(const KernelFnArgs *k_args, size_t i) noexcept {
#ifdef LUISA_DEBUG
    if (i >= k_args->args_count) {
        lc_abort_and_print_sll(k_args->internal_data, "Buffer argument out of bounds: {} >= {}", i,
                               k_args->args_count);
    }
#endif
    auto arg = k_args->args[i];
#ifdef LUISA_DEBUG
    if (arg.tag != KernelFnArg::Tag::Buffer) {
        lc_abort_and_print_sll(k_args->internal_data, "Buffer argument type mismatch: {} != {}",
                               (unsigned int)arg.tag, (unsigned int)KernelFnArg::Tag::Buffer);
    }
#endif
    return arg.buffer._0;
}

inline BufferView lc_buffer_capture(const KernelFnArgs *k_args, size_t i) noexcept {
#ifdef LUISA_DEBUG
    if (i >= k_args->captured_count) {
        lc_abort_and_print_sll(k_args->internal_data, "Buffer argument out of bounds: {} >= {}", i,
                               k_args->captured_count);
    }
#endif
    auto arg = k_args->captured[i];
#ifdef LUISA_DEBUG
    if (arg.tag != KernelFnArg::Tag::Buffer) {
        lc_abort_and_print_sll(k_args->internal_data, "Buffer argument type mismatch: {} != {}",
                               (unsigned int)arg.tag, (unsigned int)KernelFnArg::Tag::Buffer);
    }
#endif
    return arg.buffer._0;
}

inline BindlessArray lc_bindless_arg(const KernelFnArgs *k_args, size_t i) noexcept {
#ifdef LUISA_DEBUG
    if (i >= k_args->args_count) {
        lc_abort_and_print_sll(k_args->internal_data, "Bindless argument out of bounds: {} >= {}", i,
                               k_args->args_count);
    }
#endif
    auto arg = k_args->args[i];
#ifdef LUISA_DEBUG
    if (arg.tag != KernelFnArg::Tag::BindlessArray) {
        lc_abort_and_print_sll(k_args->internal_data, "Bindless argument type mismatch: {} != {}",
                               (unsigned int)arg.tag, (unsigned int)KernelFnArg::Tag::BindlessArray);
    }
#endif
    return arg.bindless_array._0;
}

inline BindlessArray lc_bindless_capture(const KernelFnArgs *k_args, size_t i) noexcept {
#ifdef LUISA_DEBUG
    if (i >= k_args->captured_count) {
        lc_abort_and_print_sll(k_args->internal_data, "Bindless argument out of bounds: {} >= {}", i,
                               k_args->captured_count);
    }
#endif
    auto arg = k_args->captured[i];
#ifdef LUISA_DEBUG
    if (arg.tag != KernelFnArg::Tag::BindlessArray) {
        lc_abort_and_print_sll(k_args->internal_data, "Bindless argument type mismatch: {} != {}",
                               (unsigned int)arg.tag, (unsigned int)KernelFnArg::Tag::BindlessArray);
    }
#endif
    return arg.bindless_array._0;
}

inline BufferView
lc_bindless_buffer(const KernelFnArgs *k_args, const BindlessArray &array, size_t buf_index) noexcept {
#ifdef LUISA_DEBUG
    if (buf_index >= array.buffers_count) {
        lc_abort_and_print_sll(k_args->internal_data, "Bindless buffer index out of bounds: {} >= {}", buf_index,
                               array.buffers_count);
    }
#endif
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


inline size_t lc_bindless_buffer_size(const KernelFnArgs *k_args, const BindlessArray &array, size_t buf_index, size_t stride) noexcept {
    auto buf = lc_bindless_buffer(k_args, array, buf_index);
    return buf.size / stride;
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

inline void lc_set_instance_visibility(const Accel &accel, lc_uint inst_id, bool visible) noexcept {
    accel.set_instance_visibility(accel.handle, inst_id, visible);
}

inline void lc_set_instance_transform(const Accel &accel, lc_uint inst_id, const lc_float4x4 &transform) noexcept {
    auto m4 = lc_bit_cast<Mat4>(transform);
    accel.set_instance_transform(accel.handle, inst_id, &m4);
}
using LC_RayQueryAll = RayQuery;
using LC_RayQueryAny = RayQuery;

template<bool TERMINATE_ON_FISRST_HIT>
inline RayQuery make_rq(const Accel &accel, const Ray &ray, uint8_t mask) {
    RayQuery rq{};
    rq.hit.inst = ~0u;
    rq.hit.prim = ~0u;
    rq.ray = ray;
    rq.mask = mask;
    rq.user_data = nullptr;
    rq.terminate_on_first = TERMINATE_ON_FISRST_HIT;
    rq.accel = &accel;
    return rq;
}

inline LC_RayQueryAny lc_ray_query_any(const Accel &accel, const Ray &ray, uint8_t mask) {
    return make_rq<true>(accel, ray, mask);
}
inline LC_RayQueryAll lc_ray_query_all(const Accel &accel, const Ray &ray, uint8_t mask) {
    return make_rq<false>(accel, ray, mask);
}
inline Ray lc_ray_query_world_space_ray(const RayQuery &rq) {
    return rq.ray;
}

inline TriangleHit lc_ray_query_triangle_candidate_hit(const RayQuery &rq) {
    return rq.cur_triangle_hit;
}
inline ProceduralHit lc_ray_query_procedural_candidate_hit(const RayQuery &rq) {
    return rq.cur_procedural_hit;
}
inline void lc_ray_query_commit_triangle(RayQuery &rq) {
    rq.cur_commited = true;
}
inline void lc_ray_query_commit_procedural(RayQuery &rq, float t) {
    rq.cur_commited = true;
    rq.cur_committed_ray_t = t;
}
inline void lc_ray_query_terminate(RayQuery &rq) {
    rq.terminated = true;
}

template<class T, class P>
struct Callbacks {
    T on_triangle_hit;
    P on_procedural_hit;
};
template<class T, class P>
void on_triangle_hit_wrapper(RayQuery *rq) {
    auto callbacks = reinterpret_cast<Callbacks<T, P> *>(rq->user_data);
    callbacks->on_triangle_hit(rq->cur_triangle_hit);
}
template<class T, class P>
void on_procedural_hit_wrapper(RayQuery *rq) {
    auto callbacks = reinterpret_cast<Callbacks<T, P> *>(rq->user_data);
    callbacks->on_procedural_hit(rq->cur_procedural_hit);
}
template<class T, class P>
inline void lc_ray_query(RayQuery &rq, T on_triangle_hit, P on_procedural_hit) {
    auto callbacks = Callbacks<T, P>{on_triangle_hit, on_procedural_hit};
    rq.user_data = &callbacks;
    auto accel = rq.accel;
    return accel->ray_query(accel->handle, &rq, on_triangle_hit_wrapper<T, P>, on_procedural_hit_wrapper<T, P>);
}
inline CommitedHit lc_ray_query_committed_hit(RayQuery &rq) {
    return rq.hit;
}