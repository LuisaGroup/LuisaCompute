

struct alignas(16) Ray {
    float orig_x;
    float orig_y;
    float orig_z;
    float tmin;
    float dir_x;
    float dir_y;
    float dir_z;
    float tmax;
};

struct alignas(8) TriangleHit {
    uint32_t inst;
    uint32_t prim;
    float bary[2];
    float committed_ray_t;
};

using Hit = TriangleHit;

struct alignas(16) Mat4 {
    float _0[16];
};

struct alignas(8) CommitedHit {
    uint32_t inst;
    uint32_t prim;
    float bary[2];
    uint32_t hit_type;
    float committed_ray_t;
};

struct ProceduralHit {
    uint32_t inst;
    uint32_t prim;
};

struct alignas(16) RayQuery {
    CommitedHit hit;
    Ray ray;
    uint8_t mask;
    float cur_committed_ray_t;
    TriangleHit cur_triangle_hit;
    ProceduralHit cur_procedural_hit;
    bool cur_commited;
    bool terminate_on_first;
    bool terminated;
    void *user_data;
    const Accel *accel;
};

using OnHitCallback = void(*)(RayQuery*);

struct Accel {
    const void *handle;
    Hit (*trace_closest)(const void*, const Ray*, uint32_t);
    bool (*trace_any)(const void*, const Ray*, uint32_t);
    void (*set_instance_visibility)(const void*, uint32_t, uint32_t);
    void (*set_instance_transform)(const void*, uint32_t, const Mat4*);
    void (*set_instance_user_id)(const void*, uint32_t, uint32_t);
    Mat4 (*instance_transform)(const void*, uint32_t);
    uint32_t (*instance_user_id)(const void*, uint32_t);
    void (*ray_query)(const void*, RayQuery*, OnHitCallback, OnHitCallback);
};

struct BufferView {
    uint8_t *data;
    size_t size;
    uint64_t ty;
};

struct Texture {
    uint8_t *data;
    uint32_t width;
    uint32_t height;
    uint32_t depth;
    uint8_t storage;
    uint8_t dimension;
    uint8_t mip_levels;
    uint8_t pixel_stride_shift;
    size_t mip_offsets[16];
    uint8_t sampler;
};

struct BindlessArray {
    const BufferView *buffers;
    size_t buffers_count;
    const Texture *texture2ds;
    size_t texture2ds_count;
    const Texture *texture3ds;
    size_t texture3ds_count;
};

struct KernelFnArg {
    enum class Tag {
        Buffer,
        BindlessArray,
        Accel,
        Texture,
        Uniform,
    };

    struct Buffer_Body {
        BufferView _0;
    };

    struct BindlessArray_Body {
        BindlessArray _0;
    };

    struct Accel_Body {
        Accel _0;
    };

    struct Texture_Body {
        Texture _0;
        uint8_t _1;
    };

    struct Uniform_Body {
        const uint8_t *_0;
    };

    Tag tag;
    union {
        Buffer_Body buffer;
        BindlessArray_Body bindless_array;
        Accel_Body accel;
        Texture_Body texture;
        Uniform_Body uniform;
    };
};

struct CpuCustomOp {
    uint8_t *data;
    /// func(data, args); func should modify args in place
    void (*func)(uint8_t*, uint8_t*);
};

struct KernelFnArgs {
    const KernelFnArg *captured;
    size_t captured_count;
    const KernelFnArg *args;
    size_t args_count;
    uint32_t dispatch_id[3];
    uint32_t thread_id[3];
    uint32_t dispatch_size[3];
    uint32_t block_id[3];
    const CpuCustomOp *custom_ops;
    size_t custom_ops_count;
    const void *internal_data;
};
