

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

struct alignas(16) Hit {
    uint32_t inst_id;
    uint32_t prim_id;
    float u;
    float v;
};

struct alignas(16) Mat4 {
    float _0[16];
};

struct Accel {
    const void *handle;
    Hit (*trace_closest)(const void*, const Ray*);
    bool (*trace_any)(const void*, const Ray*);
    void (*set_instance_visibility)(const void*, uint32_t, bool);
    void (*set_instance_transform)(const void*, uint32_t, const Mat4*);
    Mat4 (*instance_transform)(const void*, uint32_t);
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
    const Texture *textures;
    size_t textures_count;
};

struct KernelFnArg {
    enum class Tag {
        Buffer,
        BindlessArray,
        Accel,
        Texture,
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
    };

    Tag tag;
    union {
        Buffer_Body buffer;
        BindlessArray_Body bindless_array;
        Accel_Body accel;
        Texture_Body texture;
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
};
