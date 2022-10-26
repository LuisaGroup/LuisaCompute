use bitflags::bitflags;
use std::ffi::c_void;
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct Buffer(pub u64);

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct Context(pub u64);

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct Device(pub u64);

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct Texture(pub u64);

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct Stream(pub u64);

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct Event(pub u64);

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct BindlessArray(pub u64);

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct Mesh(pub u64);

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct Accel(pub u64);

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct IrModule(pub u64);

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct Shader(pub u64);

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct NodeRef(pub u64);

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub enum AccelUsageHint {
    FastTrace,
    FastUpdate,
    FastBuild,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub enum AccelBuildRequest {
    PreferUpdate,
    ForceBuild,
}
// #[repr(C)]
// #[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
// pub struct AccelBuildModificationFlags(pub u32);
// impl AccelBuildModificationFlags {
//     pub const EMPTY: Self = Self(0);
//     pub const MESH: Self = Self(1 << 0);
//     pub const TRANSFORM: Self = Self(1 << 1);
//     pub const VISIBILITY_ON: Self = Self(1 << 2);
//     pub const VISIBILITY_OFF: Self = Self(1 << 3);
//     pub const VISIBILITY: Self = Self(Self::VISIBILITY_ON.0 | Self::VISIBILITY_OFF.0);
// }
bitflags! {
    #[repr(C)]
    pub struct AccelBuildModificationFlags : u32 {
        const EMPTY = 0;
        const MESH = 1 << 0;
        const TRANSFORM = 1 << 1;
        const VISIBILITY_ON = 1 << 2;
        const VISIBILITY_OFF = 1 << 3;
        const VISIBILITY = Self::VISIBILITY_ON.bits | Self::VISIBILITY_OFF.bits;
    }
}
// impl AccelBuildModificationFlags {
//     fn is_empty(&self) -> bool {
//         self.0 == 0
//     }
// }
// impl std::ops::BitAnd for AccelBuildModificationFlags {
//     type Output = Self;
//     fn bitand(self, rhs: Self) -> Self {
//         Self(self.0 & rhs.0)
//     }
// }
// impl std::ops::BitOr for AccelBuildModificationFlags {
//     type Output = Self;
//     fn bitor(self, rhs: Self) -> Self {
//         Self(self.0 | rhs.0)
//     }
// }
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct AccelBuildModification {
    pub index: u32,
    pub flags: AccelBuildModificationFlags,
    pub mesh: u64,
    pub affine: [f32; 12],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub enum PixelStorage {
    Byte1,
    Byte2,
    Byte4,
    Short1,
    Short2,
    Short4,
    Int1,
    Int2,
    Int4,
    Half1,
    Half2,
    Half4,
    Float1,
    Float2,
    Float4,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub enum PixelFormat {
    R8Sint,
    R8Uint,
    R8Unorm,

    Rgb8Sint,
    Rgb8Uint,
    Rgb8Unorm,

    Rgba8Sint,
    Rgba8Uint,
    Rgba8Unorm,

    R16Sint,
    R16Uint,
    R16Unorm,

    Rg16Sint,
    Rg16Uint,
    Rg16Unorm,

    Rgba16Sint,
    Rgba16Uint,
    Rgba16Unorm,

    R32Sint,
    R32Uint,

    Rg32Sint,
    Rg32Uint,

    R16f,
    Rg32f,
    Rgba32f,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub enum SamplerFilter {
    Point,
    LinearPoint,
    LinearLinear,
    Anisotropic,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub enum SamplerAddress {
    Edge,
    Repeat,
    Mirror,
    Zero,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct Sampler {
    pub filter: SamplerFilter,
    pub address: SamplerAddress,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct BufferArgument {
    pub buffer: Buffer,
    pub offset: usize,
    pub size: usize,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct TextureArgument {
    pub texture: Texture,
    pub level: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct UniformArgument {
    pub data: *const u8,
    pub size: usize,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub enum Argument {
    Buffer(BufferArgument),
    Texture(TextureArgument),
    Uniform(UniformArgument),
    Accel(Accel),
    BindlessArray(BindlessArray),
}
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct Capture {
    pub node: NodeRef,
    pub arg: Argument,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct KernelModule {
    pub ir_module: IrModule,
    pub captured: *const Capture,
    pub captured_count: usize,
    pub args: *const NodeRef,
    pub args_count: usize,
    pub shared: *const NodeRef,
    pub shared_count: usize,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct CallableModule {
    pub ir_module: IrModule,
    pub args: *const NodeRef,
    pub args_count: usize,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct BufferUploadCommand {
    pub buffer: Buffer,
    pub offset: usize,
    pub size: usize,
    pub data: *const u8,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct BufferDownloadCommand {
    pub buffer: Buffer,
    pub offset: usize,
    pub size: usize,
    pub data: *mut u8,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct BufferCopyCommand {
    pub src: Buffer,
    pub src_offset: usize,
    pub dst: Buffer,
    pub dst_offset: usize,
    pub size: usize,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct BufferToTextureCopyCommand {
    pub buffer: Buffer,
    pub buffer_offset: usize,
    pub texture: Texture,
    pub storage: PixelStorage,
    pub texture_level: u32,
    pub texture_size: [u32; 3],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct TextureToBufferCopyCommand {
    pub buffer: Buffer,
    pub buffer_offset: usize,
    pub texture: Texture,
    pub storage: PixelStorage,
    pub texture_level: u32,
    pub texture_size: [u32; 3],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct TextureUploadCommand {
    pub texture: Texture,
    pub storage: PixelStorage,
    pub level: u32,
    pub size: [u32; 3],
    pub data: *const u8,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct TextureDownloadCommand {
    pub texture: Texture,
    pub storage: PixelStorage,
    pub level: u32,
    pub size: [u32; 3],
    pub data: *mut u8,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct ShaderDispatchCommand {
    pub shader: Shader,
    pub m: KernelModule,
    pub dispatch_size: [u32; 3],
    pub args: *const Argument,
    pub args_count: usize,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct MeshBuildCommand {
    pub mesh: Mesh,
    pub request: AccelBuildRequest,
    pub vertex_buffer: Buffer,
    pub vertex_buffer_offset: usize,
    pub vertex_buffer_size: usize,
    pub index_buffer: Buffer,
    pub index_buffer_offset: usize,
    pub index_buffer_size: usize,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct AccelBuildCommand {
    pub accel: Accel,
    pub request: AccelBuildRequest,
    pub instance_count: u32,
    pub modifications: *const AccelBuildModification,
    pub modifications_count: usize,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub enum Command {
    BufferUpload(BufferUploadCommand),
    BufferDownload(BufferDownloadCommand),
    BufferCopy(BufferCopyCommand),
    BufferToTextureCopy(BufferToTextureCopyCommand),
    TextureToBufferCopy(TextureToBufferCopyCommand),
    TextureUpload(TextureUploadCommand),
    TextureDownload(TextureDownloadCommand),
    ShaderDispatch(ShaderDispatchCommand),
    MeshBuild(MeshBuildCommand),
    AccelBuild(AccelBuildCommand),
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct CommandList {
    pub commands: *const Command,
    pub commands_count: usize,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct AppContext {
    pub gc_context: *mut c_void,
    pub ir_context: *mut c_void,
}
pub fn __dummy() {}
