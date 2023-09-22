use std::ffi::c_void;

#[derive(Clone, Copy)]
#[repr(C)]
pub struct KernelFnArgs {
    pub captured: *const KernelFnArg,
    pub captured_count: usize,
    pub args: *const KernelFnArg,
    pub args_count: usize,
    pub dispatch_id: [u32; 3],
    pub thread_id: [u32; 3],
    pub dispatch_size: [u32; 3],
    pub block_id: [u32; 3],
    pub custom_ops: *const CpuCustomOp,
    pub custom_ops_count: usize,
    pub internal_data: *const c_void,
}
#[repr(C)]
pub struct CpuCustomOp {
    pub data: *mut u8,
    /// func(data, args); func should modify args in place
    pub func: extern "C" fn(*mut u8, *mut u8),
}
unsafe impl Send for KernelFnArgs {}
unsafe impl Sync for KernelFnArgs {}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct BufferView {
    pub data: *mut u8,
    pub size: usize,
    pub ty: u64,
}
impl Default for BufferView {
    fn default() -> Self {
        Self {
            data: std::ptr::null_mut(),
            size: 0,
            ty: 0,
        }
    }
}
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug)]
pub struct Ray {
    pub orig_x: f32,
    pub orig_y: f32,
    pub orig_z: f32,
    pub tmin: f32,
    pub dir_x: f32,
    pub dir_y: f32,
    pub dir_z: f32,
    pub tmax: f32,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Aabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}
#[derive(Clone, Copy)]
#[repr(u32)]
pub enum HitType {
    Miss = 0,
    Triangle = 1,
    Procedural = 2,
}

#[repr(C, align(8))]
#[derive(Copy, Clone, Default, Debug)]
pub struct CommitedHit {
    pub inst: u32,
    pub prim: u32,
    pub bary: [f32; 2],
    pub hit_type: u32,
    pub committed_ray_t: f32,
}
impl CommitedHit {
    pub fn set_from_triangle_hit(&mut self, hit: TriangleHit) {
        self.inst = hit.inst;
        self.prim = hit.prim;
        self.bary = hit.bary;
        self.hit_type = HitType::Triangle as u32;
        self.committed_ray_t = hit.committed_ray_t;
    }
    pub fn set_from_procedural_hit(&mut self, hit: ProceduralHit, t: f32) {
        self.inst = hit.inst;
        self.prim = hit.prim;
        self.hit_type = HitType::Procedural as u32;
        self.committed_ray_t = t;
    }
}
#[repr(C, align(8))]
#[derive(Copy, Clone, Debug)]
pub struct TriangleHit {
    pub inst: u32,
    pub prim: u32,
    pub bary: [f32; 2],
    pub committed_ray_t: f32,
}
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ProceduralHit {
    pub inst: u32,
    pub prim: u32,
}
pub type Hit = TriangleHit;
#[repr(C, align(16))]
#[derive(Copy, Clone)]
pub struct Mat4(pub [f32; 16]);

pub type OnHitCallback = extern "C" fn(&mut RayQuery);
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Accel {
    pub handle: *const c_void,
    pub trace_closest: extern "C" fn(*const c_void, &Ray, u32) -> Hit,
    pub trace_any: extern "C" fn(*const c_void, &Ray, u32) -> bool,
    pub set_instance_visibility: extern "C" fn(*const c_void, u32, u32),
    pub set_instance_transform: extern "C" fn(*const c_void, u32, &Mat4),
    pub set_instance_user_id: extern "C" fn(*const c_void, u32, u32),
    pub instance_transform: extern "C" fn(*const c_void, u32) -> Mat4,
    pub instance_user_id: extern "C" fn(*const c_void, u32) -> u32,
    pub ray_query: extern "C" fn(*const c_void, &mut RayQuery, OnHitCallback, OnHitCallback),
}

#[repr(C, align(16))]
#[derive(Copy, Clone)]
pub struct RayQuery {
    pub hit: CommitedHit,
    pub ray: Ray,
    pub mask: u8,
    pub cur_committed_ray_t: f32,
    pub cur_triangle_hit: TriangleHit,
    pub cur_procedural_hit: ProceduralHit,
    pub cur_commited: bool,
    pub terminate_on_first: bool,
    pub terminated: bool,
    pub user_data: *mut c_void,
    pub accel: *const Accel,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct BindlessArray {
    pub buffers: *const BufferView,
    pub buffers_count: usize,
    pub texture2ds: *const Texture,
    pub texture2ds_count: usize,
    pub texture3ds: *const Texture,
    pub texture3ds_count: usize,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Texture {
    pub data: *mut u8,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub storage: u8,
    pub dimension: u8,
    pub mip_levels: u8,
    pub pixel_stride_shift: u8,
    pub mip_offsets: [usize; 16],
    pub sampler: u8,
}
impl Default for Texture {
    fn default() -> Self {
        Self {
            data: std::ptr::null_mut(),
            width: 0,
            height: 0,
            depth: 0,
            storage: 0,
            dimension: 0,
            mip_levels: 0,
            pixel_stride_shift: 0,
            mip_offsets: [0; 16],
            sampler: 0,
        }
    }
}
#[derive(Clone, Copy)]
#[repr(C)]
pub enum KernelFnArg {
    Buffer(BufferView),
    BindlessArray(BindlessArray),
    Accel(Accel),
    Texture(Texture, u8),
    Uniform(*const u8),
}
