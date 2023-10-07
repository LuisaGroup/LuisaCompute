use std::{os::raw::c_void, ptr::null_mut};

use super::resource::BufferImpl;
use crate::panic_abort;
use api::{
    AccelBuildModification, AccelBuildModificationFlags, AccelBuildRequest, AccelUsageHint,
    MeshBuildCommand, ProceduralPrimitiveBuildCommand,
};
use embree_sys as sys;
use lazy_static::lazy_static;
use luisa_compute_api_types as api;
use luisa_compute_cpu_kernel_defs as defs;
use parking_lot::{Mutex, RwLock};
struct Device(sys::RTCDevice);
unsafe impl Send for Device {}
unsafe impl Sync for Device {}

lazy_static! {
    static ref DEVICE: Mutex<Device> = Mutex::new(Device(std::ptr::null_mut()));
}
fn init_device() {
    let mut device = DEVICE.lock();
    if device.0.is_null() {
        device.0 = unsafe { sys::rtcNewDevice(std::ptr::null()) }
    }
}
pub struct GeometryImpl {
    pub(crate) handle: sys::RTCScene,
    #[allow(dead_code)]
    usage: AccelUsageHint,
    built: bool,
    lock: Mutex<()>,
}
macro_rules! check_error {
    ($device:expr) => {{
        let err = sys::rtcGetDeviceError($device);
        if err != sys::RTC_ERROR_NONE {
            panic_abort!("Embree error: {}", err);
        }
    }};
}
impl GeometryImpl {
    pub unsafe fn new(
        hint: api::AccelUsageHint,
        _allow_compact: bool,
        _allow_update: bool,
    ) -> Self {
        init_device();
        let device = DEVICE.lock();
        let handle = sys::rtcNewScene(device.0);
        let flags = match hint {
            AccelUsageHint::FastBuild => sys::RTC_BUILD_QUALITY_LOW,
            AccelUsageHint::FastTrace => sys::RTC_BUILD_QUALITY_HIGH,
        };

        sys::rtcSetSceneFlags(handle, flags);

        Self {
            handle,
            usage: hint,
            built: false,
            lock: Mutex::new(()),
        }
    }
    pub unsafe fn build_procedural(&mut self, cmd: &ProceduralPrimitiveBuildCommand) {
        let device = DEVICE.lock();
        let device = device.0;
        let _lk = self.lock.lock();
        let request = cmd.request;
        let need_rebuild = request == AccelBuildRequest::ForceBuild || !self.built;

        unsafe extern "C" fn bounds_func(args: *const sys::RTCBoundsFunctionArguments) {
            let args = &*args;
            let aabb_buffer = args.geometryUserPtr as *const defs::Aabb;
            let aabb = unsafe { &*aabb_buffer.add(args.primID as usize) };
            let bounds = &mut *(args.bounds_o as *mut sys::RTCBounds);
            bounds.lower_x = aabb.min[0];
            bounds.lower_y = aabb.min[1];
            bounds.lower_z = aabb.min[2];
            bounds.upper_x = aabb.max[0];
            bounds.upper_y = aabb.max[1];
            bounds.upper_z = aabb.max[2];
        }
        let aabb_buffer = &*(cmd.aabb_buffer.0 as *const BufferImpl);

        if need_rebuild {
            let geometry = sys::rtcNewGeometry(device, sys::RTC_GEOMETRY_TYPE_USER);

            sys::rtcSetGeometryUserData(geometry, aabb_buffer.data as *mut c_void);
            sys::rtcSetGeometryUserPrimitiveCount(geometry, cmd.aabb_count as u32);
            sys::rtcSetGeometryBoundsFunction(geometry, Some(bounds_func), null_mut());

            check_error!(device);
            sys::rtcCommitGeometry(geometry);
            check_error!(device);
            if self.built {
                sys::rtcDetachGeometry(self.handle, 0);
                check_error!(device);
            } else {
                self.built = true;
            }
            sys::rtcAttachGeometryByID(self.handle, geometry, 0);
            check_error!(device);
            sys::rtcReleaseGeometry(geometry);
            check_error!(device);
        } else {
            let geometry = sys::rtcGetGeometry(self.handle, 0);
            sys::rtcSetGeometryUserPrimitiveCount(geometry, cmd.aabb_count as u32);
            check_error!(device);
            sys::rtcSetGeometryBoundsFunction(
                geometry,
                Some(bounds_func),
                (aabb_buffer.data as *mut u8).add(cmd.aabb_buffer_offset) as *mut c_void,
            );
            check_error!(device);
            sys::rtcCommitGeometry(geometry);
            check_error!(device);
        }
        sys::rtcCommitScene(self.handle);
        check_error!(device);
    }
    pub unsafe fn build_mesh(&mut self, cmd: &MeshBuildCommand) {
        let device = DEVICE.lock();
        let device = device.0;
        let _lk = self.lock.lock();
        let request = cmd.request;
        let need_rebuild = request == AccelBuildRequest::ForceBuild || !self.built;

        if need_rebuild {
            let geometry = sys::rtcNewGeometry(device, sys::RTC_GEOMETRY_TYPE_TRIANGLE);
            let vbuffer = &*(cmd.vertex_buffer.0 as *const BufferImpl);
            let ibuffer = &*(cmd.index_buffer.0 as *const BufferImpl);
            sys::rtcSetSharedGeometryBuffer(
                geometry,
                sys::RTC_BUFFER_TYPE_VERTEX,
                0,
                sys::RTC_FORMAT_FLOAT3,
                vbuffer.data as *const c_void,
                cmd.vertex_buffer_offset,
                cmd.vertex_stride,
                cmd.vertex_buffer_size / cmd.vertex_stride,
            );
            check_error!(device);
            sys::rtcSetSharedGeometryBuffer(
                geometry,
                sys::RTC_BUFFER_TYPE_INDEX,
                0,
                sys::RTC_FORMAT_UINT3,
                ibuffer.data as *const c_void,
                cmd.index_buffer_offset,
                cmd.index_stride,
                cmd.index_buffer_size / cmd.index_stride,
            );
            check_error!(device);
            sys::rtcCommitGeometry(geometry);
            check_error!(device);
            if self.built {
                sys::rtcDetachGeometry(self.handle, 0);
                check_error!(device);
            } else {
                self.built = true;
            }
            sys::rtcAttachGeometryByID(self.handle, geometry, 0);
            check_error!(device);
            sys::rtcReleaseGeometry(geometry);
            check_error!(device);
        } else {
            let geometry = sys::rtcGetGeometry(self.handle, 0);
            sys::rtcUpdateGeometryBuffer(geometry, sys::RTC_BUFFER_TYPE_VERTEX, 0);
            check_error!(device);
            sys::rtcCommitGeometry(geometry);
            check_error!(device);
        }
        sys::rtcCommitScene(self.handle);
        check_error!(device);
    }
}
impl Drop for GeometryImpl {
    fn drop(&mut self) {
        unsafe {
            sys::rtcReleaseScene(self.handle);
        }
    }
}
#[derive(Clone, Copy)]
struct Instance {
    affine: [f32; 12],
    user_id: u32,
    visible: u32,
    opaque: bool,
    dirty: bool,
    geometry: sys::RTCGeometry,
}
impl Instance {
    pub fn valid(&self) -> bool {
        self.geometry != std::ptr::null_mut()
    }
}
impl Default for Instance {
    fn default() -> Self {
        Self {
            affine: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            user_id: 0,
            visible: 0xff,
            opaque: true,
            dirty: false,
            geometry: std::ptr::null_mut(),
        }
    }
}
pub struct AccelImpl {
    pub(crate) handle: sys::RTCScene,
    instances: Vec<RwLock<Instance>>,
    instances_fast: Vec<Instance>,
}
#[derive(Clone, Copy)]
#[repr(C)]
struct RayQueryContext {
    parent: sys::RTCRayQueryContext,
    rq: *mut defs::RayQuery,
    on_triangle_hit: defs::OnHitCallback,
    on_procedural_hit: defs::OnHitCallback,
}
impl AccelImpl {
    pub unsafe fn new() -> Self {
        init_device();
        let device = DEVICE.lock();
        let handle = sys::rtcNewScene(device.0);
        Self {
            handle,
            instances: Vec::new(),
            instances_fast: Vec::new(),
        }
    }
    pub unsafe fn update(
        &mut self,
        instance_count: usize,
        modifications: &[AccelBuildModification],
        update_instance_buffer_only: bool,
    ) {
        let device = DEVICE.lock();
        let device = device.0;
        for instance in &self.instances {
            let instance = instance.read();
            let geometry = instance.geometry;
            if instance.dirty {
                sys::rtcSetGeometryTransform(
                    geometry,
                    0,
                    sys::RTC_FORMAT_FLOAT3X4_ROW_MAJOR,
                    instance.affine.as_ptr() as *const c_void,
                );
                sys::rtcSetGeometryMask(geometry, instance.visible as u32);
            }
        }
        while instance_count > self.instances.len() {
            self.instances.push(RwLock::new(Instance::default()));
        }
        while instance_count < self.instances.len() {
            let last = self.instances.pop().unwrap().into_inner();
            if last.valid() {
                sys::rtcDetachGeometry(self.handle, self.instances.len() as u32);
            }
        }
        for m in modifications {
            if m.flags.contains(AccelBuildModificationFlags::PRIMITIVE) {
                let mesh = &*(m.mesh as *const GeometryImpl);
                if !mesh.built {
                    panic_abort!("Mesh not built");
                }
                unsafe {
                    let affine = m.affine;
                    let geometry = sys::rtcNewGeometry(device, sys::RTC_GEOMETRY_TYPE_INSTANCE);
                    sys::rtcCommitGeometry(geometry);
                    sys::rtcSetGeometryInstancedScene(geometry, mesh.handle);
                    sys::rtcAttachGeometryByID(self.handle, geometry, m.index);
                    sys::rtcSetGeometryEnableFilterFunctionFromArguments(geometry, false);
                    *self.instances[m.index as usize].write() = Instance {
                        affine,
                        visible: 0xff,
                        user_id: m.user_id,
                        opaque: true,
                        dirty: false,
                        geometry,
                    };
                }
            }
            if m.flags.contains(AccelBuildModificationFlags::OPAQUE_ON) {
                let mut instance = self.instances[m.index as usize].write();
                instance.opaque = true;
                instance.dirty = true;
                sys::rtcSetGeometryEnableFilterFunctionFromArguments(
                    instance.geometry,
                    !instance.opaque,
                );
            } else if m.flags.contains(AccelBuildModificationFlags::OPAQUE_OFF) {
                let mut instance = self.instances[m.index as usize].write();
                instance.opaque = false;
                instance.dirty = true;
                sys::rtcSetGeometryEnableFilterFunctionFromArguments(
                    instance.geometry,
                    !instance.opaque,
                );
            };
            if m.flags.contains(AccelBuildModificationFlags::TRANSFORM) {
                let mut instance = self.instances[m.index as usize].write();
                let geometry = instance.geometry;
                assert!(!geometry.is_null());
                let affine = m.affine;
                sys::rtcSetGeometryTransform(
                    geometry,
                    0,
                    sys::RTC_FORMAT_FLOAT3X4_ROW_MAJOR,
                    affine.as_ptr() as *const c_void,
                );
                instance.affine = affine;
                instance.dirty = true;
            }
            if m.flags.contains(AccelBuildModificationFlags::VISIBILITY) {
                let mut instance = self.instances[m.index as usize].write();
                let geometry = instance.geometry;
                assert!(!geometry.is_null());
                sys::rtcEnableGeometry(geometry);
                sys::rtcSetGeometryMask(geometry, m.visibility as u32);
                instance.visible = m.visibility;
                instance.dirty = true;
            }
            if m.flags.contains(AccelBuildModificationFlags::USER_ID) {
                let mut instance = self.instances[m.index as usize].write();
                let geometry = instance.geometry;
                assert!(!geometry.is_null());
                // TODO: check if this should work
                sys::rtcSetGeometryUserData(geometry, m.user_id as u64 as *mut c_void);
                instance.user_id = m.user_id;
                instance.dirty = true;
            }
        }
        if update_instance_buffer_only {
            return;
        }
        for instance in &self.instances {
            let mut instance = instance.write();
            if instance.valid() && instance.dirty {
                sys::rtcCommitGeometry(instance.geometry);
                instance.dirty = false;
            }
        }

        sys::rtcCommitScene(self.handle);
        self.instances_fast.clear();
        for instance in &self.instances {
            let instance = instance.read();
            if instance.valid() {
                self.instances_fast.push(*instance);
            }
        }
    }
    #[inline]
    pub unsafe fn trace_closest(&self, ray: &defs::Ray, mask: u32) -> defs::Hit {
        let mut rayhit = sys::RTCRayHit {
            ray: sys::RTCRay {
                org_x: ray.orig_x,
                org_y: ray.orig_y,
                org_z: ray.orig_z,
                tnear: ray.tmin,
                dir_x: ray.dir_x,
                dir_y: ray.dir_y,
                dir_z: ray.dir_z,
                time: 0.0,
                tfar: ray.tmax,
                mask: mask as u32,
                id: 0,
                flags: 0,
            },
            hit: sys::RTCHit {
                Ng_x: 0.0,
                Ng_y: 0.0,
                Ng_z: 0.0,
                u: 0.0,
                v: 0.0,
                primID: u32::MAX,
                geomID: u32::MAX,
                instID: [u32::MAX],
            },
        };
        let mut args = sys::RTCIntersectArguments {
            flags: sys::RTC_RAY_QUERY_FLAG_INCOHERENT,
            feature_mask: sys::RTC_FEATURE_FLAG_ALL,
            filter: None,
            intersect: None,
            context: std::ptr::null_mut(),
        };

        sys::rtcIntersect1(self.handle, &mut rayhit as *mut _, &mut args as *mut _);
        if rayhit.hit.geomID != u32::MAX && rayhit.hit.primID != u32::MAX {
            defs::TriangleHit {
                inst: rayhit.hit.instID[0],
                prim: rayhit.hit.primID,
                bary: [rayhit.hit.u, rayhit.hit.v],
                committed_ray_t: rayhit.ray.tfar,
            }
        } else {
            defs::TriangleHit {
                inst: u32::MAX,
                prim: u32::MAX,
                bary: [0.0, 0.0],
                committed_ray_t: ray.tmax,
            }
        }
    }
    #[inline]
    pub unsafe fn trace_any(&self, ray: &defs::Ray, mask: u32) -> bool {
        let mut ray = sys::RTCRay {
            org_x: ray.orig_x,
            org_y: ray.orig_y,
            org_z: ray.orig_z,
            tnear: ray.tmin,
            dir_x: ray.dir_x,
            dir_y: ray.dir_y,
            dir_z: ray.dir_z,
            time: 0.0,
            tfar: ray.tmax,
            mask: mask as u32,
            id: 0,
            flags: 0,
        };
        let mut args = sys::RTCOccludedArguments {
            flags: sys::RTC_RAY_QUERY_FLAG_INCOHERENT,
            feature_mask: sys::RTC_FEATURE_FLAG_ALL,
            filter: None,
            occluded: None,
            context: std::ptr::null_mut(),
        };
        sys::rtcOccluded1(self.handle, &mut ray as *mut _, &mut args as *mut _);
        ray.tfar < 0.0
    }
    #[inline]
    pub unsafe fn instance_transform(&self, id: u32) -> [f32; 12] {
        let geometry = sys::rtcGetGeometry(self.handle, id);
        let mut affine = [0.0; 12];
        sys::rtcGetGeometryTransform(
            geometry,
            0.0,
            sys::RTC_FORMAT_FLOAT3X4_ROW_MAJOR,
            affine.as_mut_ptr() as *mut c_void,
        );
        affine
    }
    #[inline]
    pub unsafe fn instance_user_id(&self, id: u32) -> u32 {
        let geometry = sys::rtcGetGeometry(self.handle, id);
        sys::rtcGetGeometryUserData(geometry) as u64 as u32
    }
    #[inline]
    pub unsafe fn set_instance_transform(&self, id: u32, affine: [f32; 12]) {
        let mut instance = self.instances[id as usize].write();
        assert!(instance.valid());
        instance.affine = affine;
        instance.dirty = true;
    }
    #[inline]
    pub unsafe fn set_instance_visibility(&self, id: u32, visibility: u32) {
        let mut instance = self.instances[id as usize].write();
        assert!(instance.valid());
        instance.visible = visibility as u32;
        instance.dirty = true;
    }
    #[inline]
    pub unsafe fn set_instance_user_id(&self, id: u32, user_id: u32) {
        let mut instance = self.instances[id as usize].write();
        assert!(instance.valid());
        instance.user_id = user_id;
        instance.dirty = true;
    }

    #[inline]
    pub unsafe fn ray_query(
        &self,
        rq: &mut defs::RayQuery,
        on_triangle_hit: defs::OnHitCallback,
        on_procedural_hit: defs::OnHitCallback,
    ) {
        let ray = rq.ray;
        let mut ray = sys::RTCRay {
            org_x: ray.orig_x,
            org_y: ray.orig_y,
            org_z: ray.orig_z,
            tnear: ray.tmin,
            dir_x: ray.dir_x,
            dir_y: ray.dir_y,
            dir_z: ray.dir_z,
            time: 0.0,
            tfar: ray.tmax,
            mask: rq.mask as u32,
            id: 0,
            flags: 0,
        };
        let mut ctx = RayQueryContext {
            parent: sys::RTCRayQueryContext {
                instID: [u32::MAX; 1],
            },
            rq,
            on_procedural_hit,
            on_triangle_hit,
        };

        /* Helper functions to access ray packets of runtime size N */
        // RTC_FORCEINLINE float& RTCRayN_org_x(RTCRayN* ray, unsigned int N, unsigned int i) { return ((float*)ray)[0*N+i]; }
        // RTC_FORCEINLINE float& RTCRayN_org_y(RTCRayN* ray, unsigned int N, unsigned int i) { return ((float*)ray)[1*N+i]; }
        // RTC_FORCEINLINE float& RTCRayN_org_z(RTCRayN* ray, unsigned int N, unsigned int i) { return ((float*)ray)[2*N+i]; }
        // RTC_FORCEINLINE float& RTCRayN_tnear(RTCRayN* ray, unsigned int N, unsigned int i) { return ((float*)ray)[3*N+i]; }

        // RTC_FORCEINLINE float& RTCRayN_dir_x(RTCRayN* ray, unsigned int N, unsigned int i) { return ((float*)ray)[4*N+i]; }
        // RTC_FORCEINLINE float& RTCRayN_dir_y(RTCRayN* ray, unsigned int N, unsigned int i) { return ((float*)ray)[5*N+i]; }
        // RTC_FORCEINLINE float& RTCRayN_dir_z(RTCRayN* ray, unsigned int N, unsigned int i) { return ((float*)ray)[6*N+i]; }
        // RTC_FORCEINLINE float& RTCRayN_time (RTCRayN* ray, unsigned int N, unsigned int i) { return ((float*)ray)[7*N+i]; }

        // RTC_FORCEINLINE float&        RTCRayN_tfar (RTCRayN* ray, unsigned int N, unsigned int i) { return ((float*)ray)[8*N+i]; }
        // RTC_FORCEINLINE unsigned int& RTCRayN_mask (RTCRayN* ray, unsigned int N, unsigned int i) { return ((unsigned*)ray)[9*N+i]; }
        // RTC_FORCEINLINE unsigned int& RTCRayN_id   (RTCRayN* ray, unsigned int N, unsigned int i) { return ((unsigned*)ray)[10*N+i]; }
        // RTC_FORCEINLINE unsigned int& RTCRayN_flags(RTCRayN* ray, unsigned int N, unsigned int i) { return ((unsigned*)ray)[11*N+i]; }
        /* Helper functions to access hit packets of runtime size N */
        // RTC_FORCEINLINE float& RTCHitN_Ng_x(RTCHitN* hit, unsigned int N, unsigned int i) { return ((float*)hit)[0*N+i]; }
        // RTC_FORCEINLINE float& RTCHitN_Ng_y(RTCHitN* hit, unsigned int N, unsigned int i) { return ((float*)hit)[1*N+i]; }
        // RTC_FORCEINLINE float& RTCHitN_Ng_z(RTCHitN* hit, unsigned int N, unsigned int i) { return ((float*)hit)[2*N+i]; }

        // RTC_FORCEINLINE float& RTCHitN_u(RTCHitN* hit, unsigned int N, unsigned int i) { return ((float*)hit)[3*N+i]; }
        // RTC_FORCEINLINE float& RTCHitN_v(RTCHitN* hit, unsigned int N, unsigned int i) { return ((float*)hit)[4*N+i]; }

        // RTC_FORCEINLINE unsigned int& RTCHitN_primID(RTCHitN* hit, unsigned int N, unsigned int i) { return ((unsigned*)hit)[5*N+i]; }
        // RTC_FORCEINLINE unsigned int& RTCHitN_geomID(RTCHitN* hit, unsigned int N, unsigned int i) { return ((unsigned*)hit)[6*N+i]; }
        // RTC_FORCEINLINE unsigned int& RTCHitN_instID(RTCHitN* hit, unsigned int N, unsigned int i, unsigned int l) { return ((unsigned*)hit)[7*N+i+N*l]; }

        // /* Helper functions to extract RTCRayN and RTCHitN from RTCRayHitN */
        // RTC_FORCEINLINE RTCRayN* RTCRayHitN_RayN(RTCRayHitN* rayhit, unsigned int N) { return (RTCRayN*)&((float*)rayhit)[0*N]; }
        // RTC_FORCEINLINE RTCHitN* RTCRayHitN_HitN(RTCRayHitN* rayhit, unsigned int N) { return (RTCHitN*)&((float*)rayhit)[12*N]; }
        unsafe extern "C" fn filter_fn(args: *const sys::RTCFilterFunctionNArguments) {
            let args = &*args;
            if *args.valid == 0 {
                return;
            }
            let ctx = &mut *(args.context as *mut RayQueryContext);
            debug_assert!(args.N == 1);
            let hit = args.hit as *mut u32;
            let prim_id = *hit.add(5);
            let geom_id = *hit.add(6);
            let inst_id = *hit.add(7);
            let hit_u = *(args.hit as *mut f32).add(3);
            let hit_v = *(args.hit as *mut f32).add(4);
            let t_far = &mut *(args.ray as *mut f32).add(8);
            debug_assert!(prim_id != u32::MAX && geom_id != u32::MAX);
            let rq = &mut *ctx.rq;
            rq.ray.tmax = *t_far;
            rq.cur_triangle_hit = defs::TriangleHit {
                prim: prim_id,
                inst: inst_id,
                bary: [hit_u, hit_v],
                committed_ray_t: *t_far,
            };
            let on_triangle_hit = ctx.on_triangle_hit;
            rq.cur_commited = false;
            rq.terminated = false;
            on_triangle_hit(rq);
            if !rq.cur_commited {
                *args.valid = 0;
            } else {
                rq.hit.set_from_triangle_hit(rq.cur_triangle_hit);
            }
            if rq.terminated {
                *t_far = f32::NEG_INFINITY;
            }
        }
        unsafe extern "C" fn occluded_fn(args: *const sys::RTCOccludedFunctionNArguments) {
            let args = &*args;
            if *args.valid == 0 {
                return;
            }
            let ctx = &mut *(args.context as *mut RayQueryContext);
            debug_assert!(args.N == 1);

            let t_near = *(args.ray as *mut f32).add(3);
            let t_far = &mut *(args.ray as *mut f32).add(8);
            let cur_inst_id = (*args.context).instID[0];

            let rq = &mut *ctx.rq;
            rq.cur_procedural_hit = defs::ProceduralHit {
                prim: args.primID,
                inst: cur_inst_id,
            };
            rq.ray.tmax = *t_far;
            rq.cur_commited = false;
            rq.terminated = false;
            (ctx.on_procedural_hit)(rq);
            if !rq.cur_commited {
                *args.valid = 0;
            } else {
                // eprintln!("accepting hit");
                if rq.cur_committed_ray_t >= *t_far || rq.cur_committed_ray_t < t_near {
                    *args.valid = 0;
                    return;
                }
                rq.hit
                    .set_from_procedural_hit(rq.cur_procedural_hit, rq.cur_committed_ray_t);
                *t_far = rq.hit.committed_ray_t;
            }
            if rq.terminated {
                *t_far = f32::NEG_INFINITY;
            }
        }
        unsafe extern "C" fn intersect_fn(args: *const sys::RTCIntersectFunctionNArguments) {
            let args = &*args;
            if *args.valid == 0 {
                return;
            }
            let ctx = &mut *(args.context as *mut RayQueryContext);
            debug_assert!(args.N == 1);
            let hit = (args.rayhit as *mut f32).add(12) as *mut u32;
            let prim_id = &mut *hit.add(5);
            // let geom_id = &mut *hit.add(6);
            let inst_id = &mut *hit.add(7);
            let t_near = *(args.rayhit as *mut f32).add(3);
            let t_far = &mut *(args.rayhit as *mut f32).add(8);
            let cur_inst_id = (*args.context).instID[0];
            let rq = &mut *ctx.rq;
            rq.ray.tmax = *t_far;
            rq.cur_procedural_hit = defs::ProceduralHit {
                prim: args.primID,
                inst: cur_inst_id,
            };

            rq.cur_commited = false;
            rq.terminated = false;
            (ctx.on_procedural_hit)(rq);
            if !rq.cur_commited {
                *args.valid = 0;
            } else {
                // eprintln!("accepting hit");
                if rq.cur_committed_ray_t >= *t_far || rq.cur_committed_ray_t < t_near {
                    *args.valid = 0;
                    return;
                }
                rq.hit
                    .set_from_procedural_hit(rq.cur_procedural_hit, rq.cur_committed_ray_t);
                *prim_id = rq.hit.prim;
                *inst_id = rq.hit.inst;
                *t_far = rq.hit.committed_ray_t;
            }
            if rq.terminated {
                *t_far = f32::NEG_INFINITY;
            }
        }
        if rq.terminate_on_first {
            let mut args = sys::RTCOccludedArguments {
                flags: sys::RTC_RAY_QUERY_FLAG_INCOHERENT
                    | sys::RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER,
                feature_mask: sys::RTC_FEATURE_FLAG_ALL,
                filter: Some(filter_fn),
                occluded: Some(occluded_fn),
                context: &mut ctx.parent as *mut _,
            };
            sys::rtcOccluded1(self.handle, &mut ray as *mut _, &mut args as *mut _);
        } else {
            let mut rayhit = sys::RTCRayHit {
                ray,
                hit: sys::RTCHit {
                    Ng_x: 0.0,
                    Ng_y: 0.0,
                    Ng_z: 0.0,
                    u: 0.0,
                    v: 0.0,
                    primID: u32::MAX,
                    geomID: u32::MAX,
                    instID: [u32::MAX],
                },
            };
            let mut args = sys::RTCIntersectArguments {
                flags: sys::RTC_RAY_QUERY_FLAG_INCOHERENT
                    | sys::RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER,
                feature_mask: sys::RTC_FEATURE_FLAG_ALL,
                filter: Some(filter_fn),
                intersect: Some(intersect_fn),
                context: &mut ctx.parent as *mut _,
            };

            sys::rtcIntersect1(self.handle, &mut rayhit as *mut _, &mut args as *mut _);
        }
    }
}

impl Drop for AccelImpl {
    fn drop(&mut self) {
        unsafe {
            sys::rtcReleaseScene(self.handle);
        }
    }
}
