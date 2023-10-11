use luisa_compute_api_types::PixelStorage;

use rayon::prelude::{IntoParallelIterator, ParallelIterator};

const BLOCK_SIZE: usize = 4;

pub struct TextureImpl {
    pub(crate) data: *mut u8,
    pub(crate) data_size: usize,
    pub(crate) size: [u32; 3],
    pub(crate) dimension: u8,
    pub(crate) pixel_stride_shift: usize,
    pub(crate) mip_levels: u8,
    pub(crate) mip_offsets: [usize; 16],
    pub(crate) storage: PixelStorage,
    layout: std::alloc::Layout,
}

unsafe impl Send for TextureImpl {}

unsafe impl Sync for TextureImpl {}

impl Drop for TextureImpl {
    fn drop(&mut self) {
        unsafe {
            std::alloc::dealloc(self.data, self.layout);
        }
    }
}

impl TextureImpl {
    pub(super) fn new(
        dimension: u8,
        size: [u32; 3],
        storage: PixelStorage,
        levels: u8,
        _allow_simultaneous_access: bool,
    ) -> Self {
        let pixel_size = storage.size();
        let pixel_stride_shift = match pixel_size {
            1 => 0,
            2 => 1,
            4 => 2,
            8 => 3,
            16 => 4,
            _ => unreachable!(),
        };
        if dimension == 2 {
            assert_eq!(size[2], 1);
        }
        let mut data_size = 0;
        let mut mip_offsets = [0; 16];
        for level in 0..levels {
            mip_offsets[level as usize] = data_size;
            let blocks = [
                (((size[0] as usize >> level).max(1)) + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (((size[1] as usize >> level).max(1)) + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (((size[2] as usize >> level).max(1)) + BLOCK_SIZE - 1) / BLOCK_SIZE,
            ];
            data_size += if dimension == 2 {
                blocks[0] * blocks[1] * blocks[2] * BLOCK_SIZE * BLOCK_SIZE * pixel_size
            } else {
                blocks[0]
                    * blocks[1]
                    * blocks[2]
                    * BLOCK_SIZE
                    * BLOCK_SIZE
                    * BLOCK_SIZE
                    * pixel_size
            };
        }
        for level in levels..16 {
            mip_offsets[level as usize] = data_size;
        }
        let layout = std::alloc::Layout::from_size_align(data_size, 16).unwrap();
        let data = unsafe { std::alloc::alloc(layout) };
        Self {
            data,
            data_size,
            size,
            dimension,
            pixel_stride_shift,
            mip_levels: levels,
            mip_offsets,
            storage,
            layout,
        }
    }
    pub(crate) fn view(&self, level: u8) -> TextureView {
        let offset = self.mip_offsets[level as usize];
        assert!(offset <= self.data_size);
        let size = [
            (self.size[0] >> level).max(1),
            (self.size[1] >> level).max(1),
            (self.size[2] >> level).max(1),
        ];
        unsafe {
            TextureView {
                data: self.data.add(offset) as *mut u8,
                size,
                pixel_stride_shift: self.pixel_stride_shift,
                data_size: if level == 15 {
                    self.data_size - offset
                } else {
                    self.mip_offsets[level as usize + 1] - offset
                },
            }
        }
    }
    pub(crate) fn into_c_texture(
        &self,
        sampler: luisa_compute_api_types::Sampler,
    ) -> luisa_compute_cpu_kernel_defs::Texture {
        luisa_compute_cpu_kernel_defs::Texture {
            sampler: sampler.encode(),
            data: self.data,
            width: self.size[0],
            height: self.size[1],
            depth: self.size[2],
            storage: self.storage as u8,
            dimension: self.dimension,
            mip_levels: self.mip_levels,
            pixel_stride_shift: self.pixel_stride_shift as u8,
            mip_offsets: self.mip_offsets,
        }
    }
}

#[derive(Debug)]
pub(crate) struct TextureView {
    pub(crate) data: *mut u8,
    pub(crate) size: [u32; 3],
    pub(crate) pixel_stride_shift: usize,
    pub(crate) data_size: usize,
}

unsafe impl Send for TextureView {}

unsafe impl Sync for TextureView {}

impl TextureView {
    pub(crate) fn unpadded_data_size(&self) -> usize {
        self.size[0] as usize
            * self.size[1] as usize
            * self.size[2] as usize
            * (1 << self.pixel_stride_shift)
    }
    #[inline]
    pub(crate) fn get_pixel_2d(&self, x: u32, y: u32) -> *mut u8 {
        let block_x = x / BLOCK_SIZE as u32;
        let block_y = y / BLOCK_SIZE as u32;
        let grid_width = (self.size[0] + BLOCK_SIZE as u32 - 1) / BLOCK_SIZE as u32;
        let block_idx = block_x + block_y * grid_width;
        let pixel_x = x % BLOCK_SIZE as u32;
        let pixel_y = y % BLOCK_SIZE as u32;
        let pixel_idx =
            block_idx * (BLOCK_SIZE * BLOCK_SIZE) as u32 + pixel_x + pixel_y * BLOCK_SIZE as u32;
        let i = (pixel_idx as usize) << self.pixel_stride_shift;
        assert!(i <= self.data_size);
        unsafe { self.data.add(i) }
    }
    #[inline]
    pub(crate) fn get_pixel_3d(&self, x: u32, y: u32, z: u32) -> *mut u8 {
        let block_x = x / BLOCK_SIZE as u32;
        let block_y = y / BLOCK_SIZE as u32;
        let block_z = z / BLOCK_SIZE as u32;
        let grid_width = (self.size[0] + BLOCK_SIZE as u32 - 1) / BLOCK_SIZE as u32;
        let grid_height = (self.size[1] + BLOCK_SIZE as u32 - 1) / BLOCK_SIZE as u32;
        let block_idx = block_x + block_y * grid_width + block_z * grid_width * grid_height;
        let pixel_x = x % BLOCK_SIZE as u32;
        let pixel_y = y % BLOCK_SIZE as u32;
        let pixel_z = z % BLOCK_SIZE as u32;
        let pixel_idx = block_idx * (BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE) as u32
            + pixel_x
            + pixel_y * BLOCK_SIZE as u32
            + pixel_z * BLOCK_SIZE as u32 * BLOCK_SIZE as u32;
        let i = (pixel_idx as usize) << self.pixel_stride_shift;
        assert!(i <= self.data_size);
        unsafe { self.data.add(i) }
    }
    #[inline]
    pub(crate) fn copy_to_vec_par_2d(&self) -> Vec<u8> {
        // dbg!(self);
        unsafe {
            let mut data: Vec<u8> = Vec::with_capacity(self.unpadded_data_size());
            let data_ptr = data.as_mut_ptr() as u64;
            (0..(self.size[0] * self.size[1]))
                .into_par_iter()
                .for_each(|i| {
                    let (x, y) = (i % self.size[0], i / self.size[0]);
                    let data = data_ptr as *mut u8;
                    let dst = self.get_pixel_2d(x, y);
                    std::ptr::copy_nonoverlapping(
                        dst,
                        data.add(i as usize * (1 << self.pixel_stride_shift)),
                        1 << self.pixel_stride_shift,
                    );
                });
            data.set_len(self.unpadded_data_size());
            data
        }
    }
    #[inline]
    pub(crate) fn copy_from_2d(&self, mut data: *const u8) {
        for y in 0..self.size[1] {
            for x in 0..self.size[0] {
                let dst = self.get_pixel_2d(x, y);
                unsafe {
                    std::ptr::copy_nonoverlapping(data, dst, 1 << self.pixel_stride_shift);
                    data = data.add(1 << self.pixel_stride_shift);
                }
            }
        }
    }
    #[inline]
    pub(crate) fn copy_from_3d(&self, mut data: *const u8) {
        for z in 0..self.size[2] {
            for y in 0..self.size[1] {
                for x in 0..self.size[0] {
                    let dst = self.get_pixel_3d(x, y, z);
                    unsafe {
                        std::ptr::copy_nonoverlapping(data, dst, 1 << self.pixel_stride_shift);
                        data = data.add(1 << self.pixel_stride_shift);
                    }
                }
            }
        }
    }
    #[inline]
    pub(crate) fn copy_to_2d(&self, mut data: *mut u8) {
        for y in 0..self.size[1] {
            for x in 0..self.size[0] {
                let src = self.get_pixel_2d(x, y);
                unsafe {
                    std::ptr::copy_nonoverlapping(src, data, 1 << self.pixel_stride_shift);
                    data = data.add(1 << self.pixel_stride_shift);
                }
            }
        }
    }
    #[inline]
    pub(crate) fn copy_to_vec_2d(&self) -> Vec<u8> {
        let mut data: Vec<u8> = Vec::with_capacity(self.unpadded_data_size());
        unsafe {
            data.set_len(self.unpadded_data_size());
            self.copy_to_2d(data.as_mut_ptr());
        }
        data
    }
    #[inline]
    pub(crate) fn copy_to_3d(&self, mut data: *mut u8) {
        for z in 0..self.size[2] {
            for y in 0..self.size[1] {
                for x in 0..self.size[0] {
                    let src = self.get_pixel_3d(x, y, z);
                    unsafe {
                        std::ptr::copy_nonoverlapping(src, data, 1 << self.pixel_stride_shift);
                        data = data.add(1 << self.pixel_stride_shift);
                    }
                }
            }
        }
    }
}
