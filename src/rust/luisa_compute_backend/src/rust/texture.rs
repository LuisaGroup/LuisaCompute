use luisa_compute_api_types::{PixelFormat, PixelStorage};
use luisa_compute_cpu_kernel_defs as defs;
const BLOCK_SIZE: usize = 4;
pub struct TextureImpl {
    data: Vec<u32>,
    size: [u32; 3],
    dimension: u8,
    pixel_u32_count: usize,
    mip_levels: u8,
    mip_offsets: [usize; 16],
    storage: PixelStorage,
}
unsafe impl Send for TextureImpl {}
unsafe impl Sync for TextureImpl {}
impl TextureImpl {
    pub(super) fn new(dimension: u8, size: [u32; 3], storage: PixelStorage, levels: u8) -> Self {
        let pixel_size = storage.size();
        let pixel_u32_count = (pixel_size + 3) / 4;
        if dimension == 2 {
            assert_eq!(size[2], 1);
        }
        let mut data_size = 0;
        let mut data = Vec::new();
        let mut mip_offsets = [0; 16];
        for level in 0..levels {
            let blocks = [
                ((size[0] as usize >> level) + BLOCK_SIZE - 1) / BLOCK_SIZE,
                ((size[1] as usize >> level) + BLOCK_SIZE - 1) / BLOCK_SIZE,
                ((size[2] as usize >> level) + BLOCK_SIZE - 1) / BLOCK_SIZE,
            ];
            data_size += if dimension == 2 {
                blocks[0] * blocks[1] * blocks[2] * BLOCK_SIZE * BLOCK_SIZE * pixel_u32_count
            } else {
                blocks[0]
                    * blocks[1]
                    * blocks[2]
                    * BLOCK_SIZE
                    * BLOCK_SIZE
                    * BLOCK_SIZE
                    * pixel_u32_count
            };
            mip_offsets[level as usize] = data_size;
        }
        data.resize((data_size + 3) / 4, 0);

        Self {
            data,
            size,
            dimension,
            pixel_u32_count,
            mip_levels: levels,
            mip_offsets,
            storage,
        }
    }
    pub(crate) fn view(&self, level: u8) -> TextureView {
        let offset = self.mip_offsets[level as usize];
        let size = [
            self.size[0] >> level,
            self.size[1] >> level,
            self.size[2] >> level,
        ];
        unsafe {
            TextureView {
                data: self.data.as_ptr().add(offset) as *mut u32,
                size,
                pixel_u32_count: self.pixel_u32_count,
            }
        }
    }
}
pub(crate) struct TextureView {
    data: *mut u32,
    size: [u32; 3],
    pixel_u32_count: usize,
}

impl TextureView {
    #[inline]
    pub(crate) fn get_pixel_2d(&self, x: u32, y: u32) -> *mut u32 {
        let block_x = x / BLOCK_SIZE as u32;
        let block_y = y / BLOCK_SIZE as u32;
        let grid_width = (self.size[0] + BLOCK_SIZE as u32 - 1) / BLOCK_SIZE as u32;
        let block_idx = block_x + block_y * grid_width;
        let pixel_x = x % BLOCK_SIZE as u32;
        let pixel_y = y % BLOCK_SIZE as u32;
        let pixel_idx =
            block_idx * (BLOCK_SIZE * BLOCK_SIZE) as u32 + pixel_x + pixel_y * BLOCK_SIZE as u32;
        unsafe { self.data.add(pixel_idx as usize * self.pixel_u32_count) }
    }
    #[inline]
    pub(crate) fn get_pixel_3d(&self, x: u32, y: u32, z: u32) -> *mut u32 {
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
        unsafe { self.data.add(pixel_idx as usize * self.pixel_u32_count) }
    }
    #[inline]
    pub(crate) fn copy_from_2d(&self, mut data: *const u32) {
        for y in 0..self.size[1] {
            for x in 0..self.size[0] {
                let dst = self.get_pixel_2d(x, y);
                unsafe {
                    std::ptr::copy_nonoverlapping(data, dst, self.pixel_u32_count);
                    data = data.add(self.pixel_u32_count);
                }
            }
        }
    }
    #[inline]
    pub(crate) fn copy_from_3d(&self, mut data: *const u32) {
        for z in 0..self.size[2] {
            for y in 0..self.size[1] {
                for x in 0..self.size[0] {
                    let dst = self.get_pixel_3d(x, y, z);
                    unsafe {
                        std::ptr::copy_nonoverlapping(data, dst, self.pixel_u32_count);
                        data = data.add(self.pixel_u32_count);
                    }
                }
            }
        }
    }
    #[inline]
    pub(crate) fn copy_to_2d(&self, mut data: *mut u32) {
        for y in 0..self.size[1] {
            for x in 0..self.size[0] {
                let src = self.get_pixel_2d(x, y);
                unsafe {
                    std::ptr::copy_nonoverlapping(src, data, self.pixel_u32_count);
                    data = data.add(self.pixel_u32_count);
                }
            }
        }
    }
    #[inline]
    pub(crate) fn copy_to_3d(&self, mut data: *mut u32) {
        for z in 0..self.size[2] {
            for y in 0..self.size[1] {
                for x in 0..self.size[0] {
                    let src = self.get_pixel_3d(x, y, z);
                    unsafe {
                        std::ptr::copy_nonoverlapping(src, data, self.pixel_u32_count);
                        data = data.add(self.pixel_u32_count);
                    }
                }
            }
        }
    }
}
