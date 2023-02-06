use luisa_compute_api_types::{PixelFormat, PixelStorage};
const BLOCK_SIZE: usize = 4;
pub struct TextureImpl {
    data: *mut u32,
    data_size: usize,
    size: [usize; 3],
    blocks: [usize; 3],
    is_two_dimensional: bool,
    pixel_u32_count: usize,
}
unsafe impl Send for TextureImpl {}
unsafe impl Sync for TextureImpl {}
impl TextureImpl {
    pub(super) fn new(size: [usize; 3], storage: PixelStorage) -> Self {
        let pixel_size = storage.size();
        let pixel_u32_count = (pixel_size + 3) / 4;
        let is_two_dimensional = size[2] == 1;
        let blocks = [
            (size[0] + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (size[1] + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (size[2] + BLOCK_SIZE - 1) / BLOCK_SIZE,
        ];
        let data_size = if is_two_dimensional {
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
        let data = unsafe {
            std::alloc::alloc(std::alloc::Layout::from_size_align(data_size * 4, 16).unwrap())
                as *mut u32
        };
        Self {
            data,
            data_size,
            size,
            blocks,
            is_two_dimensional,
            pixel_u32_count,
        }
    }
    fn read_pixel(&self, x: u32, y: u32, z: u32) -> *mut u32 {
        if self.is_two_dimensional {
            assert_eq!(z, 0);
        }
        let block_x = x / BLOCK_SIZE as u32;
        let block_y = y / BLOCK_SIZE as u32;
        let block_z = z / BLOCK_SIZE as u32;
        let offset_x = x % BLOCK_SIZE as u32;
        let offset_y = y % BLOCK_SIZE as u32;
        let offset_z = z % BLOCK_SIZE as u32;
        let ptr = self.data;
        let linear_block_id = block_z * self.blocks[0] as u32 * self.blocks[1] as u32
            + block_y * self.blocks[0] as u32
            + block_x;
        let linear_offset_id = offset_z * BLOCK_SIZE as u32 * BLOCK_SIZE as u32 * BLOCK_SIZE as u32
            + offset_y * BLOCK_SIZE as u32 * BLOCK_SIZE as u32
            + offset_x;
        let block_stride = if self.is_two_dimensional {
            BLOCK_SIZE * BLOCK_SIZE * self.pixel_u32_count
        } else {
            BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE * self.pixel_u32_count
        };
        assert!(linear_offset_id < block_stride as u32);
        let offset = block_stride * linear_block_id as usize
            + linear_offset_id as usize * self.pixel_u32_count;
        unsafe { ptr.add(offset) }
    }
}
impl Drop for TextureImpl {
    fn drop(&mut self) {
        unsafe {
            std::alloc::dealloc(
                self.data as *mut u8,
                std::alloc::Layout::from_size_align(self.data_size * 4, 16).unwrap(),
            )
        }
    }
}
extern "C" fn texture_read_pixel(texture: *mut TextureImpl, x: u32, y: u32, z: u32) -> *mut u32 {
    unsafe { (*texture).read_pixel(x, y, z) }
}
