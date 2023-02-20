use luisa_compute_api_types::{PixelFormat, PixelStorage};
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
}
