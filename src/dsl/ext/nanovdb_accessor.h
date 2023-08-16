#include <luisa/luisa-compute.h>// FIXME: do not include the universal header in public headers

/*
	From NanoVDB.h

	Memory layout:

	Notation: "]---[" implies it has optional padding, and "][" implies zero padding

	[GridData(672B)][TreeData(64B)]---[RootData][N x Root::Tile]---[NodeData<5>]---[ModeData<4>]---[LeafData<3>]---[BLINDMETA...]---[BLIND0]---[BLIND1]---etc.
	^                                 ^         ^                  ^               ^               ^
	|                                 |         |                  |               |               |
	+-- Start of 32B aligned buffer   |         |                  |               |               +-- Node0::DataType* leafData
		GridType::DataType* gridData  |         |                  |               |
									  |         |                  |               +-- Node1::DataType* lowerData
	   RootType::DataType* rootData --+         |                  |
												|                  +-- Node2::DataType* upperData
												|
												+-- RootType::DataType::Tile* tile


	It's important to emphasize that all the grid data (defined below) are explicitly 32 byte
	aligned, which implies that any memory buffer that contains a NanoVDB grid must also be at
	32 byte aligned. That is, the memory address of the beginning of a buffer (see ascii diagram below)
	must be divisible by 32, i.e. uintptr_t(&buffer)%32 == 0! If this is not the case, the C++ standard
	says the behaviour is undefined! Normally this is not a concerns on GPUs, because they use 256 byte
	aligned allocations, but the same cannot be said about the CPU.

	GridData is always at the very beginning of the buffer immediately followed by TreeData!
	The remaining nodes and blind-data are allowed to be scattered throughout the buffer,
	though in practice they are arranged as:

	GridData: 672 bytes (e.g. magic, checksum, major, flags, index, count, size, name, map, world bbox, voxel size, class, type, offset, count)

	TreeData: 64 bytes (node counts and byte offsets)

	... optional padding ...

	RootData: size depends on ValueType (index bbox, voxel count, tile count, min/max/avg/standard deviation)

	Array of: RootData::Tile

	... optional padding ...

	Array of: Upper InternalNodes of size 32^3:  bbox, two bit masks, 32768 tile values, and min/max/avg/standard deviation values

	... optional padding ...

	Array of: Lower InternalNodes of size 16^3:  bbox, two bit masks, 4096 tile values, and min/max/avg/standard deviation values

	... optional padding ...

	Array of: LeafNodes of size 8^3: bbox, bit masks, 512 voxel values, and min/max/avg/standard deviation values
*/
namespace luisa
{
	namespace compute
	{
		class NanovdbAccessor
		{

		protected:

			UInt2 coord_to_key(Int3 coord)
			{
				auto iu = cast<uint>(coord[0]) >> 12u;
				auto ju = cast<uint>(coord[1]) >> 12u;
				auto ku = cast<uint>(coord[2]) >> 12u;
				return make_uint2(ku | (ju << 21), (iu << 10) | (ju >> 11));
			}
			UInt upper_coord_to_offset(Int3 c)
			{
				return (((c[0] & 4095) >> 7) << (2 * 5)) +
					(((c[1] & 4095) >> 7) << (5)) +
					((c[2] & 4095) >> 7);
			}
			Bool upper_child_mask(UInt child_offset, UInt n)
			{
				auto b = buffer.read<luisa::uint2>(child_offset + 32 + 4096 + (n >> 6) * 8);
				auto compareBit = n & 63;
				return ite(compareBit > 31, (b.y & (1U << (n & compareBit - 32))) > 0, (b.x & (1U << (n & compareBit))) > 0);
			}
			Float upper_get_value(UInt child_offset, UInt n)
			{
				return buffer.read<float>(child_offset + 8256 + n * 8);
			}

			UInt lower_coord_to_offset(Int3 c)
			{
				return (((c[0] & 127) >> 3) << (2 * 4)) +
					(((c[1] & 127) >> 3) << (4)) +
					((c[2] & 127) >> 3);
			}
			Bool lower_child_mask(UInt child_offset, UInt n)
			{
				auto b = buffer.read<luisa::uint2>(child_offset + 32 + 512 + (n >> 6) * 8);
				auto compareBit = n & 63;
				return ite(compareBit > 31, (b.y & (1U << (n & compareBit - 32))) > 0, (b.x & (1U << (n & compareBit))) > 0);
			}
			Float lower_get_value(UInt child_offset, UInt n)
			{
				return buffer.read<float>(child_offset + 1088 + n * 8);
			}

			UInt leaf_coord_to_offset(Int3 c)
			{
				return (((c[0] & 7) >> 0) << (2 * 3)) +
					(((c[1] & 7) >> 0) << (3)) +
					((c[2] & 7) >> 0);
			}
			Float leaf_get_value(UInt child_offset, UInt n)
			{
				//offset by 96 bytes
				return buffer.read<float>(child_offset + 96 + 4 * n);
			}
			uint tableSize = 8;
			const detail::BindlessByteAddressBuffer buffer;
		public:
			NanovdbAccessor(const detail::BindlessByteAddressBuffer buffer, uint tableSize)
				:buffer{ buffer }, tableSize{ tableSize }
			{
			}

			void get_bounds(Int3 &lower, Int3 &upper)
			{
				auto b0 = buffer->read<luisa::int2>(736);
				auto b1 = buffer->read<luisa::int2>(736 + 8);
				auto b2 = buffer->read<luisa::int2>(736 + 16);
				lower = make_int3(b0.x, b0.y, b1.x);
				upper = make_int3(b1.y, b2.x, b2.y);
			}
			Float get_value(Int3 coord)
			{
				UInt offset = 736 + 64;//tile offset
				Int tileId = -1;
				auto key = coord_to_key(coord);
				$for(i, tableSize)
				{
					auto o = offset + 32 * i;
					auto tileKey = buffer.read<uint2>(o);
					$if(all(tileKey == key))
					{
						tileId = i;
						$break;
					};
				};
				Float value = 0.f;
				$if(tileId > -1)
				{
					auto child_offset = buffer.read<luisa::uint2>(offset + 32 * tileId + 8).x;//lower bits of ulong
					$if(child_offset > 0)
					{
						offset = 736 + child_offset;//offset from root
						auto n = upper_coord_to_offset(coord);
						$if(upper_child_mask(offset, n))
						{
							child_offset = buffer.read<luisa::uint2>(offset + 8256 + 8 * n).x;
							offset += child_offset;

							n = lower_coord_to_offset(coord);
							$if(lower_child_mask(offset, n))
							{
								child_offset = buffer.read<luisa::uint2>(offset + 1088 + 8 * n).x;
								offset += child_offset;
								n = leaf_coord_to_offset(coord);
								value = leaf_get_value(offset, n);
							}
							$else
							{
								value = lower_get_value(offset,n);
							};
						}
						$else
						{
							value = upper_get_value(offset, n);
						};
					}
					$else
					{
						value = buffer.read<float>(offset + 32 * tileId + 20);
					};
				};
				return value;
			}

			//Trilinear interpolated sample
			Float get_value(Float3 coord)
			{
				auto posi = make_int3((Int)coord[0], (Int)coord[1], (Int)coord[2]);
				auto d = coord - make_float3(Float(posi.x), Float(posi.y), Float(posi.z));
				ArrayFloat<8> results{};
				$for(i, 8)
				{
					results[i] = get_value(posi + make_int3(i % 2, (i >> 1) % 2, (i >> 2) % 2));
				};
				auto d00 = lerp(results[0], results[1], d.x);
				auto d10 = lerp(results[2], results[3], d.x);
				auto d01 = lerp(results[4], results[5], d.x);
				auto d11 = lerp(results[6], results[7], d.x);
				auto d0 = lerp(d00, d10, d.y);
				auto d1 = lerp(d01, d11, d.y);
				return lerp(d0, d1, d.z);
			}
		};
	}
}
