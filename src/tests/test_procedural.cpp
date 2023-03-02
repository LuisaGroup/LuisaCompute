#include <luisa-compute.h>
#include <dsl/sugar.h>
#include <gui/backup/window.h>
#include <iostream>

using namespace luisa;
using namespace luisa::compute;


float lcg(uint& state) noexcept 
{
	constexpr auto lcg_a = 1664525u;
	constexpr auto lcg_c = 1013904223u;
	state = lcg_a * state + lcg_c;
	return cast<float>(state & 0x00ffffffu) *
		(1.0f / static_cast<float>(0x01000000u));
};

int main(int argc, char* argv[])
{
	Context context{ argv[0] };
	auto device = context.create_device("dx");
	auto stream = device.create_stream();
	auto device_image1 = device.create_image<float>(PixelStorage::BYTE4, 1280, 720, 0u);
	std::vector<std::array<uint8_t, 4u>> d{ 1280 * 720 };
	int count = 1024;
	float radius = .2f;
	std::vector<AABB> aabbs{ size_t(count) };
	uint state = 0;
	for (int i = 0; i < count; i++)
	{
		auto pos = make_float3(lcg(state) * 2.f - 1.f, lcg(state) * 2.f - 1.f, lcg(state) * 2.f - 1.f) * 10.f;
		auto max = pos + radius;
		auto min = pos - radius;
		aabbs[i].packed_max = { max.x,max.y,max.z };
		aabbs[i].packed_min = { min.x,min.y,min.z };
	}

	auto accel = device.create_accel();
	auto aabbBuffer = device.create_buffer<AABB>(count);
	stream << aabbBuffer.copy_from(aabbs.data()) << synchronize();
	auto proceduralPrimitives = device.create_procedural_primitive(aabbBuffer.view());
	stream<<proceduralPrimitives.build()<<synchronize();
	accel.emplace_back(proceduralPrimitives);
	stream<<accel.build()<<synchronize();

	Kernel2D kernel = [&](Float3 pos)
	{
		Var coord = dispatch_id().xy();
		device_image1->write(coord, make_float4(0.f, 0.f, 0.f, 1.f));
		auto frame_size = 1280.f;
		auto p = (make_float2(coord)) / frame_size * 2.f - 1.f;
		static constexpr auto fov = radians(60.8f);
		auto origin = pos;
		auto pixel = origin + make_float3(p * tan(0.5f * fov), -1.0f);
		auto direction = normalize(pixel - origin);
		auto ray = make_ray(origin, direction);

		auto q = accel->trace_all(ray);
		auto hit = q.proceed([&](Var<Hit>&& h)
		{
			commit_triangle();
		},
		[&](Var<Hit>&& h)
		{
			auto aabb = aabbBuffer->read(h.prim);

			//ray-sphere intersection
			auto origin = (aabb->min()+aabb->max())*.5f;
			auto rayOrigin = ray->origin();
			auto L = origin - rayOrigin;
			auto cosTheta = dot(ray->direction(), normalize(L));
			$if(cosTheta > 0.f)
			{
				auto d_oc = length(L);
				auto tc = d_oc * cosTheta;
				auto d = sqrt(d_oc * d_oc - tc * tc);
				$if(d <= radius)
				{
					auto t1c = sqrt(radius * radius - d * d);
					commit_primitive(tc - t1c);
				};
			};
		});
		$if(!hit->miss())
		{
			device_image1->write(coord, make_float4(make_float3(1.f/log(hit->committed_ray_t)), 1.f));
		};

	};
	auto s = device.compile(kernel);

	Window window{ "Display", uint2{ 1280,720 } };
	float3 pos = make_float3(0.f, 1.f, 18.0f);
	window.set_key_callback([&](int key, int action) noexcept
	{
		if (action == GLFW_PRESS && key == GLFW_KEY_ESCAPE)
		{
			window.set_should_close();
		}
		if ((action == GLFW_REPEAT || action == GLFW_PRESS) && key == GLFW_KEY_W)
		{
			pos.z -= 1.f;
		}
		if ((action == GLFW_REPEAT || action == GLFW_PRESS) && key == GLFW_KEY_S)
		{
			pos.z += 1.f;
		}
		if ((action == GLFW_REPEAT || action == GLFW_PRESS) && key == GLFW_KEY_Q)
		{
			pos.y += 1.f;
		}
		if ((action == GLFW_REPEAT || action == GLFW_PRESS) && key == GLFW_KEY_E)
		{
			pos.y -= 1.f;
		}
	});
	auto frame_count = 0u;
	window.run([&]
	{
		stream
			<< s(pos).dispatch(1280, 720)
			<< device_image1.copy_to(d.data())
			<< synchronize();
		window.set_background(d.data(), uint2{ 1280,720 });
	});
}