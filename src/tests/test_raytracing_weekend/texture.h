#pragma once

#include "rtweekend.h"
#include "perlin.h"

class texture {
public:
    virtual Float3 value(Float u, Float v, const Float3 &p) const = 0;
};

class solid_color : public texture {
public:
    solid_color() {}
    solid_color(float3 c) : color_value(c) {}

    solid_color(float red, float green, float blue)
        : solid_color(float3(red, green, blue)) {}

    virtual Float3 value(Float u, Float v, const Float3 &p) const override {
        return color_value;
    }

private:
    float3 color_value;
};

class checker_texture : public texture {
public:
    checker_texture() {}

    checker_texture(shared_ptr<texture> _even, shared_ptr<texture> _odd)
        : even(_even), odd(_odd) {}

    checker_texture(float3 c1, float3 c2)
        : even(make_shared<solid_color>(c1)), odd(make_shared<solid_color>(c2)) {}

    virtual Float3 value(Float u, Float v, const Float3 &p) const override {
        Float3 ret;

        auto sines = sin(10 * p.x) * sin(10 * p.y) * sin(10 * p.z);
        $if (sines < 0) {
            ret = odd->value(u, v, p);
        }
        $else {
            ret = even->value(u, v, p);
        };

        return ret;
    }

public:
    shared_ptr<texture> odd;
    shared_ptr<texture> even;
};

class noise_texture : public texture {
public:
    noise_texture(Device &d, Stream &s) : noise(perlin(d, s)) {}
    noise_texture(Device &d, Stream &s, float sc) : noise(perlin(d, s)), scale(sc) {}

    virtual Float3 value(Float u, Float v, const Float3 &p) const override {
        return Float3(1, 1, 1) * 0.5f * (1 + sin(scale * p.z + 10 * noise.turb(p)));
    }

public:
    perlin noise;
    float scale;
};

class image_texture : public texture {
public:
    const static int bytes_per_pixel = 4;

    image_texture()
        : data(nullptr), width(0), height(0), bytes_per_scanline(0) {}

    image_texture(Device &device, Stream &stream, const char *filename) {
        auto components_per_pixel = bytes_per_pixel;

        data = stbi_load(
            filename, &width, &height, &components_per_pixel, components_per_pixel);

        if (!data) {
            LUISA_ERROR("ERROR: Could not load texture image file '{}'.\n", filename);
            width = height = 0;
        }
        int pos = 1024 * 256 + 512;

        bytes_per_scanline = bytes_per_pixel * width;

        dataBuf = device.create_buffer<int>(width * height);
        stream << dataBuf.copy_from(data) << synchronize();
    }

    ~image_texture() {
        delete data;
    }

    virtual Float3 value(Float u, Float v, const Float3 &p) const override {
        // If we have no texture data, then return solid cyan as a debugging aid.
        if (data == nullptr)
            return Float3(0, 1, 1);

        // Clamp input texture coordinates to [0,1] x [1,0]
        u = clamp(u, 0.0f, 1.0f);
        v = 1.0f - clamp(v, 0.0f, 1.0f);// Flip V to image coordinates

        auto i = static_cast<Int>(u * width);
        auto j = static_cast<Int>(v * height);

        // Clamp integer mapping, since actual coordinates should be less than 1.0
        $if (i >= width) { i = width - 1; };
        $if (j >= height) { j = height - 1; };

        const Float color_scale = 1.0f / 255.0f;
        Int pixel = dataBuf->read(j * width + i);
        Int pixel_0 = pixel & 255;
        Int pixel_1 = (pixel >> 8) & 255;
        Int pixel_2 = (pixel >> 16) & 255;

        return Float3(color_scale * pixel_0, color_scale * pixel_1, color_scale * pixel_2);
    }

private:
    unsigned char *data;
    int width, height;
    int bytes_per_scanline;

    Buffer<int> dataBuf;
};