#pragma once

#include <luisa/dsl/builtin.h>
#include <luisa/dsl/func.h>

namespace luisa::compute::coro::example {

/// The compact sinusoidal neural SDF from Metin Seven's Shadertoy bunny
/// (https://www.shadertoy.com/view/3lyyWc), expressed as an ordinary Luisa DSL
/// callable so it can be dispatched as a coroutine external stage.
[[nodiscard]] inline Callable<float(float3)> make_neural_bunny_sdf() noexcept {
    return [](Float3 p) noexcept {
        auto radius = length(p);
        Float4 f00 = sin(
            p.y * make_float4(-3.02f, 1.95f, -3.42f, -.60f) +
            p.z * make_float4(3.08f, .85f, -2.25f, -.24f) -
            p.x * make_float4(-.29f, 1.16f, -3.74f, 2.89f) +
            make_float4(-.71f, 4.50f, -3.24f, -3.50f));
        Float4 f01 = sin(
            p.y * make_float4(-.40f, -3.61f, 3.23f, -.14f) +
            p.z * make_float4(-.36f, 3.64f, -3.91f, 2.66f) -
            p.x * make_float4(2.90f, -.54f, -2.75f, 2.71f) +
            make_float4(7.02f, -5.41f, -1.12f, -7.41f));
        Float4 f02 = sin(
            p.y * make_float4(-1.77f, -1.28f, -4.29f, -3.20f) +
            p.z * make_float4(-3.49f, -2.81f, -.64f, 2.79f) -
            p.x * make_float4(3.15f, 2.14f, -3.85f, 1.83f) +
            make_float4(-2.07f, 4.49f, 5.33f, -2.17f));
        Float4 f03 = sin(
            p.y * make_float4(-.49f, .68f, 3.05f, .42f) +
            p.z * make_float4(-2.87f, .78f, 3.78f, -3.41f) -
            p.x * make_float4(-2.65f, .33f, .07f, -.64f) +
            make_float4(-3.24f, -5.90f, 1.14f, -4.71f));

        Float4 f10 = sin(
                         def(make_float4x4(
                             -.34f, .06f, -.59f, -.76f, .10f, -.19f,
                             -.12f, .44f, .64f, -.02f, -.26f, .15f,
                             -.16f, .21f, .91f, .15f)) *
                             f00 +
                         def(make_float4x4(
                             .01f, .54f, -.77f, .11f, .06f, -.14f,
                             .43f, .51f, -.18f, .08f, .39f, .20f, .33f,
                             -.49f, -.10f, .19f)) *
                             f01 +
                         def(make_float4x4(
                             .27f, .22f, .43f, .53f, .18f, -.17f,
                             .23f, -.64f, -.14f, .02f, -.10f, .16f,
                             -.13f, -.06f, -.04f, -.36f)) *
                             f02 +
                         def(make_float4x4(
                             -.13f, .29f, -.29f, .08f, 1.13f, .02f,
                             -.83f, .32f, -.32f, .04f, -.31f, -.16f,
                             .14f, -.03f, -.20f, .39f)) *
                             f03 +
                         make_float4(.73f, -4.28f, -1.56f, -1.80f)) +
                     f00;
        Float4 f11 = sin(
                         def(make_float4x4(
                             -1.11f, .55f, -.12f, -1.00f, .16f, .15f,
                             -.30f, .31f, -.01f, .01f, .31f, -.42f,
                             -.29f, .38f, -.04f, .71f)) *
                             f00 +
                         def(make_float4x4(
                             .96f, -.02f, .86f, .52f, -.14f, .60f,
                             .44f, .43f, .02f, -.15f, -.49f, -.05f,
                             -.06f, -.25f, -.03f, -.22f)) *
                             f01 +
                         def(make_float4x4(
                             .52f, .44f, -.05f, -.11f, -.56f, -.10f,
                             -.61f, -.40f, -.04f, .55f, .32f, -.07f,
                             -.02f, .28f, .26f, -.49f)) *
                             f02 +
                         def(make_float4x4(
                             .02f, -.32f, .06f, -.17f, -.59f, .00f,
                             -.24f, .60f, -.06f, .13f, -.21f, -.27f,
                             -.12f, -.14f, .58f, -.55f)) *
                             f03 +
                         make_float4(-2.24f, -3.48f, -.80f, 1.41f)) +
                     f01;
        Float4 f12 = sin(
                         def(make_float4x4(
                             .44f, -.06f, -.79f, -.46f, .05f, -.60f,
                             .30f, .36f, .35f, .12f, .02f, .12f, .40f,
                             -.26f, .63f, -.21f)) *
                             f00 +
                         def(make_float4x4(
                             -.48f, .43f, -.73f, -.40f, .11f, -.01f,
                             .71f, .05f, -.25f, .25f, -.28f, -.20f,
                             .32f, -.02f, -.84f, .16f)) *
                             f01 +
                         def(make_float4x4(
                             .39f, -.07f, .90f, .36f, -.38f, -.27f,
                             -1.86f, -.39f, .48f, -.20f, -.05f, .10f,
                             -.00f, -.21f, .29f, .63f)) *
                             f02 +
                         def(make_float4x4(
                             .46f, -.32f, .06f, .09f, .72f, -.47f,
                             .81f, .78f, .90f, .02f, -.21f, .08f,
                             -.16f, .22f, .32f, -.13f)) *
                             f03 +
                         make_float4(3.38f, 1.20f, .84f, 1.41f)) +
                     f02;
        Float4 f13 = sin(
                         def(make_float4x4(
                             -.41f, -.24f, -.71f, -.25f, -.24f, -.75f,
                             -.09f, .02f, -.27f, -.42f, .02f, .03f,
                             -.01f, .51f, -.12f, -1.24f)) *
                             f00 +
                         def(make_float4x4(
                             .64f, .31f, -1.36f, .61f, -.34f, .11f,
                             .14f, .79f, .22f, -.16f, -.29f, -.70f,
                             .02f, -.37f, .49f, .39f)) *
                             f01 +
                         def(make_float4x4(
                             .79f, .47f, .54f, -.47f, -1.13f, -.35f,
                             -1.03f, -.22f, -.67f, -.26f, .10f, .21f,
                             -.07f, -.73f, -.11f, .72f)) *
                             f02 +
                         def(make_float4x4(
                             .43f, -.23f, .13f, .09f, 1.38f, -.63f,
                             1.57f, -.20f, .39f, -.14f, .42f, .13f,
                             -.57f, -.08f, -.21f, .21f)) *
                             f03 +
                         make_float4(-.34f, -3.28f, .43f, -.52f)) +
                     f03;

        f00 = sin(def(make_float4x4(
                      -.72f, .23f, -.89f, .52f, .38f, .19f, -.16f,
                      -.88f, .26f, -.37f, .09f, .63f, .29f, -.72f,
                      .30f, -.95f)) *
                      f10 +
                  def(make_float4x4(
                      -.22f, -.51f, -.42f, -.73f, -.32f, .00f, -1.03f,
                      1.17f, -.20f, -.03f, -.13f, -.16f, -.41f, .09f,
                      .36f, -.84f)) *
                      f11 +
                  def(make_float4x4(
                      -.21f, .01f, .33f, .47f, .05f, .20f, -.44f,
                      -1.04f, .13f, .12f, -.13f, .31f, .01f, -.34f,
                      .41f, -.34f)) *
                      f12 +
                  def(make_float4x4(
                      -.13f, -.06f, -.39f, -.22f, .48f, .25f, .24f,
                      -.97f, -.34f, .14f, .42f, -.00f, -.44f, .05f,
                      .09f, -.95f)) *
                      f13 +
                  make_float4(.48f, .87f, -.87f, -2.06f)) /
                  1.4f +
              f10;
        f01 = sin(def(make_float4x4(
                      -.27f, .29f, -.21f, .15f, .34f, -.23f, .85f,
                      -.09f, -1.15f, -.24f, -.05f, -.25f, -.12f, -.73f,
                      -.17f, -.37f)) *
                      f10 +
                  def(make_float4x4(
                      -1.11f, .35f, -.93f, -.06f, -.79f, -.03f, -.46f,
                      -.37f, .60f, -.37f, -.14f, .45f, -.03f, -.21f,
                      .02f, .59f)) *
                      f11 +
                  def(make_float4x4(
                      -.92f, -.17f, -.58f, -.18f, .58f, .60f, .83f,
                      -1.04f, -.80f, -.16f, .23f, -.11f, .08f, .16f,
                      .76f, .61f)) *
                      f12 +
                  def(make_float4x4(
                      .29f, .45f, .30f, .39f, -.91f, .66f, -.35f,
                      -.35f, .21f, .16f, -.54f, -.63f, 1.10f, -.38f,
                      .20f, .15f)) *
                      f13 +
                  make_float4(-1.72f, -.14f, 1.92f, 2.08f)) /
                  1.4f +
              f11;
        f02 = sin(def(make_float4x4(
                      1.00f, .66f, 1.30f, -.51f, .88f, .25f, -.67f,
                      .03f, -.68f, -.08f, -.12f, -.14f, .46f, 1.15f,
                      .38f, -.10f)) *
                      f10 +
                  def(make_float4x4(
                      .51f, -.57f, .41f, -.09f, .68f, -.50f, -.04f,
                      -1.01f, .20f, .44f, -.60f, .46f, -.09f, -.37f,
                      -1.30f, .04f)) *
                      f11 +
                  def(make_float4x4(
                      .14f, .29f, -.45f, -.06f, -.65f, .33f, -.37f,
                      -.95f, .71f, -.07f, 1.00f, -.60f, -1.68f, -.20f,
                      -.00f, -.70f)) *
                      f12 +
                  def(make_float4x4(
                      -.31f, .69f, .56f, .13f, .95f, .36f, .56f, .59f,
                      -.63f, .52f, -.30f, .17f, 1.23f, .72f, .95f,
                      .75f)) *
                      f13 +
                  make_float4(-.90f, -3.26f, -.44f, -3.11f)) /
                  1.4f +
              f12;
        f03 = sin(def(make_float4x4(
                      .51f, -.98f, -.28f, .16f, -.22f, -.17f, -1.03f,
                      .22f, .70f, -.15f, .12f, .43f, .78f, .67f, -.85f,
                      -.25f)) *
                      f10 +
                  def(make_float4x4(
                      .81f, .60f, -.89f, .61f, -1.03f, -.33f, .60f,
                      -.11f, -.06f, .01f, -.02f, -.44f, .73f, .69f,
                      1.02f, .62f)) *
                      f11 +
                  def(make_float4x4(
                      -.10f, .52f, .80f, -.65f, .40f, -.75f, .47f,
                      1.56f, .03f, .05f, .08f, .31f, -.03f, .22f,
                      -1.63f, .07f)) *
                      f12 +
                  def(make_float4x4(
                      -.18f, -.07f, -1.22f, .48f, -.01f, .56f, .07f,
                      .15f, .24f, .25f, -.09f, -.54f, .23f, -.08f,
                      .20f, .36f)) *
                      f13 +
                  make_float4(-1.11f, -4.28f, 1.02f, -.23f)) /
                  1.4f +
              f13;
        auto output =
            dot(f00, make_float4(.09f, .12f, -.07f, -.03f)) +
            dot(f01, make_float4(-.04f, .07f, -.08f, .05f)) +
            dot(f02, make_float4(-.01f, .06f, -.02f, .07f)) +
            dot(f03, make_float4(-.05f, .07f, .03f, .04f)) - .16f;
        auto neural_distance = output * .85f;
        return ite(radius > 1.0f, radius - .9f, neural_distance);
    };
}

[[nodiscard]] inline Callable<float2(float3)> make_neural_bunny_scene(
    const Callable<float(float3)> &bunny) noexcept {
    return [&bunny](Float3 p) noexcept {
        auto plane = p.z + .5f;
        auto shape = bunny(p);
        return ite(shape < plane, make_float2(shape, 1.0f),
                   make_float2(plane, 0.0f));
    };
}

[[nodiscard]] inline Callable<float3(float3)> make_neural_bunny_normal(
    const Callable<float2(float3)> &scene) noexcept {
    return [&scene](Float3 p) noexcept {
        constexpr auto epsilon = .001f;
        auto center = scene(p).x;
        auto dx = scene(p - make_float3(epsilon, 0.0f, 0.0f)).x;
        auto dy = scene(p - make_float3(0.0f, epsilon, 0.0f)).x;
        auto dz = scene(p - make_float3(0.0f, 0.0f, epsilon)).x;
        return normalize(make_float3(center - dx, center - dy, center - dz));
    };
}

}// namespace luisa::compute::coro::example
