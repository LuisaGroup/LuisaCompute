# Image Process Example

Interactive image processing window built with LuisaCompute + Dear ImGui.

* The window is split into two resolution-adaptive panes: the left pane shows the
  processed **RGB** image, the right pane shows the processed **alpha** channel in
  grayscale.
* A floating **Settings** window holds the load/save buttons and the operator
  list. Operators can be appended, inserted before a row, removed, and each
  operator has an RGBA argument edited with a color picker.
* The compute shader is dispatched **only when the settings change** (new image,
  operator added/removed/reordered, argument edited); the GUI itself runs at
  vsync (fixed FPS) and simply displays the processed textures.
* Images are loaded with `stb_image` (PNG/JPG/JPEG/BMP/TGA/HDR/PSD/GIF/PIC/PNM
  are all supported) as `PixelStorage::FLOAT4` textures, and can be saved back
  as PNG/BMP/TGA/JPG/HDR through Windows open/save file dialogs. The save dialog
  defaults to the extension of the loaded image (a `_processed` suffix is added
  to the file name) but other extensions can be typed to convert the format.
* **DPI aware:** on high-DPI displays the window scales its style, its fonts
  (re-baked at the DPI-scaled pixel size, so the text stays crisp) and its
  initial size to the monitor content scale, and follows display scaling changes
  at runtime. `ImGuiWindow::Config::dpi_aware` disables this (pixel-exact
  rendering independent of the display), and `ImGuiWindow::dpi_scale()` returns
  the applied content scale for applications (like this one) that scale their
  own hardcoded layout constants. Note that `imgui.ini` stores window sizes in
  ImGui units, so a Settings window saved on a 100% display is restored at its
  old pixel size; `ImGuiCond_FirstUseEver` applies the new scaled default on a
  fresh profile.

## Operator model

Each operator is one entry in a flat uint buffer, two uints per operator:

```
operators[i * 2 + 0] = op code
operators[i * 2 + 1] = RGBA argument quantized to 8 bit per channel
```

The kernel keeps the signature

```cpp
(ImageFloat input, ImageFloat output, ImageFloat alpha_output,
 BufferVar<uint> operators, UInt operator_count)
```

and applies the operators with a runtime `$for` loop containing a `$switch`:

```
value = input.read(coord);
$for (index, operator_count) {
    switch (operators[index * 2]) { case Add: ...; case Sub: ...; ... }
}
output.write(coord, value);
alpha_output.write(coord, gray(value.a));
```

Supported operators: `add`, `sub`, `mul`, `div`, `min`, `max`, `pow`, `abs`
(`abs` takes no argument, `div` guards its denominator). The host CPU
reference in `image_process.h` mirrors the kernel switch exactly, which is what
the headless tests validate against.

## Build and run

```bash
# configure (debug; ASan enabled by the project)
xmake f -p windows -a x64 -m debug -c
# build the example
xmake build example_image_process
# run the GUI (exchange vk with dx/cuda as needed)
xmake run example_image_process vk
# run with an image and a pre-built operator chain (skips the file dialogs)
xmake run example_image_process vk --image path/to/picture.png \
    --operators "mul 0.9 0.9 0.9 1; add 0.05 0.0 0.1 0" --save-to out.jpg
```

Command line options:

| Option | Description |
|---|---|
| `<backend>` | `vk`, `dx`, `cuda`, `metal`, `hip`, `fallback` |
| `--headless` | run the headless self test (no window) |
| `--output-dir <dir>` | directory for the headless test images (default `image_process_output`) |
| `--image <file>` | load this image on startup instead of using the dialog |
| `--save-to <file>` | save the processed image to this path before exiting |
| `--operators <spec>` | initial operator list, e.g. `"mul 0.9 0.9 0.9 1; add 0.1 0 0 0"` |
| `--frames <n>` | close the window after `n` frames (0 = until closed) |

## Headless self test

```bash
xmake run example_image_process vk --headless --output-dir image_process_out_vk
xmake run example_image_process dx --headless --output-dir image_process_out_dx
```

The self test (`headless_test.cpp`) covers:

1. **Operator encoding** – pack/unpack round trip, encoded layout, CPU reference.
2. **Operator chains on the GPU vs the CPU reference** – identity, single
   operators, all-operators chain, division by zero, a 10-operator "complex"
   chain and a 256-operator chain. The grayscale alpha output is verified too.
3. **Image formats** – PNG/BMP/TGA/JPG/HDR example images are written with
   `stb_image_write` itself, then loaded, processed on the device, saved again
   and reloaded. Lossless formats must round trip exactly (8 bit), lossy formats
   are checked with a tolerance. PPM and PSD (loadable by `stb_image` but not
   writable) are synthesized by the test.
4. **Edge cases** – empty operator list, `1x1` image, non-multiple-of-16
   resolutions (partial thread blocks) and the maximum operator count.

The test prints `Headless image process tests: <checks> checks, <failures> failures`
and returns the failure count as its exit code.

## Verifying the GUI pipeline without a mouse

The GUI code path (load -> upload -> dispatch -> display -> readback -> save) can
be exercised from the command line:

```bash
xmake run example_image_process vk --image picture.png \
    --operators "mul 0.5 0.5 0.5 1.0" --save-to out.png --frames 20
```

`tools/check_output.py` (needs Pillow + numpy, development tool only) checks the
saved result against `clamp(round(source * mul + add))` per channel, which is what
a `mul` / `add` operator chain must produce:

```bash
python tools/check_output.py picture.png out.png --mul "0.5 0.5 0.5" --tolerance 0
```

With an empty operator list the saved image must be bit-identical to the loaded
one (for lossless formats), which is how the display round trip is verified:

```bash
xmake run example_image_process vk --image picture.png --save-to out.png --frames 20
python tools/check_output.py picture.png out.png --tolerance 0   # plain comparison
```
