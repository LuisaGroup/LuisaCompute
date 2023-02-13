import numpy as np
import lcapi
from . import globalvars
from .globalvars import stream, get_global_device


class GUI:
    def __init__(self, title: str, resolution, vsync=True, show_FPS=True):
        if not globalvars.stream_support_gui:
            raise RuntimeError(
                "Stream no support GUI, use init(support_gui=True).")
        self.window = lcapi.PyWindow()
        self.window.reset(get_global_device(), globalvars.stream,
                          title, resolution[0], resolution[1], vsync)
        self.resolution = resolution
        self.clock = lcapi.Clock()
        self.tic = False

    def show(self):
        self.window.present(globalvars.stream, self.tex.handle,
                            self.resolution[0], self.resolution[1], 0, self.tex.storage)
        if self.tic:
            time = self.clock.toc()
        else:
            time = 0.0
        self.clock.tic()
        self.tic = True
        return time

    def running(self):
        return not self.window.should_close()

    def set_image(self, tex):
        self.tex = tex

    def is_key_down(self, key: int):
        return self.window.key_event(key) == 1

    def is_key_pressed(self, key: int):
        return self.window.key_event(key) == 2

    def is_key_up(self, key: int):
        return self.window.key_event(key) == 3

    def is_mouse_down(self, mouse: int):
        return self.window.mouse_event(mouse) == 1

    def is_mouse_pressed(self, mouse: int):
        return self.window.mouse_event(mouse) == 2

    def is_mouse_up(self, mouse: int):
        return self.window.mouse_event(mouse) == 3

    def cursor_pos(self):
        return self.window.cursor_pos()


class KeyCode:
    Space: int = 32
    Apostrophe: int = 39
    Comma: int = 44
    Minus: int = 45
    Period: int = 46
    Slash: int = 47
    Alpha0: int = 48
    Alpha1: int = 49
    Alpha2: int = 50
    Alpha3: int = 51
    Alpha4: int = 52
    Alpha5: int = 53
    Alpha6: int = 54
    Alpha7: int = 55
    Alpha8: int = 56
    Alpha9: int = 57
    Semicolon: int = 59
    Equal: int = 61
    A: int = 65
    B: int = 66
    C: int = 67
    D: int = 68
    E: int = 69
    F: int = 70
    G: int = 71
    H: int = 72
    I: int = 73
    J: int = 74
    K: int = 75
    L: int = 76
    M: int = 77
    N: int = 78
    O: int = 79
    P: int = 80
    Q: int = 81
    R: int = 82
    S: int = 83
    T: int = 84
    U: int = 85
    V: int = 86
    W: int = 87
    X: int = 88
    Y: int = 89
    Z: int = 90
    LeftBracket: int = 91
    Backslash: int = 92
    RightBracket: int = 93
    GraveAccent: int = 96
    World1: int = 161
    World2: int = 162
    Escape: int = 256
    Enter: int = 257
    Tab: int = 258
    Backspace: int = 259
    Insert: int = 260
    Delete: int = 261
    Right: int = 262
    Left: int = 263
    Down: int = 264
    Up: int = 265
    PageUp: int = 266
    PageDown: int = 267
    Home: int = 268
    End: int = 269
    CapsLock: int = 280
    ScrollLock: int = 281
    NumLock: int = 282
    PrintScreen: int = 283
    Pause: int = 284
    F1: int = 290
    F2: int = 291
    F3: int = 292
    F4: int = 293
    F5: int = 294
    F6: int = 295
    F7: int = 296
    F8: int = 297
    F9: int = 298
    F10: int = 299
    F11: int = 300
    F12: int = 301
    F13: int = 302
    F14: int = 303
    F15: int = 304
    F16: int = 305
    F17: int = 306
    F18: int = 307
    F19: int = 308
    F20: int = 309
    F21: int = 310
    F22: int = 311
    F23: int = 312
    F24: int = 313
    F25: int = 314
    KeyPad0: int = 320
    KeyPad1: int = 321
    KeyPad2: int = 322
    KeyPad3: int = 323
    KeyPad4: int = 324
    KeyPad5: int = 325
    KeyPad6: int = 326
    KeyPad7: int = 327
    KeyPad8: int = 328
    KeyPad9: int = 329
    KeyPadDecimal: int = 330
    KeyPadDivide: int = 331
    KeyPadMultiply: int = 332
    KeyPadSubtract: int = 333
    KeyPadAdd: int = 334
    KeyPadEnter: int = 335
    KeyPadEqual: int = 336
    LeftShift: int = 340
    LeftControl: int = 341
    LeftAlt: int = 342
    LeftSuper: int = 343
    RightShift: int = 344
    RightControl: int = 345
    RightAlt: int = 346
    RightSuper: int = 347
    Menu: int = 348
