#!/usr/bin/env python3
"""Exercise an installed wheel without a GPU; optionally dispatch on a backend.

Used by cibuildwheel as ``{python} {project}/scripts/test_wheel.py``.
Run with ``--backend fallback`` to also compile and execute a CPU kernel.
"""

import argparse
import importlib.machinery
import importlib.metadata
import importlib.util
import inspect
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
WHEEL_GLOBAL_OFFSET = 5


def make_typed_callable():
    import luisa

    captured_offset = 3

    @luisa.func
    def wheel_smoke_callable(value: int, offset: int = 7) -> int:
        return value + offset + captured_offset + WHEEL_GLOBAL_OFFSET

    return wheel_smoke_callable


def run_python_child(*arguments, timeout=60):
    result = subprocess.run(
        [sys.executable, "-I", "-X", "faulthandler", *arguments],
        capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise AssertionError(
            f"Child exited with {result.returncode}:\n{result.stdout}\n{result.stderr}")


class WheelSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import luisa
        import numpy

        cls.luisa = luisa
        cls.lcapi = luisa.lcapi
        cls.numpy = numpy

    def test_subprocess_import_enum_shutdown(self):
        code = textwrap.dedent("""
            import gc
            from luisa import lcapi

            count = 0
            for enum_type in vars(lcapi).values():
                if isinstance(enum_type, type) and hasattr(enum_type, "__members__"):
                    for member in enum_type.__members__.values():
                        assert isinstance(member.name, str)
                        assert isinstance(member.value, int)
                        count += 1
            assert count > 0, "native enum properties were not exercised"
            gc.collect()
            print("Native import, enum properties and collection passed", flush=True)
        """)
        for attempt in range(3):
            with self.subTest(attempt=attempt):
                run_python_child("-c", code)

    def test_subprocess_callback_and_ast_lifetimes(self):
        run_python_child("-c", textwrap.dedent("""
            import gc
            import sys
            import weakref
            from luisa import lcapi

            events = []
            def callback(level, message):
                events.append((level, message))

            callback_ref = weakref.ref(callback)
            lcapi.log_level_info()
            lcapi.set_log_callback(callback)
            del callback
            gc.collect()
            assert callback_ref() is not None
            lcapi.log_info("wheel callback lifetime")
            assert events == [("I", "wheel callback lifetime")]
            lcapi.set_log_callback(None)
            gc.collect()
            assert callback_ref() is None
            lcapi.log_info("wheel callback reset")
            assert len(events) == 1

            errors = []
            original_hook = sys.unraisablehook
            def failing_callback(level, message):
                raise ValueError("wheel callback failure")

            try:
                sys.unraisablehook = errors.append
                lcapi.set_log_callback(failing_callback)
                lcapi.log_info("callback exception")
            finally:
                lcapi.set_log_callback(None)
                sys.unraisablehook = original_hook
            assert len(errors) == 1
            assert isinstance(errors[0].exc_value, ValueError)
            assert str(errors[0].exc_value) == "wheel callback failure"

            def build_body():
                lcapi.builder().return_(None)

            body_ref = weakref.ref(build_body)
            builder = lcapi.FunctionBuilder.define_callable(build_body)
            del build_body
            gc.collect()
            assert body_ref() is None
            function = builder.function()
            assert isinstance(function, lcapi.Function)
            del function, builder
            gc.collect()
            # An installed callback must also be released during normal shutdown.
            lcapi.set_log_callback(lambda level, message: None)
            lcapi.log_info("wheel callback at shutdown")
            print("Callback and callable AST lifetimes passed", flush=True)
        """))

    def test_subprocess_function_retains_builder(self):
        run_python_child("-c", textwrap.dedent("""
            import gc
            import weakref
            from luisa import lcapi

            def make_kernel(count):
                def build_body():
                    for _ in range(count):
                        lcapi.builder().argument(lcapi.Type.from_("int"))
                return lcapi.FunctionBuilder.define_kernel(build_body)

            builder = make_kernel(1)
            builder_ref = weakref.ref(builder)
            function = builder.function()
            del builder
            gc.collect()
            assert builder_ref() is not None, "Function lost its owning builder"
            for count in range(2, 10):
                replacement = make_kernel(count)
                assert function.argument_size() == 1
                del replacement
            gc.collect()
            del function
            gc.collect()
            assert builder_ref() is None, "Function retained its builder after release"
            print("Function retains and releases its builder", flush=True)
        """))

    def test_subprocess_ast_views_retain_builder(self):
        run_python_child("-c", textwrap.dedent("""
            import gc
            import weakref
            from luisa import lcapi

            expressions = []
            def build_source():
                expressions.append(lcapi.builder().literal(lcapi.Type.from_("bool"), True))

            source = lcapi.FunctionBuilder.define_callable(build_source)
            source_ref = weakref.ref(source)
            expression = expressions.pop()
            del source
            gc.collect()
            assert source_ref() is not None, "Expression lost its owning builder"
            target = lcapi.FunctionBuilder.define_callable(
                lambda: lcapi.builder().return_(expression))
            del target, expression
            gc.collect()
            assert source_ref() is None, "Expression retained its builder after release"
            scopes = []
            def build_scope():
                condition = lcapi.builder().literal(lcapi.Type.from_("bool"), True)
                statement = lcapi.builder().if_(condition)
                scopes.append(statement.true_branch())

            source = lcapi.FunctionBuilder.define_callable(build_scope)
            source_ref = weakref.ref(source)
            scope = scopes.pop()
            del source
            gc.collect()
            assert source_ref() is not None, "Scope lost its owning statement and builder"
            del scope
            gc.collect()
            assert source_ref() is None, "Scope retained its owner after release"
            print("Expression and statement scopes retain and release their builders", flush=True)
        """))

    def test_subprocess_dlpack_lifetimes(self):
        run_python_child(
            str(ROOT / "src" / "tests" / "python" / "test-dlpack-errors.py"))

    def test_installed_native_module(self):
        distribution = importlib.metadata.distribution("luisa-python")
        package_file = Path(self.luisa.__file__).resolve()
        expected_file = Path(distribution.locate_file("luisa/__init__.py")).resolve()
        self.assertEqual(package_file, expected_file,
                         "test must import the installed wheel")
        self.assertFalse(package_file.is_relative_to(ROOT / "src" / "py"))
        extension_file = str(Path(self.lcapi.__file__).resolve())
        self.assertTrue(any(extension_file.endswith(suffix)
                            for suffix in importlib.machinery.EXTENSION_SUFFIXES),
                        f"lcapi must be a native extension: {extension_file}")
        print(f"Python {sys.version.split()[0]}, NumPy {self.numpy.__version__}", flush=True)
        print(f"Installed lcapi: {extension_file}", flush=True)

    def test_native_types_and_numpy_vectors(self):
        scalar = self.lcapi.Type.from_("float")
        self.assertTrue(scalar.is_scalar())
        self.assertEqual(scalar.size(), 4)
        self.assertEqual(scalar.description(), "float")
        for dimension in (2, 3, 4):
            with self.subTest(dimension=dimension):
                dtype = self.lcapi.Type.from_(f"vector<float,{dimension}>")
                self.assertTrue(dtype.is_vector())
                self.assertEqual(dtype.dimension(), dimension)
                values = self.numpy.arange(1, dimension + 1, dtype=self.numpy.float32)
                vector = getattr(self.lcapi, f"float{dimension}")(values)
                self.numpy.testing.assert_array_equal(
                    [vector[index] for index in range(dimension)], values)
                result = vector * 2.0 + 1.0
                self.numpy.testing.assert_array_equal(
                    [result[index] for index in range(dimension)], values * 2.0 + 1.0)
                vector.x = 9.0
                self.assertEqual(vector[0], 9.0)

    def test_source_annotations_closures_and_callable_ast(self):
        import ast
        import sourceinspect
        from luisa import globalvars

        function = make_typed_callable()
        python_function = function.pyfunc
        source, _ = sourceinspect.getsourcelines(python_function)
        tree = ast.parse(textwrap.dedent("".join(source)))
        self.assertIsInstance(tree.body[0], ast.FunctionDef)
        self.assertEqual(inspect.get_annotations(python_function),
                         {"value": int, "offset": int, "return": int})
        signature = inspect.signature(python_function)
        self.assertIs(signature.parameters["value"].annotation, int)
        self.assertEqual(signature.parameters["offset"].default, 7)
        closure = inspect.getclosurevars(python_function)
        self.assertEqual(closure.nonlocals["captured_offset"], 3)
        self.assertEqual(closure.globals["WHEEL_GLOBAL_OFFSET"], 5)
        self.assertEqual(python_function(1), 16)

        # Callable construction builds the real C++ AST without creating a device.
        for argument_types in ((int,), (int, int)):
            with self.subTest(argument_types=argument_types):
                compiled = function.get_compiled(
                    func_type=1, allow_ref=False, argtypes=argument_types)
                self.assertIsInstance(compiled.function, self.lcapi.Function)
                # argument_size() describes dispatchable kernels: callable
                # bindings still contain placeholders for their unbound arguments.
                self.assertIsInstance(compiled.builder, self.lcapi.FunctionBuilder)
                self.assertEqual(compiled.argtypes, argument_types)
                for name in tuple(signature.parameters)[:len(argument_types)]:
                    self.assertIsInstance(compiled.local_variable[name].expr, self.lcapi.RefExpr)
                self.assertIs(compiled.return_type, int)
                self.assertEqual(compiled.default_arg_values["offset"], 7)
                self.assertIsNone(compiled.shader_handle)
        self.assertIsNone(globalvars.device)
        self.assertIsNone(globalvars.current_context)


def add_dlpack_tests(suite):
    path = ROOT / "src" / "tests" / "python" / "test-dlpack-errors.py"
    spec = importlib.util.spec_from_file_location("wheel_dlpack_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))


def check_device_interface_owner(backend):
    import gc
    import weakref
    from luisa import lcapi

    context = lcapi.Context(str(Path(lcapi.__file__).resolve()))
    device = context.create_device(backend)
    device_ref = weakref.ref(device)
    implementation = device.impl()
    del device
    gc.collect()
    assert device_ref() is not None, "Device interface lost its owning device"
    buffer = implementation.create_buffer(lcapi.Type.from_("float"), 4)
    implementation.destroy_buffer(buffer.handle())
    del implementation
    gc.collect()
    assert device_ref() is None, "Device interface retained its owner after release"


def check_async_readback_owners(buffer, expected):
    import gc
    import weakref
    import luisa
    import numpy
    from luisa import globalvars

    owners = []
    for _ in range(3):
        target = numpy.empty_like(expected)
        owners.append(weakref.ref(target))
        buffer.copy_to(target, sync=False)
        del target
        gc.collect()
        assert all(owner() is not None for owner in owners)
        luisa.execute()
        gc.collect()
        assert all(owner() is not None for owner in owners), "Readback released before synchronization"
    retained = numpy.empty_like(expected)
    buffer.copy_to(retained, sync=False)
    luisa.synchronize()
    gc.collect()
    assert all(owner() is None for owner in owners), "Readback owners retained after synchronization"
    numpy.testing.assert_array_equal(retained, expected)

    stream = globalvars.device.create_stream(False)
    target = numpy.empty_like(expected)
    owner = weakref.ref(target)
    buffer.copy_to(target, sync=False, stream=stream)
    del target
    retained = numpy.empty_like(expected)
    buffer.copy_to(retained, sync=False, stream=stream)
    stream.execute()
    gc.collect()
    assert owner() is not None
    del stream
    gc.collect()
    assert owner() is None, "Destroyed stream retained a readback owner"
    numpy.testing.assert_array_equal(retained, expected)


def check_bindless_release(backend):
    run_python_child("-c", textwrap.dedent("""
        import gc
        import sys
        import luisa

        luisa.init(backend_name=sys.argv[1], support_gui=False)
        array = luisa.BindlessArray(4)
        buffer = luisa.Buffer(32, int)
        array.emplace(0, buffer)
        array.update(sync=True)
        del buffer
        gc.collect()
        array.remove_buffer(0)
        array.update(sync=True)

        buffer = luisa.Buffer(32, int)
        array.emplace(0, buffer)
        array.update(sync=True)
        del buffer
        gc.collect()
        replacement = luisa.Buffer(32, int)
        array.emplace(0, replacement)
        array.update(sync=True)
        del replacement
        gc.collect()
        array.remove_buffer(0)
        array.update(sync=True)
        del array
        gc.collect()
        luisa.synchronize()
        print("Bindless removal and replacement released final resource references", flush=True)
    """), backend, timeout=30)


def run_backend_smoke(backend):
    import luisa
    import numpy

    check_bindless_release(backend)
    check_device_interface_owner(backend)
    luisa.init(backend_name=backend, support_gui=False)
    values = numpy.arange(64, dtype=numpy.float32)
    buffer = luisa.Buffer.from_array(values)

    @luisa.func
    def wheel_smoke_kernel(data: luisa.BufferType(float)):
        index = dispatch_id().x
        data.write(index, data.read(index) * 2.0 + 1.0)

    wheel_smoke_kernel(buffer, dispatch_size=values.size)
    actual = buffer.numpy()
    expected = values * 2.0 + 1.0
    numpy.testing.assert_array_equal(actual, expected)
    check_async_readback_owners(buffer, expected)
    print(f"Backend {backend}: kernel, buffer values and asynchronous owners passed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", help="also compile and execute a kernel (e.g. fallback)")
    arguments = parser.parse_args()
    previous_directory = Path.cwd()
    # Keep the checkout and generated artifacts out of the import working directory.
    with tempfile.TemporaryDirectory(prefix="luisa-wheel-smoke-") as temporary:
        try:
            os.chdir(temporary)
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(WheelSmoke)
            add_dlpack_tests(suite)
            result = unittest.TextTestRunner(verbosity=2).run(suite)
            if not result.wasSuccessful():
                return 1
            if arguments.backend:
                run_backend_smoke(arguments.backend)
        finally:
            os.chdir(previous_directory)
    return 0


if __name__ == "__main__":
    sys.exit(main())
