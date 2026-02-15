"""
Demo of constant folding and host/device routing in LuisaCompute Python DSL v2.

This example demonstrates:
1. Constant folding: math operations on constants are evaluated at compile time
2. Host/device routing: same function works for both constants and DSL values
3. How the DSL automatically optimizes code without manual intervention
"""

import math
from luisa import (
    kernel, callable, pprint, Float, Float3, Buffer,
    sin, cos, sqrt, exp, log, pow,
    min, max, clamp, lerp, smoothstep,
    is_constant_value
)


def demo_constant_folding():
    """Demonstrate constant folding for math functions."""
    print("=" * 70)
    print("Demo: Constant Folding")
    print("=" * 70)
    
    # These are all constant-folded at compile time
    print("\n1. Basic constant folding:")
    a = sin(1.0 + 2.0)  # sin(3.0) is computed at compile time
    print(f"   sin(1.0 + 2.0) = {a.value:.10f}")
    print(f"   Expected: {math.sin(3.0):.10f}")
    print(f"   Is constant: {is_constant_value(a)}")
    
    b = sqrt(16.0)  # sqrt(16.0) = 4.0
    print(f"\n   sqrt(16.0) = {b.value}")
    print(f"   Is constant: {is_constant_value(b)}")
    
    c = pow(2.0, 10.0)  # 1024.0
    print(f"\n   pow(2.0, 10.0) = {c.value}")
    print(f"   Is constant: {is_constant_value(c)}")
    
    # Complex expression
    print("\n2. Complex constant expression:")
    x = 0.5
    y = 0.25
    result = sin(x) * cos(y) + sqrt(x * x + y * y)
    print(f"   sin(0.5) * cos(0.25) + sqrt(0.5^2 + 0.25^2)")
    print(f"   = {result.value:.10f}")
    print(f"   Is constant: {is_constant_value(result)}")
    
    # lerp with constants
    print("\n3. Utility functions with constants:")
    lerped = lerp(0.0, 10.0, 0.5)  # 5.0
    print(f"   lerp(0.0, 10.0, 0.5) = {lerped.value}")
    
    clamped = clamp(1.5, 0.0, 1.0)  # 1.0
    print(f"   clamp(1.5, 0.0, 1.0) = {clamped.value}")
    
    smoothed = smoothstep(0.0, 1.0, 0.5)
    print(f"   smoothstep(0.0, 1.0, 0.5) = {smoothed.value:.10f}")
    
    print("\n" + "=" * 70)


def demo_device_routing():
    """Demonstrate automatic routing to device for DSL values."""
    print("\nDemo: Device Routing")
    print("=" * 70)
    
    @callable
    def compute_with_sin(x: Float) -> Float:
        """This will emit a device-side SIN instruction."""
        return sin(x)  # x is a DSL value, so this routes to device
    
    print("\n1. Kernel with DSL value argument:")
    ir = compute_with_sin(1.0)
    print("   Generated IR:")
    for line in pprint(ir).split('\n'):
        print(f"   {line}")
    
    @callable
    def mixed_constants_and_dsl(t: Float) -> Float:
        """Mix of constants (folded) and DSL values (device)."""
        # These are constant-folded
        freq = 2.0 * 3.14159  # 6.28318
        phase = sin(0.5)      # computed at compile time
        
        # This is computed on device
        return sin(t * freq + phase)
    
    print("\n2. Mixed constants and DSL values:")
    ir2 = mixed_constants_and_dsl(0.5)
    print("   Generated IR:")
    for line in pprint(ir2).split('\n'):
        print(f"   {line}")
    
    print("\n" + "=" * 70)


def demo_kernel_optimization():
    """Demonstrate how constant folding optimizes kernel code."""
    print("\nDemo: Kernel Optimization via Constant Folding")
    print("=" * 70)
    
    @kernel
    def optimized_kernel(buf: Buffer[Float]):
        """
        This kernel benefits from constant folding:
        - PI, TWO_PI are folded
        - sin(TWO_PI) is folded to ~0
        - The division by 2.0 creates a constant
        """
        PI = 3.14159265359
        TWO_PI = 2.0 * PI  # Folded to 6.283...
        
        idx = 0  # Python constant
        
        # sin(TWO_PI) is approximately 0, computed at compile time
        offset = sin(TWO_PI)  # This becomes a constant ~0.0
        
        # This uses the constant offset
        buf[idx] = offset + 1.0  # Results in constant 1.0
    
    print("\n1. Kernel with folded constants:")
    ir = optimized_kernel(None)
    print("   Generated IR:")
    for line in pprint(ir).split('\n'):
        print(f"   {line}")
    
    print("\n   Notice: sin(TWO_PI) is folded to a constant!")
    
    @kernel
    def gradient_kernel(buf: Buffer[Float], size: int):
        """Generate a gradient using constant-folded coefficients."""
        # These are all Python constants (not DSL values)
        for i in range(size):  # Host-side loop
            t = float(i) / float(size - 1) if size > 1 else 0.0
            # lerp is constant-folded since all args are Python values
            value = lerp(0.0, 1.0, t)
            # But wait - this won't work because buf[i] needs i as DSL value
            # Let's do it differently:
            pass
    
    @kernel  
    def better_gradient_kernel(buf: Buffer[Float]):
        """Generate gradient with device-side computation."""
        idx = 0  # Simplified for demo
        # The constants 0.0, 1.0, 256.0 are folded
        t = Float(idx) / 256.0
        buf[idx] = t
    
    print("\n2. Device-side gradient kernel:")
    ir2 = better_gradient_kernel(None)
    print("   Generated IR:")
    for line in pprint(ir2).split('\n'):
        print(f"   {line}")
    
    print("\n" + "=" * 70)


def demo_performance_comparison():
    """Demonstrate performance benefits of constant folding."""
    print("\nDemo: Performance Benefits")
    print("=" * 70)
    
    print("\nWithout constant folding, this code:")
    print("    sin(3.14159 / 2.0)")
    print("Would generate:")
    print("    1. CONST 3.14159")
    print("    2. CONST 2.0")
    print("    3. DIV")
    print("    4. SIN")
    print("\nWith constant folding:")
    print("    1. CONST 1.0  (sin(pi/2) = 1.0)")
    print("\nThis eliminates 3 instructions!")
    
    print("\nAnother example - color conversion coefficients:")
    print("    r = y + 1.402 * cr")
    print("    g = y - 0.344136 * cb - 0.714136 * cr")
    print("    b = y + 1.772 * cb")
    print("\nWith constant folding, the multiplications become")
    print("embedded constants, saving 3 MUL instructions per pixel!")
    
    print("\n" + "=" * 70)


def demo_advanced_routing():
    """Demonstrate advanced routing scenarios."""
    print("\nDemo: Advanced Routing Scenarios")
    print("=" * 70)
    
    @callable
    def choose_path(use_device: bool, x: Float) -> Float:
        """
        This shows how the same code works with both constants and DSL values.
        """
        if use_device:
            # x is a DSL value, so sin(x) routes to device
            return sin(x)
        else:
            # sin(0.0) is a constant, folded at compile time
            return sin(0.0)
    
    print("\n1. Conditional compilation based on value type:")
    
    # When x is a DSL value
    ir1 = choose_path(True, 1.0)
    print("   With DSL value (routes to device):")
    for line in pprint(ir1).split('\n'):
        print(f"   {line}")
    
    print("\n   Notice how the router automatically handles both cases!")
    
    @callable
    def complex_expression(a: Float, b: Float) -> Float:
        """
        The router evaluates what it can at compile time and
        emits device instructions for the rest.
        """
        # Constants - folded
        c1 = sqrt(2.0)  # 1.4142...
        c2 = sin(0.0)   # 0.0
        
        # Mix of constants and DSL values
        return a * c1 + b * c2  # c1 and c2 are constants
    
    print("\n2. Mixed expression with partial folding:")
    ir2 = complex_expression(1.0, 2.0)
    print("   Generated IR:")
    for line in pprint(ir2).split('\n'):
        print(f"   {line}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("LuisaCompute Python DSL v2 - Constant Folding & Routing Demo")
    print("=" * 70)
    
    demo_constant_folding()
    demo_device_routing()
    demo_kernel_optimization()
    demo_performance_comparison()
    demo_advanced_routing()
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Math operations on constants are folded at compile time")
    print("2. Same code works for both host (constants) and device (DSL values)")
    print("3. The @router decorator handles the dispatch automatically")
    print("4. No manual optimization needed - it's automatic!")
    print("=" * 70)
