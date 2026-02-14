"""
Example usage of the LuisaCompute Python DSL v2.

This demonstrates the multistage programming model with complete
type hinting support and automatic constant folding.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from luisa import (
    # Types
    float32, int32, bool_,
    float2, float3, float4,
    Buffer,
    
    # Decorators
    kernel, callable,
    
    # Builder (for advanced usage)
    IRBuilder,
)


def example_basic_callable():
    """Example: Basic callable function."""
    print("=" * 60)
    print("Example: Basic Callable Function")
    print("=" * 60)
    
    @callable
    def add(a: float32, b: float32) -> float32:
        return a + b
    
    print(f"Function: add(a: float32, b: float32) -> float32")
    
    # Stage 3: Execute the builder to generate IR
    ir_func = add(1.0, 2.0)
    
    print("\nGenerated IR:")
    print(ir_func)
    print()


def example_control_flow():
    """Example: Control flow with if-else."""
    print("=" * 60)
    print("Example: Control Flow (if-else)")
    print("=" * 60)
    
    @callable
    def abs_val(x: float32) -> float32:
        if x >= 0.0:
            return x
        else:
            return -x
    
    print(f"Function: abs_val(x: float32) -> float32")
    
    ir_func = abs_val(-5.0)
    
    print("\nGenerated IR:")
    print(ir_func)
    print()


def example_constant_folding():
    """Example: Constant folding with captured variables."""
    print("=" * 60)
    print("Example: Constant Folding")
    print("=" * 60)
    
    # These are captured variables (Stage 3 constants)
    threshold = 0.5
    scale = 2.0
    
    @callable
    def process(x: float32) -> float32:
        # All these conditions are evaluated at Stage 3!
        if threshold > 0.0:  # Constant folded to True
            if x > threshold:
                return x * scale  # scale is constant 2.0
            else:
                return 0.0
        else:
            return x  # Dead code - never generated
    
    print(f"Captured variables: threshold={threshold}, scale={scale}")
    print(f"Function: process(x: float32) -> float32")
    
    ir_func = process(1.0)
    
    print("\nGenerated IR:")
    print(ir_func)
    print("Note: The else branch of 'if threshold > 0.0' was eliminated")
    print("      because threshold is known at Stage 3!")
    print()


def example_loop():
    """Example: For loop (dynamic)."""
    print("=" * 60)
    print("Example: Dynamic Loop")
    print("=" * 60)
    
    @callable
    def sum_array(n: int32) -> float32:
        total = 0.0
        for i in range(4):  # Small constant range for demo
            total = total + 1.0
        return total
    
    print(f"Function: sum_array(n: int32) -> float32")
    print(f"Note: This generates a dynamic loop on the device")
    
    ir_func = sum_array(10)
    
    print("\nGenerated IR:")
    print(ir_func)
    print()


def example_kernel():
    """Example: Kernel function."""
    print("=" * 60)
    print("Example: Kernel Function")
    print("=" * 60)
    
    @kernel
    def saxpy(a: float32, x: float32, y: float32) -> float32:
        # Simplified version without buffers for demo
        return a * x + y
    
    print(f"Function: saxpy(a: float32, x: float32, y: float32) -> float32")
    print(f"Note: Marked as @kernel for parallel execution")
    
    ir_func = saxpy(2.0, 1.0, 3.0)
    
    print("\nGenerated IR:")
    print(ir_func)
    print()


def example_multistage():
    """Example: Demonstrating multistage programming."""
    print("=" * 60)
    print("Example: Multistage Programming")
    print("=" * 60)
    
    print("""
    The DSL uses 3 stages:
    
    Stage 1: Parse (at decoration time)
        - Parse Python AST
        - Extract type annotations
        - Analyze captured variables
    
    Stage 2: Builder Function Generation (at decoration time)
        - Create a staged function
        - Set up the builder infrastructure
    
    Stage 3: IR Generation (at call time)
        - Execute builder with actual values
        - Captured variables become constants
        - Generate actual IR instructions
    
    This enables powerful optimizations:
    - Constant folding of captured variables
    - Dead code elimination
    - Loop unrolling for small constant bounds
    """)
    
    # Configuration captured at Stage 3
    USE_OPTIMIZATION = True
    ITERATIONS = 4
    
    @callable
    def optimized_compute(x: float32) -> float32:
        result = x
        
        if USE_OPTIMIZATION:  # Constant folded at Stage 3
            # This branch is always taken
            for i in range(ITERATIONS):  # Can be unrolled at Stage 3
                result = result * 1.1
        else:
            # This is dead code - never generated
            result = result * 2.0
        
        return result
    
    print(f"Configuration: USE_OPTIMIZATION={USE_OPTIMIZATION}, ITERATIONS={ITERATIONS}")
    print(f"Function: optimized_compute(x: float32) -> float32")
    
    ir_func = optimized_compute(1.0)
    
    print("\nGenerated IR:")
    print(ir_func)
    print("\nNote: The else branch was eliminated, and the loop could be unrolled")
    print("      because ITERATIONS is known at Stage 3!")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("*" * 60)
    print("* LuisaCompute Python DSL v2 - Examples")
    print("*" * 60)
    print("\n")
    
    example_basic_callable()
    example_control_flow()
    example_constant_folding()
    example_loop()
    example_kernel()
    example_multistage()
    
    print("*" * 60)
    print("* All examples completed!")
    print("*" * 60)


if __name__ == "__main__":
    main()
