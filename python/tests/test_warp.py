"""Tests for warp operations."""

import pytest
from luisa import (
    int32, float32,
    # Warp query
    warp_is_first_active_lane, warp_first_active_lane, warp_active_count_bits,
    # Warp reduction
    warp_sum, warp_product, warp_min, warp_max,
    warp_all, warp_any, warp_all_equal,
    # Warp prefix
    warp_prefix_sum, warp_prefix_product, warp_prefix_count_bits,
    # Warp broadcast
    warp_read_lane, warp_read_first_lane,
    # Warp bitwise
    warp_bit_and, warp_bit_or, warp_bit_xor, warp_bit_mask,
)


def test_warp_query_functions_exist():
    """Test that warp query functions are defined."""
    print("Testing warp query functions exist...")
    
    assert callable(warp_is_first_active_lane)
    assert callable(warp_first_active_lane)
    assert callable(warp_active_count_bits)
    
    print("  ✓ Warp query functions exist OK")


def test_warp_reduction_functions_exist():
    """Test that warp reduction functions are defined."""
    print("Testing warp reduction functions exist...")
    
    assert callable(warp_sum)
    assert callable(warp_product)
    assert callable(warp_min)
    assert callable(warp_max)
    assert callable(warp_all)
    assert callable(warp_any)
    assert callable(warp_all_equal)
    
    print("  ✓ Warp reduction functions exist OK")


def test_warp_prefix_functions_exist():
    """Test that warp prefix functions are defined."""
    print("Testing warp prefix functions exist...")
    
    assert callable(warp_prefix_sum)
    assert callable(warp_prefix_product)
    assert callable(warp_prefix_count_bits)
    
    print("  ✓ Warp prefix functions exist OK")


def test_warp_broadcast_functions_exist():
    """Test that warp broadcast functions are defined."""
    print("Testing warp broadcast functions exist...")
    
    assert callable(warp_read_lane)
    assert callable(warp_read_first_lane)
    
    print("  ✓ Warp broadcast functions exist OK")


def test_warp_bitwise_functions_exist():
    """Test that warp bitwise functions are defined."""
    print("Testing warp bitwise functions exist...")
    
    assert callable(warp_bit_and)
    assert callable(warp_bit_or)
    assert callable(warp_bit_xor)
    assert callable(warp_bit_mask)
    
    print("  ✓ Warp bitwise functions exist OK")
