"""
Pytest configuration and fixtures for Luisa DSL tests.
"""

import pytest
from luisa import pprint


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "pprint: mark test to pretty print IR")


@pytest.fixture
def print_ir():
    """
    Fixture that returns a function to print and return IR.
    Automatically extracts IR if a StagedFunction or SpecializedFunctionProxy is passed.
    """

    def _print_ir(obj, title=None):
        # Extract IR if it's a staged function
        ir = obj.ir if hasattr(obj, 'ir') else obj
        
        if title:
            print(f"\n{'=' * 60}")
            print(f"  {title}")
            print(f"{'=' * 60}")
        else:
            print(f"\n{'=' * 60}")
            print("  Generated IR")
            print(f"{'=' * 60}")

        print(pprint(ir, recursive=True))
        print(f"{'=' * 60}\n")
        return ir

    return _print_ir


@pytest.fixture
def verify_ir():
    """
    Fixture that verifies the generated IR matches a reference string.
    Normalizes whitespace for comparison.
    Automatically extracts IR if a StagedFunction or SpecializedFunctionProxy is passed.
    """

    def _verify(obj, expected_ref, recursive=True):
        # Extract IR if it's a staged function
        ir = obj.ir if hasattr(obj, 'ir') else obj
        
        # Generate IR without location info for comparison
        actual = pprint(ir, recursive=recursive, show_location=False)
        
        def normalize(s):
            return "\n".join(line.strip() for line in s.strip().splitlines() if line.strip())
            
        actual_norm = normalize(actual)
        expected_norm = normalize(expected_ref)
        
        if actual_norm != expected_norm:
            print("\nIR Mismatch!")
            print("\n--- ACTUAL ---")
            print(actual)
            print("\n--- EXPECTED ---")
            print(expected_ref)
            
            # Show diff if possible
            import difflib
            diff = difflib.unified_diff(
                expected_norm.splitlines(),
                actual_norm.splitlines(),
                fromfile='expected',
                tofile='actual',
                lineterm=''
            )
            print("\n--- DIFF ---")
            print('\n'.join(diff))
            
            assert actual_norm == expected_norm

    return _verify


@pytest.fixture
def verify_execution():
    """
    Fixture that verifies the IR was actually generated.
    Automatically extracts IR if a StagedFunction or SpecializedFunctionProxy is passed.
    """

    def _verify(obj, min_blocks=1, min_instructions=1):
        # Extract IR if it's a staged function
        ir = obj.ir if hasattr(obj, 'ir') else obj
        
        assert ir is not None, "IR should not be None"
        assert hasattr(ir, 'blocks'), "IR should have blocks attribute"
        assert len(ir.blocks) >= min_blocks, f"IR should have at least {min_blocks} block(s)"

        total_instructions = sum(len(b.instructions) for b in ir.blocks)
        assert total_instructions >= min_instructions, \
            f"IR should have at least {min_instructions} instruction(s), got {total_instructions}"

        print(f"✓ Execution verified: {len(ir.blocks)} blocks, {total_instructions} instructions")
        return ir

    return _verify
