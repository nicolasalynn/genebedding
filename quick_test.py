#!/usr/bin/env python3
"""
Quick smoke test for wrapper APIs.

This is a minimal test script to verify the basic API works.
For comprehensive testing, use test_wrappers.py
"""

import sys

def test_base_wrapper():
    """Test that BaseWrapper can be imported and has the right structure."""
    print("Testing BaseWrapper...")
    from genebeddings.wrappers import BaseWrapper

    # Check methods exist
    assert hasattr(BaseWrapper, 'embed')
    assert hasattr(BaseWrapper, 'predict_nucleotides')
    assert hasattr(BaseWrapper, 'predict_tracks')
    assert hasattr(BaseWrapper, 'supports_capability')
    assert hasattr(BaseWrapper, 'get_capabilities')

    print("✓ BaseWrapper has all required methods")
    return True


def test_wrapper_imports():
    """Test that all wrappers can be imported."""
    print("\nTesting wrapper imports...")

    wrappers = [
        'BaseWrapper',
        'BorzoiWrapper',
        'CaduceusWrapper',
        'ConvNovaWrapper',
        'NTWrapper',
        'RiNALMoWrapper',
        'SpeciesLMWrapper',
    ]

    success_count = 0
    for wrapper in wrappers:
        try:
            exec(f"from genebeddings.wrappers import {wrapper}")
            print(f"✓ {wrapper} imported successfully")
            success_count += 1
        except ImportError as e:
            print(f"✗ Failed to import {wrapper}: {e}")

    print(f"\n{success_count}/{len(wrappers)} wrappers imported successfully")
    return success_count == len(wrappers)


def test_wrapper_inheritance():
    """Test that all wrappers inherit from BaseWrapper."""
    print("\nTesting wrapper inheritance...")

    from genebeddings.wrappers import (
        BaseWrapper,
        BorzoiWrapper,
        CaduceusWrapper,
        ConvNovaWrapper,
        NTWrapper,
        RiNALMoWrapper,
        SpeciesLMWrapper,
    )

    wrappers = [
        BorzoiWrapper,
        CaduceusWrapper,
        ConvNovaWrapper,
        NTWrapper,
        RiNALMoWrapper,
        SpeciesLMWrapper,
    ]

    all_inherit = True
    for wrapper_class in wrappers:
        if issubclass(wrapper_class, BaseWrapper):
            print(f"✓ {wrapper_class.__name__} inherits from BaseWrapper")
        else:
            print(f"✗ {wrapper_class.__name__} does NOT inherit from BaseWrapper")
            all_inherit = False

    return all_inherit


def test_api_signatures():
    """Test that all wrappers have the correct method signatures."""
    print("\nTesting API method signatures...")

    from genebeddings.wrappers import (
        BorzoiWrapper,
        CaduceusWrapper,
        NTWrapper,
    )

    import inspect

    # Check embed signature
    wrappers_to_check = [
        ('BorzoiWrapper', BorzoiWrapper),
        ('CaduceusWrapper', CaduceusWrapper),
        ('NTWrapper', NTWrapper),
    ]

    all_correct = True
    for name, wrapper_class in wrappers_to_check:
        # Check embed method
        if hasattr(wrapper_class, 'embed'):
            sig = inspect.signature(wrapper_class.embed)
            params = list(sig.parameters.keys())

            # Should have: self, seq, pool (kwonly), return_numpy (kwonly)
            if 'seq' in params and 'pool' in params and 'return_numpy' in params:
                print(f"✓ {name}.embed() has correct signature")
            else:
                print(f"✗ {name}.embed() has incorrect signature: {params}")
                all_correct = False
        else:
            print(f"✗ {name} missing embed() method")
            all_correct = False

    return all_correct


def test_capability_methods():
    """Test that capability discovery methods work."""
    print("\nTesting capability discovery...")

    try:
        # Import a wrapper (don't initialize, just check the class)
        from genebeddings.wrappers import BaseWrapper

        # Check that the methods exist and have reasonable behavior
        assert hasattr(BaseWrapper, 'supports_capability')
        assert hasattr(BaseWrapper, 'get_capabilities')
        assert hasattr(BaseWrapper, '_implements_predict_nucleotides')
        assert hasattr(BaseWrapper, '_implements_predict_tracks')

        print("✓ Capability discovery methods present")
        return True
    except Exception as e:
        print(f"✗ Capability discovery test failed: {e}")
        return False


def main():
    print("=" * 80)
    print("QUICK SMOKE TEST FOR GENEBEDDINGS WRAPPERS")
    print("=" * 80)

    tests = [
        ("Base Wrapper Structure", test_base_wrapper),
        ("Wrapper Imports", test_wrapper_imports),
        ("Wrapper Inheritance", test_wrapper_inheritance),
        ("API Signatures", test_api_signatures),
        ("Capability Discovery", test_capability_methods),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All smoke tests passed! The API is properly structured.")
        print("\nNext steps:")
        print("  - Run: python test_wrappers.py")
        print("  - This will let you test individual models with actual inference")
        return 0
    else:
        print("\n⚠ Some tests failed. Please review the API implementation.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
