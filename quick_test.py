#!/usr/bin/env python3
"""
Quick smoke test for wrapper APIs.

This is a minimal test script to verify the basic API works.
For comprehensive testing, use test_wrappers.py
"""

import importlib
import sys


def test_package_import():
    """Test that the main package imports correctly."""
    print("Testing package import...")
    import genebeddings
    assert hasattr(genebeddings, '__version__')
    assert hasattr(genebeddings, 'SingleVariantGeometry')
    assert hasattr(genebeddings, 'EpistasisGeometry')
    assert hasattr(genebeddings, 'VariantEmbeddingDB')
    assert hasattr(genebeddings, 'DBMetadata')
    print(f"  genebeddings v{genebeddings.__version__}")
    print("  All public API symbols present")
    return True


def test_base_wrapper():
    """Test that BaseWrapper can be imported and has the right structure."""
    print("\nTesting BaseWrapper...")
    from genebeddings.wrappers import BaseWrapper

    # Check methods exist
    assert hasattr(BaseWrapper, 'embed')
    assert hasattr(BaseWrapper, 'predict_nucleotides')
    assert hasattr(BaseWrapper, 'predict_tracks')
    assert hasattr(BaseWrapper, 'supports_capability')
    assert hasattr(BaseWrapper, 'get_capabilities')

    print("  BaseWrapper has all required methods")
    return True


def test_wrapper_imports():
    """Test that all wrappers can be imported."""
    print("\nTesting wrapper imports...")

    wrappers = [
        'BaseWrapper',
        'AlphaGenomeWrapper',
        'BorzoiWrapper',
        'CaduceusWrapper',
        'ConvNovaWrapper',
        'DNABERTWrapper',
        'Evo2Wrapper',
        'GPNMSAWrapper',
        'HyenaDNAWrapper',
        'MutBERTWrapper',
        'NTWrapper',
        'RiNALMoWrapper',
        'SpeciesLMWrapper',
        'SpliceAIWrapper',
        'SpliceBertWrapper',
    ]

    success_count = 0
    for wrapper in wrappers:
        try:
            mod = importlib.import_module('genebeddings.wrappers')
            getattr(mod, wrapper)
            print(f"  {wrapper} imported successfully")
            success_count += 1
        except (ImportError, AttributeError) as e:
            print(f"  FAIL: {wrapper}: {e}")

    print(f"\n  {success_count}/{len(wrappers)} wrappers imported successfully")
    return success_count == len(wrappers)


def test_wrapper_inheritance():
    """Test that all wrappers inherit from BaseWrapper."""
    print("\nTesting wrapper inheritance...")

    from genebeddings.wrappers import BaseWrapper

    wrapper_names = [
        'BorzoiWrapper',
        'CaduceusWrapper',
        'ConvNovaWrapper',
        'DNABERTWrapper',
        'Evo2Wrapper',
        'GPNMSAWrapper',
        'HyenaDNAWrapper',
        'MutBERTWrapper',
        'NTWrapper',
        'RiNALMoWrapper',
        'SpeciesLMWrapper',
        'SpliceAIWrapper',
        'SpliceBertWrapper',
    ]

    mod = importlib.import_module('genebeddings.wrappers')
    all_inherit = True
    for name in wrapper_names:
        try:
            wrapper_class = getattr(mod, name)
            if issubclass(wrapper_class, BaseWrapper):
                print(f"  {name} inherits from BaseWrapper")
            else:
                print(f"  FAIL: {name} does NOT inherit from BaseWrapper")
                all_inherit = False
        except (AttributeError, ImportError) as e:
            print(f"  SKIP: {name}: {e}")

    return all_inherit


def test_api_signatures():
    """Test that key wrappers have the correct method signatures."""
    print("\nTesting API method signatures...")

    import inspect
    mod = importlib.import_module('genebeddings.wrappers')

    wrappers_to_check = ['BorzoiWrapper', 'CaduceusWrapper', 'NTWrapper']

    all_correct = True
    for name in wrappers_to_check:
        try:
            wrapper_class = getattr(mod, name)
            if hasattr(wrapper_class, 'embed'):
                sig = inspect.signature(wrapper_class.embed)
                params = list(sig.parameters.keys())
                if 'seq' in params and 'pool' in params and 'return_numpy' in params:
                    print(f"  {name}.embed() has correct signature")
                else:
                    print(f"  FAIL: {name}.embed() has incorrect signature: {params}")
                    all_correct = False
            else:
                print(f"  FAIL: {name} missing embed() method")
                all_correct = False
        except (AttributeError, ImportError) as e:
            print(f"  SKIP: {name}: {e}")

    return all_correct


def test_capability_methods():
    """Test that capability discovery methods work."""
    print("\nTesting capability discovery...")

    from genebeddings.wrappers import BaseWrapper

    assert hasattr(BaseWrapper, 'supports_capability')
    assert hasattr(BaseWrapper, 'get_capabilities')
    assert hasattr(BaseWrapper, '_implements_predict_nucleotides')
    assert hasattr(BaseWrapper, '_implements_predict_tracks')

    print("  Capability discovery methods present")
    return True


def main():
    print("=" * 70)
    print("QUICK SMOKE TEST FOR GENEBEDDINGS")
    print("=" * 70)

    tests = [
        ("Package Import", test_package_import),
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
            print(f"\n  CRASH: '{test_name}': {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")
    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
