from pathlib import Path
import importlib.util
import inspect
import sys
import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TIMING_MODULE_PATH = (
    PROJECT_ROOT
    / "src"
    / "signal_processing"
    / "01_timing.py"
)


# ============================================================================
# SAFE MODULE IMPORT
# ============================================================================

def load_module(module_name: str, module_path: Path):

    spec = importlib.util.spec_from_file_location(
        module_name,
        module_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load module:\n{module_path}"
        )

    module = importlib.util.module_from_spec(spec)

    # Required for dataclasses and module introspection
    sys.modules[module_name] = module

    spec.loader.exec_module(module)

    return module


timing = load_module(
    "signal_processing_timing",
    TIMING_MODULE_PATH,
)


# ============================================================================
# HEADER
# ============================================================================

print("=" * 78)
print("TIMING MODULE API + SYNTHETIC VALIDATION")
print("=" * 78)

print("\nModule:")
print(TIMING_MODULE_PATH)


# ============================================================================
# API INSPECTION
# ============================================================================

print("\n" + "-" * 78)
print("ACTUAL FUNCTION SIGNATURES")
print("-" * 78)

functions = [
    "detect_gaps",
    "detect_discontinuities",
    "split_into_segments",
    "validate_segment",
]

for name in functions:

    if hasattr(timing, name):

        function = getattr(timing, name)

        print(f"\n{name}:")
        print(inspect.signature(function))

        print("\nDocstring:")
        print(inspect.getdoc(function) or "No docstring")

    else:

        print(f"\n✗ {name} NOT FOUND")


# ============================================================================
# CONFIGURATION
# ============================================================================

print("\n" + "-" * 78)
print("TIMING CONFIGURATION")
print("-" * 78)

if hasattr(timing, "TimingConfig"):

    print(
        inspect.signature(timing.TimingConfig)
    )

    config = timing.TimingConfig()

    print("\nDefault configuration:")
    print(config)


# ============================================================================
# SYNTHETIC DATA
# ============================================================================

print("\n" + "-" * 78)
print("SYNTHETIC TIMESTAMP DATA")
print("-" * 78)

timestamps = pd.Series(
    [
        0.00,
        0.01,
        0.02,
        0.03,
        0.04,

        # GAP

        0.20,
        0.21,
        0.22,
        0.23,

        # GAP

        0.50,
        0.51,
        0.52,
    ],
    name="timestamp",
)

print(timestamps.to_string(index=False))


test_data = pd.DataFrame(
    {
        "timestamp": timestamps,
        "ax": range(len(timestamps)),
    }
)

print("\nTest dataframe:")
print(test_data)


# ============================================================================
# GAP DETECTION
# ============================================================================

print("\n" + "-" * 78)
print("1. GAP DETECTION")
print("-" * 78)

if hasattr(timing, "detect_gaps"):

    try:

        signature = inspect.signature(
            timing.detect_gaps
        )

        print(
            "\nCalling detect_gaps using its actual signature..."
        )

        # The timing module currently appears to use
        # TimingConfig rather than a direct sampling-rate keyword.

        parameters = signature.parameters

        print("\nParameters:")

        for name, parameter in parameters.items():
            print(
                f"  {name}: "
                f"{parameter.kind} "
                f"default={parameter.default}"
            )

        # ------------------------------------------------------------
        # Try the most likely current API.
        # ------------------------------------------------------------

        config = timing.TimingConfig(
            expected_sampling_rate_hz=100.0
        )

        result = timing.detect_gaps(
            timestamps,
            config,
        )

        print("\n✓ detect_gaps executed")

        print("\nResult:")
        print(result)

    except Exception as exc:

        print("\n✗ detect_gaps failed")
        print(
            f"{type(exc).__name__}: {exc}"
        )


# ============================================================================
# DISCONTINUITY DETECTION
# ============================================================================

print("\n" + "-" * 78)
print("2. DISCONTINUITY DETECTION")
print("-" * 78)

if hasattr(timing, "detect_discontinuities"):

    try:

        signature = inspect.signature(
            timing.detect_discontinuities
        )

        print(
            "\nCalling detect_discontinuities..."
        )

        print("\nParameters:")

        for name, parameter in signature.parameters.items():
            print(
                f"  {name}: "
                f"{parameter.kind} "
                f"default={parameter.default}"
            )

        config = timing.TimingConfig(
            expected_sampling_rate_hz=100.0
        )

        result = timing.detect_discontinuities(
            timestamps,
            config,
        )

        print("\n✓ detect_discontinuities executed")

        print("\nResult:")
        print(result)

    except Exception as exc:

        print("\n✗ detect_discontinuities failed")
        print(
            f"{type(exc).__name__}: {exc}"
        )


# ============================================================================
# SEGMENT SPLITTING
# ============================================================================

print("\n" + "-" * 78)
print("3. CONTINUOUS SEGMENT SPLITTING")
print("-" * 78)

if hasattr(timing, "split_into_segments"):

    try:

        signature = inspect.signature(
            timing.split_into_segments
        )

        print("\nFunction signature:")
        print(signature)

        print("\nParameters:")

        for name, parameter in signature.parameters.items():
            print(
                f"  {name}: "
                f"{parameter.kind} "
                f"default={parameter.default}"
            )

        config = timing.TimingConfig(
            expected_sampling_rate_hz=100.0
        )

        # Most likely current structure:
        result = timing.split_into_segments(
            test_data,
            config,
        )

        print("\n✓ split_into_segments executed")

        print("\nResult type:")
        print(type(result))

        print("\nResult:")
        print(result)

    except Exception as exc:

        print("\n✗ split_into_segments failed")
        print(
            f"{type(exc).__name__}: {exc}"
        )


# ============================================================================
# SEGMENT VALIDATION
# ============================================================================

print("\n" + "-" * 78)
print("4. SEGMENT VALIDATION")
print("-" * 78)

if hasattr(timing, "validate_segment"):

    try:

        signature = inspect.signature(
            timing.validate_segment
        )

        print("\nFunction signature:")
        print(signature)

        print("\nParameters:")

        for name, parameter in signature.parameters.items():
            print(
                f"  {name}: "
                f"{parameter.kind} "
                f"default={parameter.default}"
            )

        config = timing.TimingConfig(
            expected_sampling_rate_hz=100.0
        )

        result = timing.validate_segment(
            test_data,
            config,
        )

        print("\n✓ validate_segment executed")

        print("\nResult:")
        print(result)

    except Exception as exc:

        print("\n✗ validate_segment failed")
        print(
            f"{type(exc).__name__}: {exc}"
        )


# ============================================================================
# FINAL
# ============================================================================

print("\n" + "=" * 78)
print("TIMING API INSPECTION COMPLETE")
print("=" * 78)

print(
    """
IMPORTANT:

This test intentionally does not modify the timing module.

Its purpose is to expose the exact API of 01_timing.py so that
the final validation test can be written against the real implementation.

No:
    - filtering
    - resampling
    - interpolation
    - artifact removal
    - feature extraction
    - ML training

has been performed.
"""
)