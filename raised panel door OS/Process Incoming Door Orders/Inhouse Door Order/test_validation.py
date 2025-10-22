#!/usr/bin/env python3
"""Test the new two-stage validation logic."""

import sys
sys.path.insert(0, '/home/nando/projects/anyDoor/raised panel door OS/Process Incoming Door Orders/Inhouse Door Order')

from claude_verification import is_suspicious_measurement, matches_valid_measurement_pattern

def test_validation():
    """Test various measurement formats."""

    test_cases = [
        # (text, raw_ocr, expected_suspicious, description)
        ("6 1", "6 1/", True, "M9 case - incomplete fraction (missing denominator)"),
        ("6 1", None, True, "M9 case - incomplete fraction (no raw_ocr)"),
        ("6 1/16", None, False, "Valid whole + fraction"),
        ("30 1/4 F", None, False, "Valid with F notation"),
        ("18 O", None, False, "Valid whole number with O notation"),
        ("24 3/4 NH", None, False, "Valid with NH notation"),
        ("30", None, False, "Valid whole number"),
        ("6 1/", "6 1/", True, "Incomplete fraction (no denominator)"),
        ("6.5", None, True, "Invalid - decimal"),
        ("20 XY", None, True, "Invalid - bad notation"),
        ("8/8", None, True, "Invalid - impossible fraction"),
        ("150", None, True, "Invalid - over 120 inches"),
        ("20—3/4", "20—3/4", True, "Invalid - em dash not removed by cleaning"),
        ("20 3/4", "20—3/4", False, "Valid - cleaning removed dash from raw OCR"),
        ("6 2/8", None, True, "Invalid - even numerator for /8 fraction"),
        ("30 1/2", None, False, "Valid 1/2 fraction"),
        ("24 3/4", None, False, "Valid 3/4 fraction"),
    ]

    print("Testing two-stage validation logic:\n")
    print("=" * 80)

    passed = 0
    failed = 0

    for text, raw_ocr, expected_suspicious, description in test_cases:
        is_susp, reason = is_suspicious_measurement(text, raw_ocr)

        status = "✓ PASS" if is_susp == expected_suspicious else "✗ FAIL"
        if is_susp == expected_suspicious:
            passed += 1
        else:
            failed += 1

        print(f"{status} | '{text}'")
        print(f"       | {description}")
        print(f"       | Expected: {expected_suspicious}, Got: {is_susp}")
        if reason:
            print(f"       | Reason: {reason}")
        print("-" * 80)

    print(f"\nResults: {passed} passed, {failed} failed out of {len(test_cases)} tests")

    if failed == 0:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {failed} test(s) failed")

    return failed == 0


if __name__ == '__main__':
    success = test_validation()
    sys.exit(0 if success else 1)
