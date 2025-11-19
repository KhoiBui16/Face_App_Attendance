"""
Test script for camera display improvements
"""

import cv2
import config


def test_camera_resolution():
    """Test camera resolution settings"""
    print("=" * 60)
    print("CAMERA DISPLAY OPTIMIZATION TEST")
    print("=" * 60)

    print("\n1. Testing Camera Resolution Settings:")
    print(f"   - Camera Index: {config.CAMERA_INDEX}")
    print(f"   - Target Resolution: {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}")
    print(f"   - Target FPS: {config.CAMERA_FPS}")
    print(f"   - Display Height: {config.DISPLAY_HEIGHT}")

    # Try to open camera
    cap = cv2.VideoCapture(config.CAMERA_INDEX)

    if not cap.isOpened():
        print("\n❌ FAIL: Cannot open camera")
        return False

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

    # Read actual settings
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"\n2. Actual Camera Settings:")
    print(f"   - Resolution: {actual_width}x{actual_height}")
    print(f"   - FPS: {actual_fps}")

    # Read a test frame
    ret, frame = cap.read()

    if not ret:
        print("\n❌ FAIL: Cannot read frame")
        cap.release()
        return False

    h, w = frame.shape[:2]
    print(f"\n3. Frame Properties:")
    print(f"   - Frame size: {w}x{h}")
    print(f"   - Frame dtype: {frame.dtype}")

    # Calculate display size
    display_height = config.DISPLAY_HEIGHT
    display_width = int(w * display_height / h)

    print(f"\n4. Display Sizing:")
    print(f"   - Display resolution: {display_width}x{display_height}")
    print(f"   - Scale ratio: {display_height/h:.2f}")

    # Test resize
    try:
        display_resized = cv2.resize(frame, (display_width, display_height))
        print(f"   - Resized shape: {display_resized.shape}")
        print("   ✅ Resize successful")
    except Exception as e:
        print(f"   ❌ Resize failed: {e}")
        cap.release()
        return False

    cap.release()

    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)

    checks = []

    # Check resolution
    if actual_width >= 640 and actual_height >= 480:
        print("✅ Camera resolution: GOOD (>= 640x480)")
        checks.append(True)
    else:
        print(f"⚠️  Camera resolution: LOW ({actual_width}x{actual_height})")
        checks.append(False)

    # Check FPS
    if actual_fps >= 20:
        print(f"✅ Camera FPS: GOOD ({actual_fps} fps)")
        checks.append(True)
    else:
        print(f"⚠️  Camera FPS: LOW ({actual_fps} fps)")
        checks.append(False)

    # Check display size
    if display_width > 0 and display_height > 0:
        print(f"✅ Display sizing: VALID ({display_width}x{display_height})")
        checks.append(True)
    else:
        print("❌ Display sizing: INVALID")
        checks.append(False)

    # Check aspect ratio preservation
    original_aspect = w / h
    display_aspect = display_width / display_height
    aspect_diff = abs(original_aspect - display_aspect)

    if aspect_diff < 0.01:
        print(f"✅ Aspect ratio: PRESERVED ({original_aspect:.2f})")
        checks.append(True)
    else:
        print(
            f"⚠️  Aspect ratio: CHANGED ({original_aspect:.2f} → {display_aspect:.2f})"
        )
        checks.append(False)

    print("\n" + "=" * 60)

    if all(checks):
        print("🎉 ALL TESTS PASSED!")
        return True
    else:
        print("⚠️  SOME TESTS FAILED - Check warnings above")
        return False


def test_performance():
    """Test frame processing performance"""
    print("\n\n" + "=" * 60)
    print("PERFORMANCE TEST")
    print("=" * 60)

    cap = cv2.VideoCapture(config.CAMERA_INDEX)

    if not cap.isOpened():
        print("\n❌ Cannot open camera")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

    import time

    frame_count = 0
    start_time = time.time()

    print("\nProcessing 30 frames...")

    while frame_count < 30:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # Simulate processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Simulate resize for display
        display_height = config.DISPLAY_HEIGHT
        display_width = int(w * display_height / h)
        display_resized = cv2.resize(rgb_frame, (display_width, display_height))

        frame_count += 1

    elapsed = time.time() - start_time
    fps = frame_count / elapsed

    cap.release()

    print(f"\nResults:")
    print(f"   - Frames processed: {frame_count}")
    print(f"   - Time elapsed: {elapsed:.2f}s")
    print(f"   - FPS achieved: {fps:.1f}")
    print(f"   - Frame time: {1000/fps:.1f}ms")

    if fps >= 25:
        print("\n✅ Performance: EXCELLENT (>= 25 FPS)")
        return True
    elif fps >= 15:
        print("\n✅ Performance: GOOD (>= 15 FPS)")
        return True
    else:
        print(f"\n⚠️  Performance: SLOW ({fps:.1f} FPS)")
        return False


if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "CAMERA DISPLAY OPTIMIZATION TEST" + " " * 16 + "║")
    print("╚" + "═" * 58 + "╝")

    test1_result = test_camera_resolution()
    test2_result = test_performance()

    print("\n\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    if test1_result:
        print("✅ Camera Resolution Test: PASSED")
    else:
        print("❌ Camera Resolution Test: FAILED")

    if test2_result:
        print("✅ Performance Test: PASSED")
    else:
        print("❌ Performance Test: FAILED")

    if test1_result and test2_result:
        print("\n🎉 ALL TESTS PASSED - Camera display is optimized!")
    else:
        print("\n⚠️  SOME TESTS FAILED - Check results above")

    print("=" * 60)
