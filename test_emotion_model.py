"""
Test script for new emotion and anti-spoof models
"""

import config
import face_processing
import cv2
import numpy as np


def test_model_loading():
    """Test if models load correctly"""
    print("=" * 60)
    print("MODEL LOADING TEST")
    print("=" * 60)

    print(f"\n1. Configuration Check:")
    print(f"   - Emotion Model: {config.EMOTION_MODEL_PATH}")
    print(f"   - Spoof Model: {config.SPOOF_MODEL_PATH}")
    print(f"   - Emotion Labels: {config.EMOTION_LABELS}")
    print(f"   - Emotion IMG Size: {config.EMOTION_IMG_SIZE}")
    print(f"   - Spoof IMG Size: {config.SPOOF_IMG_SIZE}")
    print(f"   - Spoof Threshold: {config.SPOOF_THRESHOLD}")

    print(f"\n2. Loading Models...")
    try:
        detector, embed_model, spoof_model, emotion_model = (
            face_processing.load_models()
        )

        if detector:
            print("   ✅ MTCNN Detector loaded")
        else:
            print("   ❌ MTCNN Detector failed")
            return False

        if embed_model:
            print("   ✅ Face Recognition Model loaded")
        else:
            print("   ❌ Face Recognition Model failed")
            return False

        if emotion_model:
            print("   ✅ Emotion Model loaded (ResNet50)")
            print(f"      - Input shape: {emotion_model.input_shape}")
            print(f"      - Output shape: {emotion_model.output_shape}")
        else:
            print("   ⚠️  Emotion Model not loaded")

        if spoof_model:
            print("   ✅ Anti-Spoof Model loaded (ResNet50)")
            print(f"      - Input shape: {spoof_model.input_shape}")
            print(f"      - Output shape: {spoof_model.output_shape}")
        else:
            print("   ⚠️  Anti-Spoof Model not loaded")

        return True

    except Exception as e:
        print(f"   ❌ Error loading models: {e}")
        return False


def test_emotion_labels():
    """Test emotion label mapping"""
    print("\n\n" + "=" * 60)
    print("EMOTION LABELS TEST")
    print("=" * 60)

    print(f"\nExpected Mapping:")
    expected = {
        "0": "Anger",
        "1": "Disgust",
        "2": "Fear",
        "3": "Happy",
        "4": "Sadness",
        "5": "Surprise",
        "6": "Neutral",
        "7": "Contempt",
    }

    for idx, label in expected.items():
        actual_label = config.EMOTION_LABELS[int(idx)]
        icon = config.EMOTION_ICONS.get(actual_label, "")
        match = "✅" if actual_label == label else "❌"
        print(f"   {match} {idx}: {label:10s} → {actual_label:10s} {icon}")

    # Check all labels have icons
    print(f"\nIcon Mapping:")
    all_have_icons = True
    for label in config.EMOTION_LABELS:
        icon = config.EMOTION_ICONS.get(label, "")
        has_icon = "✅" if icon else "❌"
        print(f"   {has_icon} {label:10s}: {icon}")
        if not icon:
            all_have_icons = False

    return all_have_icons


def test_emotion_prediction():
    """Test emotion prediction with dummy image"""
    print("\n\n" + "=" * 60)
    print("EMOTION PREDICTION TEST")
    print("=" * 60)

    _, _, _, emotion_model = face_processing.load_models()

    if emotion_model is None:
        print("\n⚠️  Emotion model not loaded, skipping test")
        return False

    # Create dummy face image
    dummy_face = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    print(f"\n1. Testing with dummy face image...")
    print(f"   - Input shape: {dummy_face.shape}")

    try:
        emotion_result = face_processing.detect_emotion(dummy_face)
        print(f"   ✅ Emotion detection successful")
        print(f"   - Result: {emotion_result}")

        # Check if result is valid
        if emotion_result != "N/A" and emotion_result != "Unknown":
            # Extract label from result (format: "Label Icon")
            parts = emotion_result.split()
            if len(parts) > 0:
                detected_label = parts[0]
                if detected_label in config.EMOTION_LABELS:
                    print(f"   ✅ Valid emotion label: {detected_label}")
                    return True
                else:
                    print(f"   ❌ Invalid emotion label: {detected_label}")
                    return False

        return True

    except Exception as e:
        print(f"   ❌ Emotion detection failed: {e}")
        return False


def test_model_output_shape():
    """Test if emotion model outputs correct shape"""
    print("\n\n" + "=" * 60)
    print("MODEL OUTPUT SHAPE TEST")
    print("=" * 60)

    _, _, _, emotion_model = face_processing.load_models()

    if emotion_model is None:
        print("\n⚠️  Emotion model not loaded, skipping test")
        return False

    import tensorflow as tf

    # Create test input
    test_input = np.random.rand(1, 224, 224, 3).astype("float32")
    test_input = tf.keras.applications.resnet.preprocess_input(test_input)

    print(f"\n1. Test Input:")
    print(f"   - Shape: {test_input.shape}")
    print(f"   - Dtype: {test_input.dtype}")

    try:
        output = emotion_model(test_input, training=False).numpy()

        print(f"\n2. Model Output:")
        print(f"   - Shape: {output.shape}")
        print(f"   - Dtype: {output.dtype}")
        print(f"   - Min value: {output.min():.6f}")
        print(f"   - Max value: {output.max():.6f}")
        print(f"   - Sum: {output.sum():.6f}")

        # Check if output has 8 classes (0-7)
        if output.shape[1] == 8:
            print(f"   ✅ Output shape correct: 8 classes (0-7)")

            # Show predictions for all classes
            print(f"\n3. Predictions for each class:")
            for i, label in enumerate(config.EMOTION_LABELS):
                prob = output[0][i]
                print(f"      {i}: {label:10s} = {prob:.6f}")

            return True
        else:
            print(f"   ❌ Output shape incorrect: expected 8, got {output.shape[1]}")
            return False

    except Exception as e:
        print(f"   ❌ Model prediction failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "EMOTION & ANTI-SPOOF MODEL TEST" + " " * 17 + "║")
    print("╚" + "═" * 58 + "╝")

    results = []

    # Test 1: Model Loading
    result1 = test_model_loading()
    results.append(("Model Loading", result1))

    # Test 2: Emotion Labels
    result2 = test_emotion_labels()
    results.append(("Emotion Labels", result2))

    # Test 3: Emotion Prediction
    result3 = test_emotion_prediction()
    results.append(("Emotion Prediction", result3))

    # Test 4: Model Output Shape
    result4 = test_model_output_shape()
    results.append(("Model Output Shape", result4))

    # Final Summary
    print("\n\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status:12s} - {test_name}")

    all_passed = all(result for _, result in results)

    if all_passed:
        print("\n🎉 ALL TESTS PASSED - Models ready for production!")
    else:
        print("\n⚠️  SOME TESTS FAILED - Check errors above")

    print("=" * 60)
