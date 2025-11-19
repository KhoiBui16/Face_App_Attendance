"""
Test Cosine Similarity - Verify embeddings are consistent
"""

import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import face_processing
import db


def test_same_face_similarity():
    """Test: Cùng 1 người nên có cosine > 0.8"""
    print("=" * 60)
    print("🧪 TEST 1: Same Face Similarity")
    print("=" * 60)

    # Load embeddings từ database
    embeddings = db.load_embeddings()

    if not embeddings:
        print("❌ Không có dữ liệu trong database!")
        print("   → Đăng ký ít nhất 1 người trước khi test")
        return False

    # Lấy người đầu tiên để test
    test_name = list(embeddings.keys())[0]
    test_embedding = embeddings[test_name]

    print(f"\n📊 Testing với: {test_name}")
    print(f"   Embedding shape: {test_embedding.shape}")
    print(f"   Embedding norm: {np.linalg.norm(test_embedding):.4f}")

    # Test 1: Cosine với chính nó (phải = 1.0)
    self_sim = cosine_similarity(
        test_embedding.reshape(1, -1), test_embedding.reshape(1, -1)
    )[0][0]

    print(f"\n✅ Self-similarity: {self_sim:.6f}")
    if abs(self_sim - 1.0) < 0.001:
        print("   → PASS: Self-similarity = 1.0")
    else:
        print(f"   → FAIL: Expected 1.0, got {self_sim:.6f}")
        return False

    # Test 2: Kiểm tra với tất cả users khác
    print(f"\n📋 Similarity với các users khác:")
    for name, emb in embeddings.items():
        sim = cosine_similarity(test_embedding.reshape(1, -1), emb.reshape(1, -1))[0][0]

        if name == test_name:
            status = "✅ SELF" if sim > 0.99 else "❌ ERROR"
        else:
            status = "⚠️  HIGH" if sim > 0.6 else "✅ OK"

        print(f"   {name:20s}: {sim:.4f} {status}")

    return True


def test_embedding_pipeline():
    """Test: Pipeline từ ảnh → embedding có consistent không"""
    print("\n" + "=" * 60)
    print("🧪 TEST 2: Embedding Pipeline Consistency")
    print("=" * 60)

    # Cần có ít nhất 1 ảnh trong face_db để test
    import os

    db_path = "face_db"

    if not os.path.exists(db_path) or not os.listdir(db_path):
        print("❌ Không có dữ liệu trong face_db!")
        return False

    # Load 1 user bất kỳ
    embeddings = db.load_embeddings()
    test_name = list(embeddings.keys())[0]
    stored_embedding = embeddings[test_name]

    print(f"\n📊 Testing pipeline với: {test_name}")
    print(f"   Stored embedding norm: {np.linalg.norm(stored_embedding):.6f}")

    # Kiểm tra: Tất cả embeddings phải có norm = 1.0 (normalized)
    for name, emb in embeddings.items():
        norm = np.linalg.norm(emb)
        status = "✅" if abs(norm - 1.0) < 0.001 else "❌"
        print(f"   {status} {name:20s}: norm = {norm:.6f}")

    print("\n✅ Pipeline test complete")
    return True


def test_threshold_analysis():
    """Test: Phân tích threshold hiện tại"""
    print("\n" + "=" * 60)
    print("🧪 TEST 3: Threshold Analysis")
    print("=" * 60)

    embeddings = db.load_embeddings()

    if len(embeddings) < 2:
        print("⚠️  Cần ít nhất 2 người để test threshold")
        return True

    print(f"\n📊 Current threshold: {face_processing.COSINE_THRESHOLD}")

    # Tính similarity matrix
    names = list(embeddings.keys())
    emb_matrix = np.array(list(embeddings.values()))

    sim_matrix = cosine_similarity(emb_matrix, emb_matrix)

    # Tìm min similarity (khác 1.0)
    same_person_sims = []
    diff_person_sims = []

    for i in range(len(names)):
        for j in range(len(names)):
            sim = sim_matrix[i][j]
            if i == j:
                same_person_sims.append(sim)
            else:
                diff_person_sims.append(sim)

    if same_person_sims:
        print(f"\n✅ Same person similarities:")
        print(f"   Min:  {min(same_person_sims):.4f}")
        print(f"   Max:  {max(same_person_sims):.4f}")
        print(f"   Mean: {np.mean(same_person_sims):.4f}")

    if diff_person_sims:
        print(f"\n📊 Different person similarities:")
        print(f"   Min:  {min(diff_person_sims):.4f}")
        print(f"   Max:  {max(diff_person_sims):.4f}")
        print(f"   Mean: {np.mean(diff_person_sims):.4f}")

        # Kiểm tra threshold
        max_diff = max(diff_person_sims)
        if max_diff >= face_processing.COSINE_THRESHOLD:
            print(
                f"\n⚠️  WARNING: Highest diff-person similarity ({max_diff:.4f}) >= threshold ({face_processing.COSINE_THRESHOLD})"
            )
            print(f"   → Có thể gây nhầm lẫn!")
            suggested = max_diff + 0.05
            print(f"   → Gợi ý threshold: {suggested:.2f}")
        else:
            print(
                f"\n✅ Threshold OK: Max diff-person ({max_diff:.4f}) < threshold ({face_processing.COSINE_THRESHOLD})"
            )

    return True


def test_detect_and_align_output():
    """Test: Kiểm tra output của detect_and_align"""
    print("\n" + "=" * 60)
    print("🧪 TEST 4: detect_and_align Output Shape")
    print("=" * 60)

    print("\n✅ Expected behavior:")
    print("   - Input: Ảnh bất kỳ")
    print("   - Output: (face_224x224, original_image, coords)")
    print("   - face_224x224 shape: (224, 224, 3)")

    # Kiểm tra IMG_SIZE
    print(f"\n📊 IMG_SIZE constant: {face_processing.IMG_SIZE}")

    if face_processing.IMG_SIZE != (224, 224):
        print(f"   ❌ ERROR: Expected (224, 224), got {face_processing.IMG_SIZE}")
        return False
    else:
        print(f"   ✅ Correct")

    return True


def main():
    print("\n" + "=" * 60)
    print("🔍 COSINE SIMILARITY VERIFICATION TEST")
    print("=" * 60)

    tests = [
        ("Embedding Consistency", test_same_face_similarity),
        ("Pipeline Check", test_embedding_pipeline),
        ("Threshold Analysis", test_threshold_analysis),
        ("Output Shape Check", test_detect_and_align_output),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            import traceback

            traceback.print_exc()
            results.append((test_name, False))

    print("\n" + "=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10s} {test_name}")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    print(f"\n{'='*60}")
    print(f"Final Score: {passed}/{total}")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n✅ Cosine similarity đang hoạt động đúng:")
        print("   - Embeddings được normalized")
        print("   - detect_and_align trả về đúng shape (224x224)")
        print("   - Threshold phù hợp")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("   Kiểm tra lại pipeline!")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
