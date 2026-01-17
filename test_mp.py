
try:
    import mediapipe as mp
    print("1. mp.solutions:", mp.solutions)
except Exception as e:
    print("1. Failed:", e)

try:
    from mediapipe import solutions
    print("2. solutions imported")
except Exception as e:
    print("2. Failed:", e)

try:
    import mediapipe.python.solutions.face_detection
    print("3. deep import success")
except Exception as e:
    print("3. deep import failed:", e)

