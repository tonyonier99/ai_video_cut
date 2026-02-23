"""Global model holders and status tracking for the backend."""

import os

# Model Status
model_status = {"progress": 100, "status": "initializing", "message": "正準備下載模型..."}
whisper_model = None
CURRENT_WHISPER_NAME = "turbo"

# Job Status
current_job_status = {"progress": 0, "message": "Idle", "step": "idle"}

# MediaPipe / Face Detection
face_detector = None
mp_face_detection = None
face_mesh = None
df_model = None
df_state = None
face_cascade = None


def init_face_detection():
    """Initialize MediaPipe face detection with fallbacks."""
    global face_detector, mp_face_detection

    try:
        from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions
        from mediapipe.tasks.python import BaseOptions
        import urllib.request

        model_path = "blaze_face_short_range.tflite"
        if not os.path.exists(model_path):
            print("📥 Downloading MediaPipe Face Detection model...")
            url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
            urllib.request.urlretrieve(url, model_path)

        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            min_detection_confidence=0.5
        )
        face_detector = FaceDetector.create_from_options(options)
        print("✅ MediaPipe Face Detection Ready (Tasks API)")
    except Exception as e:
        print(f"⚠️ MediaPipe Tasks API Init Failed: {e}")
        try:
            import mediapipe as mp
            if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_detection'):
                mp_face_detection = mp.solutions.face_detection
                face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
                print("✅ MediaPipe Face Detection Ready (Legacy API)")
            else:
                raise ImportError("No solutions API available")
        except Exception as e2:
            print(f"⚠️ MediaPipe Legacy Init Also Failed: {e2}")
            try:
                import cv2
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                face_detector = "opencv_fallback"
                print("⚠️ Used OpenCV Haar Classifier as fallback")
            except Exception:
                face_detector = None


def init_face_mesh():
    """Initialize Face Mesh for Lip Movement Detection."""
    global face_mesh
    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✅ MediaPipe Face Mesh Ready (Lip Movement Detection)")
    except Exception as e:
        print(f"⚠️ Face Mesh Init Failed: {e}")
        face_mesh = None


def init_studio_sound():
    """Initialize Studio Sound (DFN3)."""
    global df_model, df_state
    try:
        from df.enhance import init_df
        df_model, df_state, _ = init_df()
        print("✅ Studio Sound (DFN3) Ready")
    except Exception as e:
        print(f"⚠️ DFN3 Init Failed: {e}")
        df_model = None


def init_all_models():
    """Initialize all ML models."""
    init_face_detection()
    init_face_mesh()
    init_studio_sound()
