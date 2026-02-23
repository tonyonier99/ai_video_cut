"""Video analysis services: speaker detection and smart reframing."""

import os
import uuid

import cv2
import numpy as np
import mediapipe as mp

from backend import models
from backend.config import UPLOAD_DIR


def detect_speaker_segments(clip, segment_duration=1.0):
    """
    Analyzes video clip and returns time-segmented speaker positions.
    Returns: list of {start, end, faceCenterX} dictionaries.
    """
    w, h = clip.size
    total_duration = clip.duration
    segments = []

    face_mesh = models.face_mesh
    face_detector = models.face_detector

    if face_mesh is None and (face_detector is None or not hasattr(face_detector, 'detect')):
        return [{"start": 0, "end": total_duration, "faceCenterX": 0.5}]

    local_face_mesh = face_mesh

    temp_track_filename = f"temp_speaker_{uuid.uuid4()}.mp4"
    temp_track_path = os.path.join(UPLOAD_DIR, temp_track_filename)

    frame_data = []

    try:
        clip.write_videofile(temp_track_path, codec="libx264", preset="ultrafast", audio=False, logger=None)
        cap = cv2.VideoCapture(temp_track_path)

        if not cap.isOpened():
            return [{"start": 0, "end": total_duration, "faceCenterX": 0.5}]

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        frame_step = max(1, int(fps * 0.05))

        current_frame = 0
        frames_processed = 0
        faces_found = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = current_frame / fps

            if current_frame % frame_step == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_h, frame_w = rgb_frame.shape[:2]
                frame_faces = {}

                if local_face_mesh:
                    mesh_results = local_face_mesh.process(rgb_frame)
                    if mesh_results.multi_face_landmarks:
                        for face_landmarks in mesh_results.multi_face_landmarks:
                            try:
                                upper_lip = face_landmarks.landmark[13]
                                lower_lip = face_landmarks.landmark[14]
                                nose = face_landmarks.landmark[1]
                                lip_aperture = abs(lower_lip.y - upper_lip.y)
                                face_center_x_px = nose.x * frame_w
                                bucket = int(face_center_x_px / 100)
                                frame_faces[bucket] = (face_center_x_px, lip_aperture)
                            except (IndexError, AttributeError):
                                pass

                elif face_detector and hasattr(face_detector, 'detect'):
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    res = face_detector.detect(mp_image)
                    if res.detections:
                        for det in res.detections:
                            bbox = det.bounding_box
                            fx = bbox.origin_x + bbox.width / 2
                            bucket = int(fx / 100)
                            frame_faces[bucket] = (fx, 0)

                if frame_faces:
                    frame_data.append((current_time, frame_faces))
                    faces_found += 1
                frames_processed += 1

            current_frame += 1

        cap.release()
        print(f"🔍 Speaker Detection: Processed {frames_processed} frames, found faces in {faces_found} frames")
    except Exception as e:
        print(f"⚠️ Speaker Segment Analysis Failed: {e}")
        return [{"start": 0, "end": total_duration, "faceCenterX": 0.5}]
    finally:
        if os.path.exists(temp_track_path):
            os.remove(temp_track_path)

    if not frame_data:
        return [{"start": 0, "end": total_duration, "faceCenterX": 0.5}]

    # Process frame_data to find speaker at each time window
    window_results = []

    window_start = 0
    while window_start < total_duration:
        window_end = min(window_start + segment_duration, total_duration)
        window_frames = [(t, faces) for t, faces in frame_data if window_start <= t < window_end]

        if not window_frames:
            window_start = window_end
            continue

        bucket_lip_data = {}
        bucket_x_data = {}
        bucket_count = {}

        for _, faces in window_frames:
            for bucket, (cx, lip) in faces.items():
                if bucket not in bucket_lip_data:
                    bucket_lip_data[bucket] = []
                    bucket_x_data[bucket] = []
                    bucket_count[bucket] = 0
                bucket_lip_data[bucket].append(lip)
                bucket_x_data[bucket].append(cx)
                bucket_count[bucket] += 1

        best_bucket = None
        max_var = -1
        has_lip_data = any(sum(lips) > 0 for lips in bucket_lip_data.values())

        if has_lip_data:
            for bucket, lips in bucket_lip_data.items():
                if len(lips) >= 2:
                    var = np.var(lips)
                    if var > max_var:
                        max_var = var
                        best_bucket = bucket
        else:
            max_count = 0
            for bucket, count in bucket_count.items():
                if count > max_count:
                    max_count = count
                    best_bucket = bucket

        if best_bucket is not None:
            speaker_x = np.mean(bucket_x_data[best_bucket])
        else:
            all_x = [x for xs in bucket_x_data.values() for x in xs]
            speaker_x = np.mean(all_x) if all_x else w / 2

        window_results.append((window_start, window_end, best_bucket, speaker_x))
        window_start = window_end

    # Merge with stickiness & switch confirmation
    merged_segments = []
    STICKINESS_DIST = 0.10
    confirm_count = 0

    for wr in window_results:
        ws, we, bucket, sx = wr
        norm_x = sx / w

        if not merged_segments:
            merged_segments.append({"start": ws, "end": we, "faceCenterX": norm_x, "bucket": bucket})
            continue

        prev = merged_segments[-1]
        dist = abs(norm_x - prev["faceCenterX"])
        is_different = (bucket != prev["bucket"] and dist >= STICKINESS_DIST)

        if not is_different:
            prev["end"] = we
            prev["faceCenterX"] = (prev["faceCenterX"] * 0.6) + (norm_x * 0.4)
            confirm_count = 0
        else:
            confirm_count += 1
            if confirm_count >= 2:
                merged_segments.append({
                    "start": ws, "end": we,
                    "faceCenterX": norm_x, "bucket": bucket
                })
                confirm_count = 0
            else:
                prev["end"] = we

    # Second pass: Sandwich & tiny segment filtering
    final_merged = []
    MIN_STABLE_DUR = 1.2

    i = 0
    while i < len(merged_segments):
        seg = merged_segments[i]
        dur = seg["end"] - seg["start"]

        if i > 0 and i < len(merged_segments) - 1:
            prev_seg = final_merged[-1]
            next_seg = merged_segments[i + 1]
            pos_dist = abs(prev_seg["faceCenterX"] - next_seg["faceCenterX"])
            if dur < MIN_STABLE_DUR and pos_dist < 0.15:
                prev_seg["end"] = next_seg["end"]
                i += 2
                continue

        if final_merged and (dur < 0.8):
            final_merged[-1]["end"] = seg["end"]
        else:
            final_merged.append(seg)
        i += 1

    segments = [
        {"start": s["start"], "end": s["end"], "faceCenterX": s["faceCenterX"]}
        for s in final_merged
    ]

    print(f"🎤 Speaker Segments: Detected {len(segments)} segments")
    for idx, seg in enumerate(segments[:5]):
        print(f"   Segment {idx+1}: {seg['start']:.1f}s - {seg['end']:.1f}s @ X={seg['faceCenterX']:.2f}")
    if len(segments) > 5:
        print(f"   ... and {len(segments) - 5} more segments")

    return segments if segments else [{"start": 0, "end": total_duration, "faceCenterX": 0.5}]


def apply_smart_reframing(
    clip, aspect_ratio, face_tracking, vertical_mode,
    viz_tracking="false", track_zoom=1.5, track_weight=5.0,
    track_stickiness=2.0, min_shot_duration=2.0
):
    """
    Simplified Reframing for Preview:
    Detects face in the middle of the clip and static crops to it.
    """
    if str(vertical_mode).lower() != "true":
        pass  # Proceed to tracking logic below

    w, h = clip.size
    target_ratio = 9 / 16
    target_w = int(h * target_ratio)
    target_h = h

    center_x = w / 2
    face_detector = models.face_detector
    face_mesh = models.face_mesh

    if str(face_tracking).lower() == "true" and face_detector:
        temp_track_filename = f"temp_track_{uuid.uuid4()}.mp4"
        temp_track_path = os.path.join(UPLOAD_DIR, temp_track_filename)

        face_lip_data = {}

        try:
            clip.write_videofile(temp_track_path, codec="libx264", preset="ultrafast", audio=False, logger=None)
            cap = cv2.VideoCapture(temp_track_path)

            if not cap.isOpened():
                print("⚠️ Could not open temp tracking file.")
            else:
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 30
                frame_step = int(fps * 0.2)
                if frame_step < 1:
                    frame_step = 1

                current_frame = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if current_frame % frame_step == 0:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_h, frame_w = rgb_frame.shape[:2]

                        if face_mesh:
                            mesh_results = face_mesh.process(rgb_frame)
                            if mesh_results.multi_face_landmarks:
                                for face_landmarks in mesh_results.multi_face_landmarks:
                                    try:
                                        upper_lip = face_landmarks.landmark[13]
                                        lower_lip = face_landmarks.landmark[14]
                                        nose = face_landmarks.landmark[1]
                                        lip_aperture = abs(lower_lip.y - upper_lip.y)
                                        face_center_x_px = nose.x * frame_w
                                        bucket = int(face_center_x_px / 100)
                                        if bucket not in face_lip_data:
                                            face_lip_data[bucket] = []
                                        face_lip_data[bucket].append((face_center_x_px, lip_aperture))
                                    except (IndexError, AttributeError):
                                        pass

                        elif hasattr(face_detector, 'detect'):
                            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                            res = face_detector.detect(mp_image)
                            if res.detections:
                                for det in res.detections:
                                    bbox = det.bounding_box
                                    fx = bbox.origin_x + bbox.width / 2
                                    bucket = int(fx / 100)
                                    if bucket not in face_lip_data:
                                        face_lip_data[bucket] = []
                                    face_lip_data[bucket].append((fx, 0))

                    current_frame += 1

                cap.release()

        except Exception as e:
            print(f"⚠️ Speaker Tracking Failed: {e}")
        finally:
            if os.path.exists(temp_track_path):
                os.remove(temp_track_path)

        if face_lip_data:
            speaker_bucket = None
            max_variance = -1

            for bucket, data in face_lip_data.items():
                if len(data) >= 3:
                    apertures = [d[1] for d in data]
                    variance = np.var(apertures)
                    if variance > max_variance:
                        max_variance = variance
                        speaker_bucket = bucket

            if speaker_bucket is not None and max_variance > 0.0001:
                speaker_x_positions = [d[0] for d in face_lip_data[speaker_bucket]]
                center_x = sum(speaker_x_positions) / len(speaker_x_positions)
                print(f"🎤 Speaker Detected: Face at X={int(center_x)} (Lip Variance={max_variance:.6f})")
            else:
                all_x = [d[0] for data in face_lip_data.values() for d in data]
                if all_x:
                    center_x = sum(all_x) / len(all_x)
                print(f"👁️ No clear speaker, using average face position: X={int(center_x)}")
        else:
            print("👁️ No faces detected, using center.")

    # Calculate Crop Coords
    x1 = max(0, int(center_x - target_w / 2))
    if x1 + target_w > w:
        x1 = w - target_w

    if str(vertical_mode).lower() == "true":
        from moviepy.video.fx.all import crop
        cropped_clip = crop(clip, x1=x1, y1=0, width=target_w, height=target_h)
        return (cropped_clip, float(center_x / w))
    else:
        return (clip, float(center_x / w))
