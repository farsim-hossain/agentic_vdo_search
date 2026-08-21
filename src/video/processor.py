import cv2
import base64
import io
import math
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from PIL import Image, ImageDraw, ImageFont
from scenedetect import detect, ContentDetector

@dataclass
class Keyframe:
    frame_index: int
    pts_seconds: float
    timestamp_str: str
    image: Image.Image  # PIL Image in RGB mode

@dataclass
class SceneShot:
    shot_id: int
    start_seconds: float
    end_seconds: float
    start_timestamp: str
    end_timestamp: str
    keyframes: List[Keyframe] = field(default_factory=list)

def format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

class VideoProcessor:
    def __init__(self, min_scene_len_sec: float = 1.5, max_keyframes_per_shot: int = 4):
        self.min_scene_len_sec = min_scene_len_sec
        self.max_keyframes_per_shot = max_keyframes_per_shot

    def process_video(self, video_path: str) -> List[SceneShot]:
        """Detect scenes and extract representative keyframes per shot."""
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        # Detect scenes using PySceneDetect
        scene_list = []
        try:
            scene_list = detect(str(path), ContentDetector())
        except Exception:
            scene_list = []

        shots: List[SceneShot] = []
        if scene_list:
            for idx, (start_time, end_time) in enumerate(scene_list):
                start_sec = start_time.get_seconds()
                end_sec = end_time.get_seconds()
                shots.append(
                    SceneShot(
                        shot_id=idx,
                        start_seconds=start_sec,
                        end_seconds=end_sec,
                        start_timestamp=format_timestamp(start_sec),
                        end_timestamp=format_timestamp(end_sec),
                    )
                )
        else:
            # Fallback: divide video into uniform fixed intervals (e.g. 5 seconds per shot)
            interval_sec = 5.0
            num_shots = max(1, math.ceil(duration / interval_sec))
            for idx in range(num_shots):
                start_sec = idx * interval_sec
                end_sec = min(duration, (idx + 1) * interval_sec)
                shots.append(
                    SceneShot(
                        shot_id=idx,
                        start_seconds=start_sec,
                        end_seconds=end_sec,
                        start_timestamp=format_timestamp(start_sec),
                        end_timestamp=format_timestamp(end_sec),
                    )
                )

        # Extract keyframes for each shot
        for shot in shots:
            shot.keyframes = self._extract_keyframes_for_shot(cap, fps, shot)

        cap.release()
        return shots

    def _extract_keyframes_for_shot(self, cap: cv2.VideoCapture, fps: float, shot: SceneShot) -> List[Keyframe]:
        start_frame = int(shot.start_seconds * fps)
        end_frame = int(shot.end_seconds * fps)
        frame_count = max(1, end_frame - start_frame)

        num_keyframes = min(self.max_keyframes_per_shot, frame_count)
        if num_keyframes == 1:
            frame_indices = [start_frame + frame_count // 2]
        else:
            step = (frame_count - 1) / (num_keyframes - 1)
            frame_indices = [int(start_frame + i * step) for i in range(num_keyframes)]

        keyframes = []
        for f_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            pts_sec = f_idx / fps
            ts_str = format_timestamp(pts_sec)

            keyframes.append(
                Keyframe(
                    frame_index=f_idx,
                    pts_seconds=pts_sec,
                    timestamp_str=ts_str,
                    image=pil_img,
                )
            )

        return keyframes

    @staticmethod
    def create_storyboard(keyframes: List[Keyframe], tile_size: Tuple[int, int] = (320, 180)) -> Image.Image:
        """Composite keyframe images into a single timestamped storyboard contact sheet."""
        if not keyframes:
            raise ValueError("No keyframes provided for storyboard creation")

        num_frames = len(keyframes)
        cols = 2 if num_frames <= 4 else 3
        rows = math.ceil(num_frames / cols)

        tile_w, tile_h = tile_size
        margin = 4
        canvas_w = cols * tile_w + (cols + 1) * margin
        canvas_h = rows * tile_h + (rows + 1) * margin

        canvas = Image.new("RGB", (canvas_w, canvas_h), color=(20, 20, 20))
        draw = ImageDraw.Draw(canvas)

        for i, kf in enumerate(keyframes):
            r = i // cols
            c = i % cols
            x = margin + c * (tile_w + margin)
            y = margin + r * (tile_h + margin)

            # Resize frame image
            resized = kf.image.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
            canvas.paste(resized, (x, y))

            # Draw timestamp badge
            badge_text = f"[{kf.timestamp_str}]"
            # Draw black box behind text for high legibility
            draw.rectangle([(x + 4, y + tile_h - 24), (x + 90, y + tile_h - 4)], fill=(0, 0, 0, 200))
            draw.text((x + 8, y + tile_h - 22), badge_text, fill=(255, 255, 255))

        return canvas

    @staticmethod
    def storyboard_to_base64(storyboard: Image.Image, format: str = "JPEG", quality: int = 85) -> str:
        """Convert a storyboard PIL image to base64 data URL string."""
        buf = io.BytesIO()
        storyboard.save(buf, format=format, quality=quality)
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/{format.lower()};base64,{b64_str}"
