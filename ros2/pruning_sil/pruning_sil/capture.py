"""Read a recorded pruning capture. No ROS, no simulator, no licensed assets.

Everything here works on a plain directory of recorded JSON, PNG and NPY, so the
exporter, the parity harness and their tests can run without a ROS installation
and without the orchard meshes.

The one subtlety worth stating plainly is ``controller_source_frame_index``. The
recorded controller decided frame N's command from the image captured at frame
N-1, and the capture records that index per frame. Frame 0 carries -1, meaning no
image had been observed yet. Any replay that ignores this compares a decision
against the wrong image and will agree with the recording for the wrong reason.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

#: Frame 0 records this: the controller had not yet seen an image.
NO_SOURCE_FRAME = -1


class CaptureError(RuntimeError):
    """A capture is missing, inconsistent, or not the capture that was asked for."""


class Capture:
    """One recorded run: per-frame telemetry plus wrist RGB and optical-Z depth."""

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        report_path = self.directory / "report.json"
        frames_path = self.directory / "frames.json"
        for path in (report_path, frames_path):
            if not path.is_file():
                raise CaptureError(f"Capture is missing {path.name}: {self.directory}")
        self.report = json.loads(report_path.read_text(encoding="utf-8"))
        document = json.loads(frames_path.read_text(encoding="utf-8"))
        self.frames = document["frames"] if isinstance(document, dict) else document
        self.fps = float(document.get("fps", 10)) if isinstance(document, dict) else 10.0
        if not self.frames:
            raise CaptureError(f"Capture records no frames: {self.directory}")
        indexes = [int(frame["index"]) for frame in self.frames]
        if indexes != list(range(len(indexes))):
            raise CaptureError("Recorded frame indexes are not ordered and contiguous")

    def __len__(self):
        return len(self.frames)

    @property
    def job_id(self):
        return str(self.report.get("job_id", self.directory.name))

    @property
    def camera_matrix(self):
        """Wrist intrinsics as a 3x3 array, straight from the recorded report."""
        matrix = self.report.get("camera", {}).get("wrist_intrinsics")
        if matrix is None:
            raise CaptureError("Capture report has no wrist intrinsics")
        return np.asarray(matrix, dtype=float)

    @property
    def wrist_resolution(self):
        width, height = self.report.get("camera", {}).get("wrist_resolution", (0, 0))
        return int(width), int(height)

    def rgb(self, index):
        """Wrist RGB for one frame, as uint8 HxWx3.

        OpenCV is used rather than Pillow because the tracker already requires it,
        so the replay adds no dependency the controller did not already have.
        """
        import cv2

        path = self.directory / "frames" / f"wrist_{int(index):05d}.png"
        if not path.is_file():
            raise CaptureError(f"Missing wrist image: {path}")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise CaptureError(f"Wrist image could not be decoded: {path}")
        # imread yields BGR; the recorded controller was given RGB.
        return np.ascontiguousarray(image[:, :, ::-1], dtype=np.uint8)

    def depth(self, index):
        """Optical-Z depth in metres, float32 HxW. Non-hits stay non-finite."""
        path = self.directory / "frames" / f"depth_{int(index):05d}.npy"
        if not path.is_file():
            raise CaptureError(f"Missing depth array: {path}")
        return np.load(path).astype(np.float32, copy=False)

    def has_images(self):
        return (self.directory / "frames").is_dir()

    def world_from_optical(self, index):
        """Camera pose in world as a 4x4, from the recorded wrist pose."""
        frame = self.frames[int(index)]
        rotation = frame.get("wrist_rotation_w_ros")
        position = frame.get("wrist_position_w_m")
        if rotation is None or position is None:
            raise CaptureError(f"Frame {index} has no recorded wrist pose")
        transform = np.eye(4, dtype=float)
        transform[:3, :3] = np.asarray(rotation, dtype=float)
        transform[:3, 3] = np.asarray(position, dtype=float)
        return transform

    def source_frame_index(self, index):
        """Which image the recorded controller used for this frame's decision."""
        value = self.frames[int(index)].get("controller_source_frame_index", NO_SOURCE_FRAME)
        return int(value)

    def recorded_decision(self, index):
        return self.frames[int(index)].get("visual_servo_decision") or {}

    def recorded_measurement(self, index):
        return (self.frames[int(index)].get("live_vision") or {}).get("measurement") or {}

    def tof(self, index, side):
        """One 8x8 grid in metres plus its validity mask. Invalid stays non-finite."""
        if side not in ("left", "right"):
            raise CaptureError(f"Unknown time-of-flight side: {side}")
        frame = self.frames[int(index)]
        grid = frame.get(f"tof_{side}_m")
        valid = frame.get(f"tof_{side}_valid")
        if grid is None or valid is None:
            raise CaptureError(f"Frame {index} has no {side} time-of-flight grid")
        metres = np.asarray([[np.nan if cell is None else float(cell) for cell in row] for row in grid], dtype=np.float32)
        mask = np.asarray(valid, dtype=bool)
        if metres.shape != mask.shape:
            raise CaptureError("Time-of-flight grid and validity mask disagree in shape")
        # An invalid zone must never present a usable range, whatever was stored.
        metres[~mask] = np.nan
        return metres, mask

    def seed_pixel(self):
        """The one supplied pixel. Everything after this comes from tracking."""
        measurement = self.recorded_measurement(0)
        pixel = measurement.get("pixel_xy")
        if pixel is None:
            for index in range(len(self.frames)):
                pixel = self.recorded_measurement(index).get("pixel_xy")
                if pixel is not None:
                    break
        if pixel is None:
            raise CaptureError("Capture records no tracked pixel to seed from")
        return tuple(float(value) for value in pixel)

    def target(self):
        """Branch identity, axis and radius, from scene metadata rather than vision."""
        scene = self.report.get("blender_scene", {})
        target = scene.get("target", {})
        if not target:
            raise CaptureError("Capture report has no selected target")
        return {
            "id": target.get("id"),
            "axis_w": target.get("axis_w"),
            "radius_m": target.get("radius_m"),
            "identity_source": target.get("identity_source"),
        }

    def home_pose(self):
        """The recorded home pose, taken from the first frame's tool pose."""
        pose = self.frames[0].get("tool_pose_wxyz")
        if pose is None:
            raise CaptureError("Capture records no initial tool pose")
        return np.asarray(pose, dtype=float)

    def photometric_normalization(self):
        return str(self.report.get("photometric_normalization", "raw"))

    def stamp_ns(self, index):
        """Capture time in nanoseconds, from the recorded frame time."""
        return int(round(float(self.frames[int(index)]["time_s"]) * 1e9))
