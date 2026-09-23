"""Reader and exporter behaviour, on a procedural fixture with no licensed assets."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fixture import HEIGHT, SEED_PIXEL, WIDTH, write_capture  # noqa: E402
from pruning_sil import topics as T  # noqa: E402
from pruning_sil.capture import NO_SOURCE_FRAME, Capture, CaptureError  # noqa: E402


@pytest.fixture
def capture(tmp_path):
    return Capture(write_capture(tmp_path / "capture", frames=6, release_frame=4))


def test_reader_reports_recorded_shape_and_seed(capture):
    assert len(capture) == 6
    assert capture.wrist_resolution == (WIDTH, HEIGHT)
    assert capture.seed_pixel() == SEED_PIXEL
    assert capture.camera_matrix.shape == (3, 3)
    assert capture.photometric_normalization() == "raw"


def test_frame_zero_records_no_source_image(capture):
    # The controller had seen nothing yet; a replay must not invent a command.
    assert capture.source_frame_index(0) == NO_SOURCE_FRAME
    assert capture.source_frame_index(3) == 2


def test_missing_capture_is_an_error_not_an_empty_result(tmp_path):
    with pytest.raises(CaptureError, match="missing report.json"):
        Capture(tmp_path / "absent")


def test_invalid_tof_zones_never_present_a_usable_range(capture):
    metres, mask = capture.tof(0, "left")
    assert metres.shape == (T.TOF_ROWS, T.TOF_COLS)
    assert mask.shape == metres.shape
    # Every invalid zone must be non-finite, whatever the recording stored.
    assert not np.isfinite(metres[~mask]).any()
    assert np.isfinite(metres[mask]).all()


def test_unknown_tof_side_is_refused(capture):
    with pytest.raises(CaptureError, match="Unknown time-of-flight side"):
        capture.tof(0, "middle")


def test_depth_is_metres_and_rgb_round_trips(capture):
    depth = capture.depth(0)
    assert depth.dtype == np.float32
    assert depth.shape == (HEIGHT, WIDTH)
    rgb = capture.rgb(0)
    assert rgb.shape == (HEIGHT, WIDTH, 3)
    assert rgb.dtype == np.uint8
    # The moving patch must actually differ from the background, or the fixture
    # is not exercising the tracker at all.
    assert rgb.std() > 1.0


def test_stamps_follow_recorded_capture_time_not_wall_clock(capture):
    assert capture.stamp_ns(0) == 100_000_000
    stamps = [capture.stamp_ns(index) for index in range(len(capture))]
    assert stamps == sorted(stamps)
    assert len(set(stamps)) == len(stamps)


def test_tof_layout_matches_the_real_labels_and_sizes(capture):
    (row_label, row_size, row_stride), (col_label, col_size, col_stride) = T.tof_layout_dims()
    # Labels and sizes copy vl53l8cx_msgs Row8x8 and Column8x8 exactly.
    assert (row_label, row_size) == ("row", 8)
    assert (col_label, col_size) == ("column", 8)
    # Strides deliberately follow the ROS MultiArrayDimension convention, which
    # is NOT the real message's convention. Both are recorded so a future bridge
    # translates rather than silently mismatching.
    assert (row_stride, col_stride) == T.TOF_STRIDES_ROS == (64, 8)
    assert T.TOF_STRIDES_REAL == (8, 1)


def test_every_published_topic_declares_a_message_type():
    from pruning_sil.export_rosbag2 import TOPIC_TYPES

    published = {
        T.TOPIC_WRIST_IMAGE, T.TOPIC_WRIST_DEPTH, T.TOPIC_WRIST_CAMERA_INFO,
        T.TOPIC_TOF_LEFT, T.TOPIC_TOF_RIGHT, T.TOPIC_TOF_LEFT_VALID, T.TOPIC_TOF_RIGHT_VALID,
        T.TOPIC_JOINT_STATES, T.TOPIC_TOOL_POSE, T.TOPIC_RECORDED_COMMAND, T.TOPIC_RECORDED_RELEASE,
    }
    assert published <= set(TOPIC_TYPES)


def test_real_rig_mapping_names_a_type_for_every_mapped_topic():
    for topic, mapping in T.REAL_RIG_MAPPING.items():
        assert topic.startswith("/"), topic
        assert mapping["real_type"] and mapping["sil_type"]
    # The proposed command must be documented as advisory against MoveIt Servo.
    assert "TwistStamped" in T.REAL_RIG_MAPPING[T.TOPIC_PROPOSED_COMMAND]["real_type"]


def test_the_package_states_what_it_does_not_claim():
    joined = " ".join(T.NOT_CLAIMED).lower()
    assert "hardware-in-the-loop" in joined
    assert "never actuated" in joined
