from __future__ import annotations

import pytest

from isaaclab_pruning.sim.render_quality import CaptureQuality, apply_capture_quality, settings_readback


class FakeSettings:
    def __init__(self, ignore=None):
        self.values = {}
        self.ignore = ignore

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value):
        if key != self.ignore:
            self.values[key] = value


def test_profile_uses_offline_samples_not_legacy_realtime_light_samples():
    quality = CaptureQuality()
    settings = quality.carb_settings()
    assert settings["/rtx/rendermode"] == "PathTracing"
    assert settings["/rtx/pathtracing/spp"] == 32
    assert settings["/rtx/pathtracing/totalSpp"] == 64
    assert "/rtx/directLighting/sampledLighting/samplesPerPixel" not in settings
    assert quality.updates_per_capture == 4
    assert settings["/rtx/pathtracing/maxSamplesPerLaunch"] >= (1280 * 720 + 480 * 320 + 640 * 480) * 32


def test_profile_disables_ngx_dependencies_and_preserves_depth_aovs():
    settings = CaptureQuality().carb_settings()
    assert settings["/rtx/pathtracing/optixDenoiser/enabled"]
    assert settings["/rtx/pathtracing/optixDenoiser/blendFactor"] == 0.0
    for key in (
        "/rtx-transient/dldenoiser/enabled",
        "/rtx-transient/dlssg/enabled",
        "/rtx/pathtracing/optixDenoiser/AOV",
        "/rtx/post/motionblur/enabled",
        "/omni/replicator/captureMotionBlur",
    ):
        assert settings[key] is False
    assert settings["/rtx/post/aa/op"] == 0


def test_native_presentation_size_does_not_change_cv_sensor_grid():
    quality = CaptureQuality(overview_width=1920, total_samples=128)
    assert quality.overview_resolution == (1920, 1080)
    assert quality.wrist_resolution == (480, 320)
    assert quality.updates_per_capture == 6


@pytest.mark.parametrize(
    "kwargs",
    [
        {"overview_width": 960},
        {"overview_width": True},
        {"samples_per_tick": 0},
        {"samples_per_tick": 64},
        {"samples_per_tick": True},
        {"total_samples": 16},
        {"total_samples": 65},
        {"total_samples": 1024},
        {"optix_denoiser": 1},
    ],
)
def test_profile_rejects_unsupported_or_unbounded_capture_settings(kwargs):
    with pytest.raises(ValueError):
        CaptureQuality(**kwargs)


def test_settings_report_records_readback_not_only_requested_config():
    settings = FakeSettings()
    report = apply_capture_quality(settings, CaptureQuality())
    assert report["settings_match"]
    assert report["actual"] == report["requested"]
    settings.set("/rtx/rendermode", "RealTimePathTracing")
    later = settings_readback(settings, report["requested"])
    assert not later["settings_match"]
    assert later["mismatches"]["/rtx/rendermode"]["actual"] == "RealTimePathTracing"
    assert "not measured" in report["evidence_scope"]


def test_missing_carb_override_is_not_silently_reported_as_quality_enabled():
    with pytest.raises(RuntimeError, match="did not persist"):
        apply_capture_quality(FakeSettings(ignore="/rtx/rendermode"), CaptureQuality())


def test_raw_profile_can_disable_denoiser_for_matched_static_comparison():
    settings = FakeSettings()
    report = apply_capture_quality(settings, CaptureQuality(optix_denoiser=False))
    assert report["profile"] == "offline_pathtracing_raw"
    assert settings.get("/rtx/pathtracing/optixDenoiser/enabled") is False
