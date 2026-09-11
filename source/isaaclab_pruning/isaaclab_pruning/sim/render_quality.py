"""Explicit, NGX-independent offline RGB capture settings for the pinned v60 stack.

This module deliberately imports neither Isaac nor Carb so its configuration and
readback checks run on a CPU. Apply after creating the environment: Lab's optional
``RenderCfg.antialiasing_mode`` calls ``set_render_rtx_realtime`` last, which can
silently replace an earlier PathTracing override.

Source contracts: pinned Replicator 1.13.25 ``scripts/settings.py`` and NVIDIA's
https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_pt.html
Readback confirms configured settings, not rendered quality or achieved sample
count. Inspect an actual preview and retain timing/physics evidence separately.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import ceil
from typing import Any, Protocol


class Settings(Protocol):
    def get(self, name: str) -> Any: ...

    def set(self, name: str, value: Any) -> None: ...


@dataclass(frozen=True)
class CaptureQuality:
    """Native presentation resolution and bounded samples at each frozen pose.

    Keep the wrist camera at its existing sensor resolution. Upsizing its RGB
    without changing depth and intrinsics would invalidate vision backprojection.
    Two additional Kit updates cover render-product delivery latency; they are
    not claimed as additional samples once ``totalSpp`` is reached.
    """

    overview_width: int = 1280
    samples_per_tick: int = 32
    total_samples: int = 64
    optix_denoiser: bool = True

    def __post_init__(self):
        if isinstance(self.overview_width, bool) or self.overview_width not in (1280, 1920):
            raise ValueError("overview_width must be 1280 or 1920 native pixels.")
        if type(self.samples_per_tick) is not int or not 1 <= self.samples_per_tick <= 32:
            raise ValueError("samples_per_tick must be an integer in [1, 32].")
        if type(self.total_samples) is not int or not self.samples_per_tick <= self.total_samples <= 512:
            raise ValueError("total_samples must be an integer between samples_per_tick and 512.")
        if self.total_samples % self.samples_per_tick:
            raise ValueError("total_samples must be divisible by samples_per_tick.")
        if type(self.optix_denoiser) is not bool:
            raise ValueError("optix_denoiser must be a bool.")

    @property
    def overview_resolution(self) -> tuple[int, int]:
        return self.overview_width, self.overview_width * 9 // 16

    @property
    def wrist_resolution(self) -> tuple[int, int]:
        return 480, 320

    @property
    def close_resolution(self) -> tuple[int, int]:
        return 640, 480

    @property
    def updates_per_capture(self) -> int:
        return ceil(self.total_samples / self.samples_per_tick) + 2

    @property
    def startup_kit_args(self) -> str:
        """Allow the offline renderer at startup; pass to AppLauncher.kit_args."""
        return "--/persistent/rtx/modes/pt/enabled=true"

    def carb_settings(self) -> dict[str, Any]:
        """Set both per-tick and total samples; the legacy lighting knob is not used."""
        total_pixels = sum(
            width * height for width, height in (self.overview_resolution, self.wrist_resolution, self.close_resolution)
        )
        return {
            "/rtx/rendermode": "PathTracing",
            "/rtx/pathtracing/spp": self.samples_per_tick,
            "/rtx/pathtracing/totalSpp": self.total_samples,
            # Prevent the Lab quality preset's launch cap from trimming samples.
            "/rtx/pathtracing/maxSamplesPerLaunch": total_pixels * self.samples_per_tick,
            "/rtx/pathtracing/maxBounces": 4,
            "/rtx/pathtracing/adaptiveSampling/enabled": False,
            "/rtx/pathtracing/cached/enabled": False,
            "/rtx/pathtracing/optixDenoiser/enabled": self.optix_denoiser,
            "/rtx/pathtracing/optixDenoiser/blendFactor": 0.0,
            "/rtx/pathtracing/optixDenoiser/temporalMode/enabled": False,
            # Color denoising must not smooth simulator depth / annotation AOVs.
            "/rtx/pathtracing/optixDenoiser/AOV": False,
            "/rtx-transient/dldenoiser/enabled": False,
            "/rtx-transient/dlssg/enabled": False,
            "/rtx/post/aa/op": 0,
            "/rtx/post/motionblur/enabled": False,
            "/omni/replicator/captureMotionBlur": False,
            "/rtx/resetPtAccumOnAnimTimeChange": True,
        }


def settings_readback(settings: Settings, requested: Mapping[str, Any]) -> dict[str, Any]:
    """Retain actual Carb values and mismatches without calling them GPU evidence."""
    actual = {key: settings.get(key) for key in requested}
    mismatches = {
        key: {"requested": value, "actual": actual[key]} for key, value in requested.items() if actual[key] != value
    }
    return {
        "requested": dict(requested),
        "actual": actual,
        "settings_match": not mismatches,
        "mismatches": mismatches,
        "evidence_scope": "Carb setting readback only; not measured sample count or visual quality",
    }


def apply_capture_quality(settings: Settings, quality: CaptureQuality) -> dict[str, Any]:
    """Apply AFTER environment initialization, before warmup; fail on lost overrides.

    For each pose, update physics/cameras, call
    ``env.sim.render(skip_app_pumping=True)`` once to flush the scene, then call
    ``simulation_app.update()`` ``quality.updates_per_capture`` times with
    timeline auto-update disabled. Do not advance physics or mutate the stage
    during these accumulation ticks. Record/read back again after warmup and at
    completion so later renderer overrides cannot go unnoticed.
    """
    requested = quality.carb_settings()
    for name, value in requested.items():
        settings.set(name, value)
    report = settings_readback(settings, requested)
    report.update(
        {
            "profile": "offline_pathtracing_optix" if quality.optix_denoiser else "offline_pathtracing_raw",
            "overview_resolution": list(quality.overview_resolution),
            "wrist_resolution": list(quality.wrist_resolution),
            "close_resolution": list(quality.close_resolution),
            "kit_updates_per_capture": quality.updates_per_capture,
            "physics_stepping_during_capture": False,
        }
    )
    if not report["settings_match"]:
        raise RuntimeError(f"RTX capture settings did not persist: {report['mismatches']}")
    return report
