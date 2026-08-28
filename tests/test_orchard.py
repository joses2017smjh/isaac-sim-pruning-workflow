from __future__ import annotations

from isaaclab_pruning.assets import build_v_trellis_layout, load_tree_splits
from isaaclab_pruning.usd.orchard import write_orchard_usda


def test_held_out_trees_match_spur_depth() -> None:
    splits = load_tree_splits()
    assert splits.held_out_envy == ("lpy_envy_00042", "lpy_envy_00065")
    assert splits.camera_rect_depth_m == 0.30
    assert splits.seeds == (0, 1, 2, 3, 4)
    assert "lpy_envy_00000" in splits.train_envy(list(splits.debug_envy) + list(splits.held_out_envy))
    assert splits.is_held_out("lpy_envy_00042")


def test_v_trellis_places_three_trees_posts_and_wires(tmp_path) -> None:
    layout = build_v_trellis_layout()
    assert len(layout.tree_translations) == 3
    assert layout.tree_translations[1] == (0.0, 1.0, 0.0)
    assert layout.tree_tilt_x_deg == -17.143
    assert len(layout.posts) == 3
    assert len(layout.wires) == 4
    assert layout.baseline_bark == "bark_brown_02"

    path = write_orchard_usda(tmp_path / "orchard.usda", layout)
    text = path.read_text(encoding="utf-8")
    assert 'upAxis = "Z"' in text
    assert "post_0" in text
    assert "wire_0" in text
    assert "bark_brown_02" in text
