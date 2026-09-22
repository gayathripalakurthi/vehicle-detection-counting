from src.utils.config import PROJECT_ROOT, load_config, resolve_path


def test_load_config_has_expected_sections():
    config = load_config()
    for section in ("model", "classes", "tracking", "counting", "io"):
        assert section in config


def test_load_config_model_section_has_required_keys():
    config = load_config()
    for key in ("name", "weights", "device", "img_size", "confidence", "iou"):
        assert key in config["model"]


def test_resolve_path_leaves_absolute_paths_untouched(tmp_path):
    absolute = tmp_path / "some_file.mp4"
    assert resolve_path(absolute) == absolute


def test_resolve_path_resolves_relative_against_project_root():
    resolved = resolve_path("configs/config.yaml")
    assert resolved == PROJECT_ROOT / "configs" / "config.yaml"
    assert resolved.exists()
