"""Frontend asset smoke tests."""

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parent.parent
ASSET_DIR = ROOT / "app_v2" / "assets"
JS_ASSETS = sorted(ASSET_DIR.glob("*.js"))


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required for JS syntax checks")
@pytest.mark.parametrize("asset_path", JS_ASSETS, ids=lambda asset_path: asset_path.name)
def test_javascript_assets_parse(asset_path: Path):
    result = subprocess.run(
        ["node", "--check", str(asset_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
