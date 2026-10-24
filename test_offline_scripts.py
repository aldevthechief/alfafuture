"""One pytest command runs every offline check: the unittest suites are collected directly, the smoke
scripts (scripted models over real data, kept as scripts) run as subprocesses with their output in the failure."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    'test_pipeline.py',
    'common/test_roles.py',
    'stage5_insight_detection/analytics_insights/test_agent.py',
    'stage5_insight_detection/analytics_insights/test_service.py',
    'stage5_insight_detection/analytics_insights/test_to_insight.py',
]


@pytest.mark.parametrize('script', SCRIPTS)
def test_offline_smoke_script(script):
    result = subprocess.run([sys.executable, '-X', 'utf8', str(ROOT / script)],
                            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=1800,
                            cwd=str(ROOT))
    assert result.returncode == 0, f'{script} failed:\n{result.stdout[-1000:]}\n{result.stderr[-1000:]}'
