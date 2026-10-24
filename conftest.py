"""Pytest root: the repo root on sys.path, so tests import pipeline/common/stages directly.
The old `python -X utf8 test_x.py` runners keep working: every file stays executable as a script."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Smoke scripts stay scripts (scripted models + real data, module-level body): the pytest wrapper below runs
# them as subprocesses; direct collection would execute the whole script at import time.
collect_ignore = [
    'test_pipeline.py',
    'common/test_roles.py',
    'stage5_insight_detection/analytics_insights/test_agent.py',
    'stage5_insight_detection/analytics_insights/test_service.py',
    'stage5_insight_detection/analytics_insights/test_to_insight.py',
]
