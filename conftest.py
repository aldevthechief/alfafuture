"""Pytest root: the repo root on sys.path; the smoke scripts under tests/ stay out of direct collection
(they run as subprocesses via tests/integration/test_offline_scripts.py)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

collect_ignore_glob = ['tests/integration/test_pipeline.py', 'tests/unit/common/test_roles.py',
                       'tests/unit/stage5_insight_detection/test_agent.py',
                       'tests/unit/stage5_insight_detection/test_service.py',
                       'tests/unit/stage5_insight_detection/test_to_insight.py']
