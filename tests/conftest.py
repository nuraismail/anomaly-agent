import os
import tempfile
from pathlib import Path


mpl_config_dir = Path(tempfile.gettempdir()) / "anomaly-agent-test-mpl"
mpl_config_dir.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
