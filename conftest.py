import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Isolate config/profiles from the real ~/.config during tests.
os.environ.setdefault(
    "XDG_CONFIG_HOME", tempfile.mkdtemp(prefix="llama-launcher-test-")
)
