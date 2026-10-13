import sys
from pathlib import Path

# Ensure root workspace is on python path
root = Path(__file__).parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
