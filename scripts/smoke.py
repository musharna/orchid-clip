"""Import/compile smoke test.

There is no unit-test suite here, and app.py sys.exit()s at import when the
generated assets/ payload is absent (it is not committed), so this deliberately
stops short of importing app. What it does cover is enough to catch the thing
dependency bumps actually break: whether the package still imports against the
installed torch / Pillow / open_clip, and whether every module still parses.
"""
import importlib, py_compile, pathlib, sys

# Python puts THIS file's directory (scripts/) on sys.path, not the repo
# root, so orchid_clip is not importable unless we add the root ourselves.
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

mods = ["orchid_clip", "orchid_clip.abstain", "orchid_clip.embedder", "orchid_clip.genus"]
for m in mods:
    importlib.import_module(m)
    print(f"import ok   {m}")

for f in sorted(ROOT.rglob("*.py")):
    if ".venv" in f.parts:
        continue
    py_compile.compile(str(f), doraise=True)
    print(f"compiles    {f}")

import torch, PIL  # noqa: E402  -- the deps a bump would break
print(f"torch       {torch.__version__}")
print(f"pillow      {PIL.__version__}")
print("smoke OK")
