"""Assemble the files for the Hub repo into a staging directory: the card as README.md, every build variant, and the
sources needed to rebuild. Usage: python stage_upload.py OUT_DIR   (then upload OUT_DIR to main and branch v<version>)"""
import os, shutil, sys, tomllib

ROOT = os.path.dirname(os.path.abspath(__file__))
out = sys.argv[1]
cfg = tomllib.load(open(os.path.join(ROOT, "build.toml"), "rb"))
shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
card = open(os.path.join(ROOT, "CARD.md")).read()
card = card.replace("{{ repo_id }}", cfg["general"]["hub"]["repo-id"]).replace("{{ version }}", str(cfg["general"]["version"]))
assert "{{" not in card, "unfilled placeholder in CARD.md"
open(os.path.join(out, "README.md"), "w").write(card)
for f in ["build.toml", "flake.nix", "flake.lock", "LICENSE", "build_variant.py", "pytest.ini"]:
    shutil.copy(os.path.join(ROOT, f), out)
ignore = shutil.ignore_patterns("__pycache__", ".obj", "*.o", ".pytest_cache")
for d in ["trellis2_rocm", "torch-ext", "tests", "build"]:
    shutil.copytree(os.path.join(ROOT, d), os.path.join(out, d), ignore=ignore)
variants = sorted(os.listdir(os.path.join(out, "build")))
print(f"{out}: {len(variants)} variants\n  " + "\n  ".join(variants))
