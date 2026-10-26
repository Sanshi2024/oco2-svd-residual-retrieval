"""Read-only integrity and syntax checks; never imports training scripts."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "SHA256.json").read_text())
for record in manifest["files"]:
    path = root / record["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise SystemExit("Checksum mismatch: " + record["path"])
scripts = list(root.glob("*.py"))
for path in scripts:
    ast.parse(path.read_text(), filename=path.name)
features = json.loads((root / "feature_names.json").read_text())
assert len(features) == len(set(features)) == 99
assert features[-1] == "xco2_apriori"
protocol = json.loads((root / "protocol.json").read_text())
assert protocol["pc_per_band"] == [30, 30, 30]
assert protocol["new_training_performed"] is False
print(json.dumps({"files_verified": len(manifest["files"]),
                  "python_files_parsed": len(scripts), "features": len(features),
                  "training_performed": False}))
