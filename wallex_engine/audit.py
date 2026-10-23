from dataclasses import asdict, is_dataclass
from pathlib import Path
import hashlib
import json
import os
import platform


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, default=lambda x: asdict(x) if is_dataclass(x) else str(x))


class Audit:
    """Append-only hash chain. Tamper evidence, NOT an authenticated exchange ledger."""
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.seq, self.previous = self.verify(self.path) if self.path.exists() else (0, "0"*64)

    @staticmethod
    def verify(path):
        seq, previous = 0, "0"*64
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                digest = obj.pop("hash")
                if obj["previous"] != previous or obj["seq"] != seq + 1:
                    raise ValueError("broken audit sequence")
                if hashlib.sha256(canonical(obj).encode()).hexdigest() != digest:
                    raise ValueError("audit hash mismatch")
                seq, previous = obj["seq"], digest
        return seq, previous

    def write(self, kind, ts, **payload):
        obj = {"seq": self.seq+1, "previous": self.previous, "kind": kind, "ts": ts,
               "payload": payload}
        digest = hashlib.sha256(canonical(obj).encode()).hexdigest()
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(canonical({**obj, "hash": digest}) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.seq, self.previous = self.seq+1, digest


def source_hash():
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def manifest(config, data_hash, mode):
    return {"source_sha256": source_hash(), "config": asdict(config), "data_sha256": data_hash,
            "mode": mode, "python": platform.python_version(), "engine_version": "0.1.0"}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(canonical(value))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
