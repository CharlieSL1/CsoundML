import json
import re
from pathlib import Path


def find_matching_snaps(csd_path: Path) -> Path | None:
    snaps_path = csd_path.with_suffix(".snaps")
    return snaps_path if snaps_path.exists() else None


def parse_snaps(snaps_path: Path) -> dict:
    text = snaps_path.read_text(encoding="utf-8", errors="ignore").strip()
    
    # Try JSON first
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {"_list": data}
    except:
        pass
    
    # Try key=value parsing
    kv = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([^:=\s][^:=]*)\s*[:=]\s*(.*)$", line)
        if m:
            key, val = m.group(1).strip(), m.group(2).strip()
            if val.lower() in {"true", "false"}:
                kv[key] = val.lower() == "true"
            else:
                try:
                    kv[key] = float(val) if "." in val else int(val)
                except:
                    kv[key] = val
    
    return kv if kv else {"raw": text}


def process_snaps(dataset_dir: Path, output_dir: Path, overwrite: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for csd_path in Path(dataset_dir).rglob("*.csd"):
        snaps_path = find_matching_snaps(csd_path)
        if not snaps_path:
            continue
        out_json = output_dir / (csd_path.stem + ".json")
        if out_json.exists() and not overwrite:
            print(f"Skip: {out_json.name}")
            continue
        try:
            data = parse_snaps(snaps_path)
            out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Done: {out_json.name}")
        except Exception as e:
            print(f"Failed {snaps_path.name}: {e}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("dataset_dir", nargs="?", default="CsoundDataset", type=Path)
    p.add_argument("--out", default="preprocessed_snaps", type=Path)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    process_snaps(args.dataset_dir, args.out, args.overwrite)


if __name__ == "__main__":
    main()
