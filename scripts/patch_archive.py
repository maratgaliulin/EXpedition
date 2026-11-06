"""Rewrite a platform archive with a minimal diff: copy every member verbatim, replace only what you name.

Why not re-zip the tree: `build/koz3_submission_v21_gorsnap.zip` (the artefact the platform accepted and scored
0.50190) has 7371 members, of which 838 are directory entries and 1165 are STORED (uncompressed). A fresh
`os.walk` + ZIP_DEFLATED container has 6533 members and no STORED entries. Those differences are invisible to
`unzip` but not necessarily to the platform's validator, and they make every member's metadata differ from the
known-good upload. This script keeps the container byte-for-byte and changes only the files you ask for.

  python scripts/patch_archive.py --base build/koz3_submission_v21_gorsnap.zip --out build/next.zip \
      --replace inference/koz3_infer/experts/sp_or.py=inference/koz3_infer/experts/sp_or.py

  --replace MEMBER=PATHFILE   copy PATHFILE into MEMBER (repeatable)
  --config FILE               merge FILE's top-level keys into inference/models/config.json
  --set KEY=VALUE             set one key of inference/models/config.json (bool/int/float parsed); the key is
                              added at top level if absent
  --unset KEY[,KEY]           delete top-level keys from that config, e.g. --unset diagnostic
  --verify BASE               after writing, compare member-by-member with BASE and print the difference

Never set "diagnostic" here: `main.py` no longer supports it, and a non-scoring build wastes a submission.

Both archive paths are relative to ROOT; a replaced file is read from the workspace (not from a tree copy), so
you never rebuild a 1 GB staging directory.
"""
import argparse
import json
import os
import shutil
import sys
import zipfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CFG = "inference/models/config.json"


def _scalar(s):
    low = s.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--replace", action="append", nargs="*", default=[],
                    help="MEMBER=PATH pairs, e.g. inference/koz3_infer/experts/sp_or.py=inference/.../sp_or.py")
    ap.add_argument("--config")
    ap.add_argument("--set", action="append", nargs="*", default=[])
    ap.add_argument("--unset", action="append", nargs="*", default=[],
                    help="top-level config keys to delete, e.g. --unset diagnostic")
    ap.add_argument("--verify")
    a = ap.parse_args()

    repl = {}
    for group in a.replace:
        for item in group:
            member, _, path = item.partition("=")
            if not path:
                sys.exit(f"--replace needs MEMBER=PATH, got {item!r}")
            repl[member] = os.path.join(ROOT, path)
    sets = {}
    for group in a.set:
        for item in group:
            k, _, v = item.partition("=")
            if not v:
                sys.exit(f"--set needs KEY=VALUE, got {item!r}")
            sets[k] = _scalar(v)
    unsets = [k for group in a.unset for item in group for k in item.split(",")]

    zbase = zipfile.ZipFile(a.base)
    infos = zbase.infolist()
    names = set(zbase.namelist())
    touches_cfg = bool(a.config or sets or unsets)
    for m in list(repl) + ([CFG] if touches_cfg else []):
        if m not in names:
            sys.exit(f"{m} is not a member of {a.base}")

    new_cfg = None
    if touches_cfg:
        cfg = json.loads(zbase.read(CFG))            # keep the shipped key order: do NOT sort_keys
        if a.config:
            cfg.update(json.load(open(os.path.join(ROOT, a.config))))
        cfg.update(sets)
        for k in unsets:
            cfg.pop(k, None)
        new_cfg = json.dumps(cfg, indent=1).encode()

    tmp = a.out + ".part"
    with zipfile.ZipFile(tmp, "w") as zout:
        for i in infos:
            if i.filename.endswith("/"):             # keep directory entries, exactly as shipped
                zout.writestr(i, b"")
                continue
            if i.filename in repl:
                zout.writestr(i, open(repl[i.filename], "rb").read())
            elif i.filename == CFG and new_cfg is not None:
                zout.writestr(i, new_cfg)
            else:
                zout.writestr(i, zbase.read(i))      # verbatim data, original compress_type/metadata
    shutil.move(tmp, a.out)
    print(f"{a.out}: {os.path.getsize(a.out) / 1e9:.3f} GB, {len(infos)} members "
          f"({sum(1 for i in infos if i.filename.endswith('/'))} dirs), changed: "
          f"{sorted(set(repl) | ({CFG} if new_cfg is not None else set()))}")

    if a.verify:
        ref = {i.filename: i.CRC for i in zipfile.ZipFile(a.verify).infolist() if not i.filename.endswith("/")}
        new = {i.filename: i.CRC for i in zipfile.ZipFile(a.out).infolist() if not i.filename.endswith("/")}
        print("missing:", sorted(set(ref) - set(new))[:5], "extra:", sorted(set(new) - set(ref))[:5])
        print("differing:", sorted(k for k in ref if k in new and ref[k] != new[k]))


if __name__ == "__main__":
    main()
