"""Restore the final pingpong source in a NEW official project copy; no network/GPU."""
import argparse
import hashlib
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile

from verify_archive import ROOT, verify_assets

OFFICIAL_SHA = '93fa695d27596f085131756073669a5c255486a935e2c7b94ef7719270126052'


def extract(z, target):
    for info in z.infolist():
        name = PurePosixPath(info.filename)
        path = (target / name).resolve()
        mode = info.external_attr >> 16
        if name.is_absolute() or '..' in name.parts or not path.is_relative_to(target) or stat.S_ISLNK(mode):
            raise ValueError('Unsafe archive member: ' + info.filename)
        if info.is_dir():
            path.mkdir(parents=True, exist_ok=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with z.open(info) as src, path.open('wb') as dst:
            shutil.copyfileobj(src, dst)
        if mode & 0o111:
            path.chmod(path.stat().st_mode | 0o111)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--official-zip', type=Path, required=True)
    parser.add_argument('--assets-zip', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    target = args.output.resolve()
    if target.exists():
        raise ValueError('Output must be a new directory')
    if hashlib.sha256(args.official_zip.read_bytes()).hexdigest() != OFFICIAL_SHA:
        raise ValueError('Official archive SHA256 mismatch')
    verify_assets(args.assets_zip)
    target.mkdir(parents=True)
    with zipfile.ZipFile(args.official_zip) as z:
        extract(z, target)
    inner = target / 'participant/pingpang/participant'
    shutil.copyfile(inner / 'solution.py', inner / 'official_reference.py')
    for source in (ROOT / 'runtime/participant').rglob('*.py'):
        dest = inner / source.relative_to(ROOT / 'runtime/participant')
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    with zipfile.ZipFile(args.assets_zip) as z:
        extract(z, inner)
    print('Prepared final pingpong entry: participant.visual_fast_v002:Solution')
    print('Runtime:', target / 'participant')
    print('Official framework unchanged. No GPU run or training performed.')


if __name__ == '__main__':
    main()
