"""Verify the curated source/evidence and optional learned asset ZIP; stdlib only."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify_assets(path):
    meta = json.loads((ROOT / 'metadata/assets.json').read_text(encoding='utf-8'))
    if digest(path.read_bytes()) != meta['archive']['sha256']:
        raise ValueError('Learned asset archive SHA256 mismatch')
    with zipfile.ZipFile(path) as z:
        expected = {row['path'] for row in meta['files']} | {'ASSETS.json'}
        if set(z.namelist()) != expected:
            raise ValueError('Unexpected asset member list')
        for row in meta['files']:
            data = z.read(row['path'])
            if len(data) != row['bytes'] or digest(data) != row['sha256']:
                raise ValueError('Asset mismatch: ' + row['path'])
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets-zip', type=Path)
    args = parser.parse_args()
    checked = 0
    for line in (ROOT / 'metadata/FILES.sha256').read_text(encoding='utf-8').splitlines():
        expected, relative = line.split('  ', 1)
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or digest(path.read_bytes()) != expected:
            raise ValueError('Repository file mismatch: ' + relative)
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=relative)
        checked += 1
    if args.assets_zip:
        verify_assets(args.assets_zip)
    print(f'OK: {checked} repository files; Python syntax valid; assets=' + ('verified' if args.assets_zip else 'not supplied'))


if __name__ == '__main__':
    main()
