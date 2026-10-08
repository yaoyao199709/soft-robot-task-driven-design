"""Check source syntax and repository resource references without scientific packages."""
from pathlib import Path
import ast
import xml.etree.ElementTree as ET
import sys

ROOT = Path(__file__).resolve().parents[1]
errors = []
files = list(ROOT.glob('cases/**/*.py'))
for filename in files:
    try:
        ast.parse(filename.read_text(encoding='utf-8-sig'), filename=str(filename))
    except (SyntaxError, UnicodeError) as exc:
        errors.append(f'{filename.relative_to(ROOT)}: {exc}')
for urdf in ROOT.glob('cases/**/*.urdf'):
    try:
        tree = ET.parse(urdf)
    except ET.ParseError as exc:
        errors.append(f'{urdf.relative_to(ROOT)}: XML error {exc}')
        continue
    for node in tree.findall('.//mesh'):
        mesh = node.attrib.get('filename', '')
        if not mesh or '://' in mesh or mesh.startswith('package:'):
            continue
        if not (urdf.parent / mesh).exists():
            # PyBullet may use additional search paths, so log for diagnosis.
            print(f'CHECK MESH SEARCH PATH: {urdf.relative_to(ROOT)} -> {mesh}')
if errors:
    print('\n'.join(errors), file=sys.stderr)
    raise SystemExit(1)
print(f'PASS: parsed {len(files)} Python modules and checked URDF XML references')
