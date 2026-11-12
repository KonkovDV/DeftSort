#!/usr/bin/env python3
"""Check that the facts stated in the documentation match the repository.

    python3 tools/check_docs.py            # exit code 0 = consistent, 1 = at least one mismatch (each is printed)

Checked:
  * relative links in the Markdown files point at existing files, and `#anchors` at existing headings;
  * `file.py:LINE` references name a file of the package and a line inside it;
  * the per-module line counts in the README tables equal the files' line counts;
  * the number of tests stated in README / CHANGELOG equals what pytest collects;
  * vendor/SHA256SUMS matches the vendored wheel;
  * the crate version is the same in Cargo.toml, pyproject.toml, the wheel name and the READMEs, and the ROS package
    version is the same in package.xml and setup.py;
  * README.md and README.en.md have the same structure (headings, table rows, code blocks, links).

Standard library only (pytest is needed for the test count and is skipped with a note when missing).
"""
import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / 'src' / 'team048_solution' / 'team048_solution'
CRATE = ROOT / 'rust' / 'team048_accel'
DOCS = [ROOT / 'README.md', ROOT / 'README.en.md', ROOT / 'CHANGELOG.md', *sorted((ROOT / 'docs').glob('*.md'))]
READMES = [ROOT / 'README.md', ROOT / 'README.en.md']

errors = []


def fail(where, message):
    errors.append(f'{where}: {message}')


def rel(path):
    return str(Path(path).relative_to(ROOT))


def strip_code(text):
    """Markdown without fenced code blocks (their contents are not links or headings)."""
    return re.sub(r'^```.*?^```', '', text, flags=re.S | re.M)


def anchors(text):
    """GitHub heading anchors of a Markdown text."""
    out = set()
    for title in re.findall(r'^#{1,6}\s+(.+?)\s*$', strip_code(text), flags=re.M):
        slug = re.sub(r'[^\w\- ]', '', title.lower(), flags=re.UNICODE).replace(' ', '-')
        out.add(slug)
    return out


def check_links():
    for doc in DOCS:
        text = doc.read_text(encoding='utf-8')
        for target in re.findall(r'\]\(([^)\s]+)\)', strip_code(text)):
            if re.match(r'[a-z]+://|mailto:', target):
                continue
            path, _, anchor = target.partition('#')
            dest = doc if not path else (doc.parent / path).resolve()
            if not dest.exists():
                fail(rel(doc), f'link to a missing file: {target}')
            elif anchor and dest.suffix == '.md' and anchor not in anchors(dest.read_text(encoding='utf-8')):
                fail(rel(doc), f'link to a missing heading: {target}')
        for src in re.findall(r'<img[^>]+src="([^"]+)"', text):
            if not re.match(r'[a-z]+://', src) and not (doc.parent / src).exists():
                fail(rel(doc), f'image not found: {src}')


def check_line_refs():
    for doc in DOCS:
        for name, line in re.findall(r'`?\b([a-z_]+\.py):(\d+)', doc.read_text(encoding='utf-8')):
            path = PKG / name
            if not path.exists():
                fail(rel(doc), f'{name}:{line} names a file that is not in the package')
            elif int(line) > len(path.read_text(encoding='utf-8').splitlines()):
                fail(rel(doc), f'{name}:{line} is past the end of the file')


def check_module_table():
    for doc in READMES:
        for cell, counts in re.findall(r'^\|\s*((?:`[a-z_]+\.py`(?:,\s*)?)+)\s*\|\s*([\d /]+?)\s*\|', doc.read_text(encoding='utf-8'), flags=re.M):
            names = re.findall(r'`([a-z_]+\.py)`', cell)
            stated = [int(x) for x in re.findall(r'\d+', counts)]
            if len(names) != len(stated):
                fail(rel(doc), f'module table row {cell.strip()}: {len(names)} files but {len(stated)} line counts')
                continue
            for name, n in zip(names, stated):
                actual = len((PKG / name).read_text(encoding='utf-8').splitlines())
                if actual != n:
                    fail(rel(doc), f'module table says {name} has {n} lines, the file has {actual}')


def collected_tests():
    try:
        out = subprocess.run([sys.executable, '-m', 'pytest', '--collect-only', '-q', '-p', 'no:cacheprovider'],
                             cwd=ROOT, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, f'{type(e).__name__}: {e}'
    m = re.search(r'^(\d+) tests? collected', out.stdout, flags=re.M)
    if not m:
        return None, (out.stdout + out.stderr).strip().splitlines()[-1:] or 'no output'
    return int(m.group(1)), None


def check_test_count():
    n, why = collected_tests()
    if n is None:
        print(f'note: test count not checked ({why})')
        return
    patterns = [r'tests-(\d+)-', r'alt="(\d+) tests"', r'(\d+) (?:офлайн-)?тест(?:ов|а)?\b(?! \(включая)',
                r'(\d+) offline tests', r'все (\d+)\b', r'all (\d+) pass', r'\b(\d+) tests on recorded']
    for doc in (*READMES, ROOT / 'CHANGELOG.md'):
        text = doc.read_text(encoding='utf-8')
        for pat in patterns:
            for found in re.findall(pat, text):
                if int(found) != n and int(found) > 100:        # small numbers are other suites (e.g. 18 cargo tests)
                    fail(rel(doc), f'states {found} tests, pytest collects {n}')


def check_wheel_sums():
    sums = ROOT / 'vendor' / 'SHA256SUMS'
    listed = {}
    for line in sums.read_text().splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            listed[name.strip().lstrip('*')] = digest
    wheels = sorted(p.name for p in (ROOT / 'vendor').glob('*.whl'))
    if sorted(listed) != wheels:
        fail('vendor/SHA256SUMS', f'lists {sorted(listed)}, the directory has {wheels}')
    for name, digest in listed.items():
        path = ROOT / 'vendor' / name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            fail('vendor/SHA256SUMS', f'checksum of {name} does not match')


def first(pattern, path):
    m = re.search(pattern, path.read_text(encoding='utf-8'), flags=re.M)
    return m.group(1) if m else None


def check_versions():
    crate = first(r'^version\s*=\s*"([^"]+)"', CRATE / 'Cargo.toml')
    seen = {'rust/team048_accel/pyproject.toml': first(r'^version\s*=\s*"([^"]+)"', CRATE / 'pyproject.toml')}
    for wheel in (ROOT / 'vendor').glob('*.whl'):
        seen[f'vendor/{wheel.name}'] = wheel.name.split('-')[1]
    for doc in READMES:
        seen[rel(doc)] = first(r'team048_accel`? (\d+\.\d+\.\d+)', doc)
    for where, version in seen.items():
        if version != crate:
            fail(where, f'crate version {version}, Cargo.toml says {crate}')
    pkg = ROOT / 'src' / 'team048_solution'
    xml, setup = first(r'<version>([^<]+)</version>', pkg / 'package.xml'), first(r"version='([^']+)'", pkg / 'setup.py')
    if xml != setup:
        fail('src/team048_solution', f'package.xml version {xml}, setup.py version {setup}')


def structure(text):
    body = strip_code(text)
    return {'headings': [len(h) for h in re.findall(r'^(#{1,6})\s', body, flags=re.M)],
            'table rows': len(re.findall(r'^\|', body, flags=re.M)),
            'code blocks': len(re.findall(r'^```', text, flags=re.M)) // 2,
            'links': len(re.findall(r'\]\(([^)\s]+)\)', body)),
            'images': len(re.findall(r'!\[', body))}


def check_readme_parity():
    ru, en = (structure(p.read_text(encoding='utf-8')) for p in READMES)
    for key in ru:
        if ru[key] != en[key]:
            fail('README.md vs README.en.md', f'{key} differ: {ru[key]} vs {en[key]}')


def main():
    for check in (check_links, check_line_refs, check_module_table, check_test_count, check_wheel_sums, check_versions,
                  check_readme_parity):
        check()
    for e in errors:
        print('MISMATCH', e)
    print(f'docs check: {len(errors)} mismatch(es) in {len(DOCS)} documents')
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
