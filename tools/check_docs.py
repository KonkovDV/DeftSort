#!/usr/bin/env python3
"""Check that the facts stated in the documentation match the repository.

    python3 tools/check_docs.py            # exit code 0 = consistent, 1 = at least one mismatch (each is printed)

Checked:
  * relative links in the Markdown files point at existing files, and `#anchors` at existing headings;
  * `file.py:LINE` references (and a `:LINE` that continues the file named earlier on the same line) name a file of
    the package and a non-blank line inside it; a line that is `def name` must name that function, and a citation
    inside a function must not name only other functions of that file;
  * `research/**/*.md` is an archive of past runs and is not checked;
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
DOCS = [ROOT / 'README.md', ROOT / 'README.en.md', ROOT / 'CHANGELOG.md', ROOT / 'HANDOFF.md',
        *sorted((ROOT / 'docs').glob('*.md'))]
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


def function_spans(text):
    """Half-open line ranges [start, end) of every function. A nested function is the later start."""
    lines = text.splitlines()
    defs = []
    for i, line in enumerate(lines, 1):
        m = re.match(r'^(\s*)(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(', line)
        if m:
            defs.append((len(m.group(1)), m.group(2), i))
    spans = []
    for k, (ind, name, start) in enumerate(defs):
        end = len(lines) + 1
        for ind2, _, start2 in defs[k + 1:]:
            if ind2 <= ind:
                end = start2
                break
        spans.append((name, start, end))
    return spans


def owner_at(spans, line):
    hit = [span for span in spans if span[1] <= line < span[2]]
    return max(hit, key=lambda span: span[1])[0] if hit else None


def line_citations(text):
    """(markdown line, markdown text, filename, source line). A `:N` continues the file named earlier on that line."""
    ref = re.compile(r'\b([a-z_]+\.py):(\d+)(?:-(\d+))?|(?<![\w.])\:(\d+)\b')
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        current = None
        for m in ref.finditer(line):
            if m.group(1):
                current = m.group(1)
                out.append((i, line, current, int(m.group(2))))
                if m.group(3):
                    out.append((i, line, current, int(m.group(3))))
            else:
                out.append((i, line, current, int(m.group(4))))
    return out


def mentioned_functions(md_line, names):
    """Function names of this file that the sentence puts in their own backticks."""
    found = set()
    for tick in re.findall(r'`([^`]+)`', md_line):
        if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', tick) and tick in names:
            found.add(tick)
    return found


def check_line_refs():
    loaded = {}
    for doc in DOCS:
        for md_i, md_line, name, n in line_citations(doc.read_text(encoding='utf-8')):
            where = f'{rel(doc)}:{md_i}'
            if name is None:
                fail(where, f':{n} does not follow a file.py reference on the same line')
                continue
            if name not in loaded:
                path = PKG / name
                if path.exists():
                    text = path.read_text(encoding='utf-8')
                    loaded[name] = (text.splitlines(), function_spans(text))
                else:
                    loaded[name] = None
            src = loaded[name]
            if src is None:
                fail(where, f'{name}:{n} names a file that is not in the package')
                continue
            lines, spans = src
            if n < 1 or n > len(lines):
                fail(where, f'{name}:{n} is past the end of the file ({len(lines)} lines)')
                continue
            code = lines[n - 1].strip()
            if not code:
                fail(where, f'{name}:{n} is a blank line')
                continue
            owner = owner_at(spans, n)
            mentioned = mentioned_functions(md_line, {span[0] for span in spans})
            named = bool(owner and re.search(r'\b' + re.escape(owner) + r'\b', md_line))
            if owner and re.match(r'(?:async\s+)?def\s+' + re.escape(owner) + r'\s*\(', code) and not named:
                fail(where, f'{name}:{n} is def {owner}, and the sentence does not name it')
            if owner and mentioned and owner not in mentioned and not named:
                fail(where, f'{name}:{n} is inside {owner}; the sentence names {", ".join(sorted(mentioned))}')


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
