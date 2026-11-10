"""Check every relative markdown link in a tree: the target file exists and the #anchor matches a heading
(GitHub slug rules). Also flags mentions of paths that are not in the tree (research/, HANDOFF, PROGRESS).
Usage: python md_links.py ROOT   (exit 1 on any problem)"""
import re,os,sys,glob,unicodedata
root=os.path.abspath(sys.argv[1] if len(sys.argv)>1 else '.')
def slug(h):
    h=re.sub(r'`|\*\*|\*','',h.strip()).lower()
    h=''.join(c for c in h if c.isalnum() or c in ' -_' or unicodedata.category(c).startswith('L'))
    return h.replace(' ','-')
def anchors(path,cache={}):
    if path not in cache:
        seen={};out=set();fence=False
        for l in open(path,encoding='utf-8'):
            if l.lstrip().startswith('```'):fence=not fence;continue
            m=None if fence else re.match(r'#{1,6}\s+(.*)',l)
            if m:
                s=slug(m.group(1));n=seen.get(s,0);out.add(s if n==0 else f'{s}-{n}');seen[s]=n+1
        cache[path]=out
    return cache[path]
bad=[];n=0
for md in sorted(glob.glob(os.path.join(root,'**','*.md'),recursive=True)):
    rel=os.path.relpath(md,root)
    if rel.startswith(('build/','install/','log/')) or '/target/' in rel:continue
    text=open(md,encoding='utf-8').read();body=re.sub(r'```.*?```','',text,flags=re.S)
    for m in re.finditer(r'\]\(([^)\s]+)\)',body):
        t=m.group(1)
        if re.match(r'[a-z]+:',t):continue
        n+=1;path,_,frag=t.partition('#')
        tgt=os.path.normpath(os.path.join(os.path.dirname(md),path)) if path else md
        if not os.path.exists(tgt):bad.append(f'{rel}: missing target {t}');continue
        if frag and tgt.endswith('.md') and frag not in anchors(tgt):bad.append(f'{rel}: missing anchor {t}')
    for m in re.finditer(r'research/|HANDOFF|PROGRESS_RU|kaggle-sim|/workspace/|/home/',text):
        line=text[:m.start()].count('\n')+1;bad.append(f'{rel}:{line}: mentions non-shipped path {m.group(0)!r}')
print('links checked',n,'problems',len(bad));[print(' ',b) for b in bad];sys.exit(1 if bad else 0)
