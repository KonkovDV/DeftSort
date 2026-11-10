"""file:line references in docs/*.md and README*.md, each tied to the exact text of the code line it names.
Store: doc_refs.json next to this script, key "doc|file|line" -> stripped text of that line.
  --init ROOT   anchor every reference of the current docs to the code under ROOT (a tree the docs were checked against)
  --fix         move every reference to the line of the current code that carries its anchored text, update the store
  --add         anchor references that have no anchor yet to the current code (review them by hand)
  --check       (default) every reference has an anchor and points at a line with exactly that text
Reference forms: `pkg/node.py:98`, `node.py:98-110`, `T/test_x.py:5,17`, `run.sh:4`, and `:107` continuing the last file
on the same line. pkg/ = src/team048_solution/team048_solution/, T/ = src/team048_solution/test/."""
import re,sys,os,json,glob
HERE=os.path.dirname(os.path.abspath(__file__));R=os.path.normpath(os.path.join(HERE,'..','..','..'))
STORE=os.path.join(HERE,'doc_refs.json')
DIRS=['src/team048_solution/team048_solution','src/team048_solution/test','.','tools','rust/team048_accel/benches']
DOCS=sorted(glob.glob(os.path.join(R,'docs','*.md')))+[os.path.join(R,f) for f in ('README.md','README.en.md')]
TOK=re.compile(r'(?<![\w.])((?:[\w]+/)*[\w]+\.(?:py|sh)):(\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*)|(?<![\w/.\]])`:(\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*)`')
def resolve(name,root):
    base=os.path.basename(name)
    hits=[d for d in DIRS if os.path.exists(os.path.join(root,d,base))]
    if len(hits)!=1:raise SystemExit(f'cannot resolve {name}: {hits}')
    return os.path.normpath(os.path.join(hits[0],base))
def refs(doc,root):
    """[(start, end, file, [numbers as strings]) ...] in document order."""
    out=[]
    for ln in open(doc).read().split('\n'):pass
    text=open(doc).read();last=None;pos=0
    for line in text.split('\n'):
        last=None
        for m in TOK.finditer(line):
            if m.group(1):last=resolve(m.group(1),root);nums=m.group(2);s,e=m.span(2)
            elif last:nums=m.group(3);s,e=m.span(3)
            else:continue
            out.append((pos+s,pos+e,last,nums))
        pos+=len(line)+1
    return text,out
def lines(root,f,cache={}):
    k=(root,f)
    if k not in cache:cache[k]=[l.strip() for l in open(os.path.join(root,f)).read().split('\n')]
    return cache[k]
def nums_of(s):return [int(x) for x in re.split(r'[-,]',s)]
def key(doc,f,n):return f'{os.path.relpath(doc,R)}|{f}|{n}'
def find(root,f,txt,near):
    L=lines(root,f);hits=[i+1 for i,l in enumerate(L) if l==txt]
    if not hits:return None
    return min(hits,key=lambda h:abs(h-near))
def main():
    a=sys.argv[1:];store=json.load(open(STORE)) if os.path.exists(STORE) else {}
    mode=a[0] if a else '--check'
    if mode=='--init':
        root=a[1];store={}
        for d in DOCS:
            _,rs=refs(d,R)
            for s,e,f,nums in rs:
                for n in nums_of(nums):
                    L=lines(root,f)
                    if not 1<=n<=len(L) or not L[n-1]:raise SystemExit(f'{d}: {f}:{n} empty/out of range in {root}')
                    store[key(d,f,n)]=L[n-1]
        json.dump(store,open(STORE,'w'),indent=0,ensure_ascii=False,sort_keys=True);print('anchored',len(store));return
    bad=[];new={};nref=0
    for d in DOCS:
        text,rs=refs(d,R);pieces=[];last=0
        for s,e,f,nums in rs:
            parts=re.split(r'([-,])',nums);outp=[]
            for p in parts:
                if p in '-,':outp.append(p);continue
                n=int(p);nref+=1;k=key(d,f,n)
                if k not in store:
                    if mode=='--add':
                        L=lines(R,f);new[k]=L[n-1] if 1<=n<=len(L) else '';outp.append(p);print('ADDED',k,'->',new[k][:80]);continue
                    bad.append(f'no anchor: {k}');outp.append(p);continue
                txt=store[k];L=lines(R,f)
                if 1<=n<=len(L) and L[n-1]==txt:new[k]=txt;outp.append(p);continue
                h=find(R,f,txt,n)
                if mode=='--fix' and h:new[key(d,f,h)]=txt;outp.append(str(h))
                else:bad.append(f'{k}: line text changed (anchor {txt[:60]!r}, now at {h})');outp.append(p)
            pieces.append(text[last:s]+''.join(outp));last=e
        out=''.join(pieces)+text[last:]
        if mode=='--fix' and out!=text:open(d,'w').write(out);print('fixed',os.path.relpath(d,R))
    if mode in('--fix','--add'):
        if not bad:json.dump(new if mode=='--fix' else {**store,**new},open(STORE,'w'),indent=0,ensure_ascii=False,sort_keys=True)
    print('references',nref,'problems',len(bad));[print(' ',b) for b in bad]
    sys.exit(1 if bad else 0)
main()
