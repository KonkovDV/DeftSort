"""For items that differ between two accel_bitwise.py outputs: is the difference only floating-point rounding?
Each value is split into its numeric tokens and the remaining text skeleton (structure, kinds, None/True/False,
strings). The skeleton and every integer token must match exactly, floats within `tol` (absolute). Hash-only stages
(deproject, masks, segment pixels) have no float view and are listed separately.
Usage: python accel_tol.py a.json 'b*.json' [tol]"""
import sys,json,glob,re,collections
NUM=re.compile(r'-?(?:\d+\.\d*(?:e[-+]?\d+)?|\d+e[-+]?\d+|\.\d+(?:e[-+]?\d+)?|\d+)')
NPW=re.compile(r'np\.(?:float64|int64|bool_)\(([^()]*)\)')
def load(x):
    d={}
    for f in sorted(glob.glob(x)):d.update(json.load(open(f)))
    return d
def split(v):
    v=NPW.sub(r'\1',str(v));return NUM.sub('#',v),NUM.findall(v)
HASH=('deproject','table_region','rect_mask','segments')
a,b=load(sys.argv[1]),load(sys.argv[2]);TOL=float(sys.argv[3]) if len(sys.argv)>3 else 1e-9
mx=collections.defaultdict(float);disc=[];hashd=collections.Counter();n=0;items=0
for k in a:
    for f in a[k]:
        items+=1
        if a[k][f]==b[k].get(f):continue
        n+=1;st=f.split(':')[0]
        if st in HASH:hashd[st]+=1;continue
        (sa,na),(sb,nb)=split(a[k][f]),split(b[k].get(f))
        bad=sa!=sb or len(na)!=len(nb)
        if not bad:
            for x,y in zip(na,nb):
                if ('.' in x or 'e' in x) and ('.' in y or 'e' in y):
                    d=abs(float(x)-float(y));mx[st]=max(mx[st],d);bad|=d>TOL
                else:bad|=x!=y
        if bad:disc.append((k.split('/')[-1][:34],f))
print(json.dumps(dict(items=items,differing=n,hash_only=dict(hashd),max_float_diff={k:float('%.3g'%v) for k,v in mx.items()},
                      discrete_or_above_tol=len(disc))))
for x in disc[:20]:print('DISCRETE',x)
if disc and '-v' in sys.argv:
    k0=[k for k in a if k.split('/')[-1].startswith(disc[0][0])][0];print(a[k0][disc[0][1]][:800]);print(b[k0][disc[0][1]][:800])
