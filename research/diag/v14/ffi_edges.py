"""Rust <-> NumPy FFI seam: every dispatched kernel vs its NumPy reference on edge inputs (NaN, inf, zeros, empty,
non-contiguous views, wrong dtypes). Rust mode required (TEAM048_ACCEL unset). For each case: both return equal
results (NaN-aware, exact), or both raise the same exception type; plus whether the extension was disabled
(silent fallback). Usage: python ffi_edges.py <repo>"""
import sys,os,numpy as np
root=sys.argv[1] if len(sys.argv)>1 else os.path.join(os.path.dirname(__file__),'..','..','..')
sys.path.insert(0,root+'/src/team048_solution');sys.path.insert(0,root+'/src/team048_solution/test')
import test_node_stub
from team048_solution import perception as PE, trays as TR, grasp as GR, _accel
assert _accel.enabled2(),_accel.REASON
def run(f,*a,**k):
    try:return ('ok',f(*a,**k))
    except Exception as e:return ('exc',type(e).__name__)
def eq(a,b):
    if type(a)!=type(b) and not (isinstance(a,(tuple,list)) and isinstance(b,(tuple,list))):
        try:return np.array_equal(np.asarray(a),np.asarray(b),equal_nan=True)
        except Exception:return False
    if isinstance(a,dict):return a.keys()==b.keys() and all(eq(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(eq(x,y) for x,y in zip(a,b))
    if isinstance(a,PE.Detection):return all(eq(getattr(a,f),getattr(b,f)) for f in ('centroid','lower','upper','pixels','features'))
    if a is None or b is None:return a is b
    try:return np.array_equal(np.asarray(a),np.asarray(b),equal_nan=True)
    except Exception:return a==b
def close(a,b):
    try:
        fa=[np.asarray(x,float).ravel() for x in (a if isinstance(a,(tuple,list)) else [a])]
        fb=[np.asarray(x,float).ravel() for x in (b if isinstance(b,(tuple,list)) else [b])]
        return all(x.shape==y.shape and np.allclose(x,y,rtol=0,atol=1e-12,equal_nan=True) for x,y in zip(fa,fb))
    except Exception:return False
rows=[];rng=np.random.default_rng(7)
def case(name,fast,ref,*a,**k):
    was=_accel._disabled;r1=run(fast,*a,**k);dis=_accel._disabled and not was;_accel._disabled=False
    r2=run(ref,*a,**k)
    same=r1[0]==r2[0] and (eq(r1[1],r2[1]) if r1[0]=='ok' else r1[1]==r2[1])
    if not same and r1[0]==r2[0]=='ok' and close(r1[1],r2[1]):same='fp'
    rows.append((name,same,dis,r1[0] if r1[0]=='ok' else 'exc '+r1[1],r2[0] if r2[0]=='ok' else 'exc '+r2[1]))
K=np.array([[343.16,0,320],[0,343.16,240],[0,0,1.]]);T=np.eye(4);T[:3,3]=[.1,0,.5]
d=rng.uniform(300,900,(48,64))
for nm,dd in [('normal',d),('nan',np.where(rng.random(d.shape)<.1,np.nan,d)),('inf',np.where(rng.random(d.shape)<.1,np.inf,d)),
              ('zeros',np.zeros_like(d)),('negative',-d),('uint16',d.astype(np.uint16)),('noncontig',np.asfortranarray(d)[:, ::2]),('empty',np.zeros((0,0)))]:
    case('deproject/'+nm,PE.deproject,PE.deproject_py,dd,K,T)
# plane: tilted-free table at z=-.2 plus clutter
H,W=60,80;P=np.zeros((H,W,3));P[...,0]=np.linspace(.2,.7,W)[None];P[...,1]=np.linspace(-.4,.4,H)[:,None];P[...,2]=-.2+rng.normal(0,.001,(H,W))
P[20:30,30:40,2]+=.05;V=np.ones((H,W),bool)
Pn=P.copy();Pn[rng.random((H,W))<.1]=np.nan
Pi=P.copy();Pi[rng.random((H,W))<.1]=np.inf
for nm,pp,vv in [('normal',P,V),('nan_valid',Pn,V),('inf_valid',Pi,V),('nan_masked',Pn,np.isfinite(Pn).all(-1)),('none_valid',P,np.zeros_like(V)),
                 ('few',P,np.pad(np.ones((5,5),bool),((0,H-5),(0,W-5)))),('noncontig',np.asfortranarray(P),V),('float32',P.astype(np.float32),V)]:
    case('table_plane/'+nm,PE.table_plane,PE.table_plane_py,pp,vv)
rects=[(.4,0.,1.,0.,0.,1.,.05,.05),(.3,.1,.6,.8,-.8,.6,.02,.1)]
R6=[(r[0],r[1],r[2:4],r[4:6],r[6],r[7]) for r in rects]
for nm,pp in [('normal',P),('nan',Pn),('inf',Pi),('noncontig',np.asfortranarray(P))]:
    case('rect_mask/'+nm,PE.rect_mask,lambda P_,R_:PE.rect_mask_py(P_,rects),pp,R6)
def treg_py(P,valid,rects=(),x=(.15,.75),y_abs=.7):
    m=valid&(P[...,0]>x[0])&(P[...,0]<x[1])&(np.abs(P[...,1])<y_abs)
    for cx,cy,hx,hy in rects:m=m&~((np.abs(P[...,0]-cx)<hx)&(np.abs(P[...,1]-cy)<hy))
    return m
for nm,pp in [('normal',P),('nan',Pn),('inf',Pi)]:
    case('table_region/'+nm,PE.table_region,treg_py,pp,V,[(.4,0,.05,.05)])
rgb=rng.integers(0,255,(H,W,3),dtype=np.uint8);plane=(np.array([0,0,1.]),.2);roi=((.15,-.6,-.5),(.75,.6,.5))
for nm,pp,vv in [('normal',P,V),('nan',Pn,V),('inf',Pi,V),('nan_masked',Pn,np.isfinite(Pn).all(-1)),('none_valid',P,np.zeros_like(V)),('noncontig',np.asfortranarray(P),V)]:
    case('segment/'+nm,PE.segment,PE.segment_py,pp,rgb,vv,plane,roi,min_pixels=10)
xy=rng.normal(0,.1,(200,2))*[1,.3]
for nm,a in [('normal',xy),('one',xy[:1]),('two',xy[:2]),('empty',np.zeros((0,2))),('nan',np.vstack([xy,[np.nan,0]])),('noncontig',xy[::2]),('fortran',np.asfortranarray(xy)),('line45',np.c_[xy[:,0],xy[:,0]]),
             ('axis_x',np.c_[np.linspace(-.1,.1,50),np.zeros(50)]),('square',np.array([[0,0],[1,0],[1,1],[0,1.]]))]:
    case('min_area_rect/'+nm,TR.min_area_rect,TR.min_area_rect_py,a)
def fp_py(cloud,plane):
    e=_accel._disabled;_accel._disabled=True
    try:return GR.footprint(cloud,plane)
    finally:_accel._disabled=e
cl=np.c_[rng.normal(.4,.03,(300,2)),rng.uniform(-.19,-.1,300)]
for nm,c in [('normal',cl),('flat_line',np.c_[np.linspace(.3,.5,20),np.zeros(20),np.full(20,-.15)]),('few',cl[:3]),('nan',np.vstack([cl,[np.nan,0,0]]))]:
    case('footprint/'+nm,GR.footprint,fp_py,c,plane)
E=np.zeros((0,0,3));EV=np.zeros((0,0),bool);ER=np.zeros((0,0,3),np.uint8)
case('table_plane/empty',PE.table_plane,PE.table_plane_py,E,EV)
case('rect_mask/empty',PE.rect_mask,lambda P_,R_:PE.rect_mask_py(P_,rects),E,R6)
case('table_region/empty',PE.table_region,treg_py,E,EV,[(.4,0,.05,.05)])
case('segment/empty',PE.segment,PE.segment_py,E,ER,EV,plane,roi)
P1=P[:1];case('segment/one_row',PE.segment,PE.segment_py,P1,rgb[:1],V[:1],plane,roi,min_pixels=1)
def trd(fast):
    e=_accel._disabled;_accel._disabled=not fast
    try:return TR.detect(P,rgb,V,'tray_left','red',plane)
    finally:_accel._disabled=e
case('trays.detect/random',lambda:trd(True),lambda:trd(False))
def trde(fast):
    e=_accel._disabled;_accel._disabled=not fast
    try:return TR.detect(E,ER,EV,'tray_left','red',plane)
    finally:_accel._disabled=e
case('trays.detect/empty',lambda:trde(True),lambda:trde(False))
bad=[r for r in rows if not r[1] or r[2]]
for r in rows:print(('FP  ' if r[1]=='fp' else 'OK  ' if r[1] and not r[2] else 'DIFF')+f' {r[0]:28s} disabled={r[2]!s:5s} fast={r[3]:24s} ref={r[4]}')
print(f'cases {len(rows)}, exact {sum(r[1] is True for r in rows)}, rounding<=1e-12 {sum(r[1]=="fp" for r in rows)}, fallback-triggered {sum(r[2] for r in rows)}')
