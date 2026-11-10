"""timeit per function: reference (_py) vs vectorised NumPy fallback vs Rust, real arm model + v7 frames."""
import sys,timeit,json,numpy as np,cycle
from team048_solution import arm as ARM, perception as PE, node as NODE, trays as TR
A=cycle.arms()['l'];ch=A.chain;col=A.col;rng=np.random.default_rng(0)
Q=[np.clip(rng.uniform(ch.lower,ch.upper)*.5,ch.lower,ch.upper) for _ in range(200)]
T=[ch.fk(q) for q in Q]
def per(f,n):return min(timeit.repeat(f,number=n,repeat=3))/n
rows=[]
def row(name,ref,new,vec=None,n=200):
    r=per(ref,n);x=per(new,n);v=per(vec,n) if vec else None
    rows.append((name,r,v,x));print(f'{name:34s} ref {r*1e6:12.1f} us'+(f'  numpy-vec {v*1e6:10.1f} us ({r/v:5.1f}x)' if v else '')+f'  rust {x*1e6:10.1f} us  ({r/x:7.1f}x)',flush=True)
it=iter(range(10**9));g=lambda L:L[next(it)%len(L)]
row('Chain.fk',lambda:ch.fk_py(g(Q)),lambda:ch.fk(g(Q)),n=2000)
row('Chain.frames',lambda:ch.frames_py(g(Q)),lambda:ch.frames(g(Q)),n=2000)
row('Chain.jacobian',lambda:ch.jacobian_py(g(Q)),lambda:ch.jacobian(g(Q)),n=2000)
def _try(f,t):
    try:return f(t,A.home,max_iter=150,q_pref=A.home,w_null=.1)
    except ValueError:return None
row('Chain.ik (null space, home seed)',lambda:(lambda t:_try(ch.ik_py,t))(g(T)),lambda:(lambda t:_try(ch.ik,t))(g(T)),n=40)
row('CollisionModel.valid',lambda:col.valid_py(g(Q)),lambda:col.valid(g(Q)),lambda:col.valid_vec(g(Q)),n=300)
P2=[(Q[i],Q[i+1]) for i in range(0,40,2)]
def cert(f):
    a,b=g(P2);return f(a,b)
row('CollisionModel.certify (sweep)',lambda:cert(col.certify_py),lambda:cert(col.certify),lambda:cert(lambda a,b:all(col.valid_vec(a+(b-a)*(i/s),col.levers@abs(b-a)/s) for s in [max(1,int(np.ceil(np.max(abs(b-a))/.003)))] for i in range(s+1))),n=3)
z=np.load(cycle.FRAMES[0]);P,valid=cycle.frame_cloud(z);tb=valid&(P[...,0]>.15)&(P[...,0]<.75)&(np.abs(P[...,1])<.7)
import team048_solution._accel as AC
def tp_vec():
    s=AC._disabled;AC._disabled=True
    try:return PE.table_plane(P,tb)
    finally:AC._disabled=s
row('perception.table_plane',lambda:PE.table_plane_py(P,tb),lambda:PE.table_plane(P,tb),tp_vec,n=5)
pl=PE.table_plane(P,tb);roi=(np.array([.18,-.62,-1.]),np.array([.62,.62,1.]))
row('perception.segment',lambda:PE.segment_py(P,z['rgb'],tb,pl,roi,min_pixels=40),lambda:PE.segment(P,z['rgb'],tb,pl,roi,min_pixels=40),n=5)
# plan_pick on a typical side grasp
c3=np.array([.40,.15,cycle.zt(.4,.15)+.03])
json.dump(rows,open('/tmp/bench_py.json','w'))
