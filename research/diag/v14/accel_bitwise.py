"""Rust (team048_accel) vs NumPy reference (TEAM048_ACCEL=0) on EVERY recorded head frame (Kaggle TRACE npz v7, v8,
v9b, v11b): byte-exact hashes of every intermediate of the node pipeline and full-precision (repr) values.
Usage: python accel_bitwise.py <tree> <out.json> [i/n shard]   (run twice, with and without TEAM048_ACCEL=0)
       python accel_bitwise.py --cmp a.json b.json   (shards: a*.json merged)"""
import sys,json,glob,hashlib,numpy as np
H=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
if sys.argv[1]=='--cmp':
    def load(x):
        d={}
        for f in sorted(glob.glob(x)):d.update(json.load(open(f)))
        return d
    a,b=(load(x) for x in sys.argv[2:4]);tot=dif=0;diffs={}
    for k in a:
        for f in a[k]:
            tot+=1
            if a[k][f]!=b[k].get(f):dif+=1;diffs.setdefault(f.split(':')[0],[]).append(k.split('/')[-1][:30])
    print(json.dumps(dict(frames=len(a),items=tot,different=dif,by_stage={k:len(v) for k,v in diffs.items()})))
    for k,v in diffs.items():print(k,v[:3])
    sys.exit()
root,outp=sys.argv[1],sys.argv[2]
sys.path.insert(0,root+'/src/team048_solution');sys.path.insert(0,root+'/src/team048_solution/test')
import test_node_stub
from team048_solution import arm as ARM, grasp as GR, perception as PE, node as N, trays as TR, _accel
print('accel',_accel.enabled(),_accel.enabled2(),_accel.REASON,file=sys.stderr)
K=np.load('/workspace/DeftSort/src/team048_solution/test/fixtures/v7_s102_start_frames.npz')['K'].reshape(3,3)
R='/workspace/DeftSort/research/kaggle-sim'
frames=sorted(glob.glob(f'{R}/out_v7/trace/*/*.npz')+glob.glob(f'{R}/out_v8/trace/*/*.npz')+glob.glob(f'{R}/out_v9b_f525ed9/trace/*/*.npz')+glob.glob(f'{R}/out_v11b_497471e/trace/*/*.npz'))
TRS=[(0.348,0.69),(0.357,-0.709)]
if len(sys.argv)>3:i_,n_=map(int,sys.argv[3].split('/'));frames=frames[i_::n_]   # shard i/n (parallel runs)
A={s:ARM.Arm(s) for s in 'lr'};out={}
def r(x):return repr(np.asarray(x,float).tolist())
for fn in frames:
    z=np.load(fn);key=fn.split('kaggle-sim/')[1];o={}
    T=ARM.fk_link('camera_head_link',{'head_pitch_joint':float(z['head_pitch'])});dep=z['depth'].astype(float)
    P=PE.deproject(dep,K,T);v=dep>0;o['deproject']=H(P)
    reg=PE.table_region(P,v);o['table_region']=H(reg)
    try:pl=PE.table_plane(P,reg);o['plane']=r(np.r_[pl[0],pl[1]])
    except Exception as e:o['plane']=type(e).__name__;out[key]=o;continue
    o['near_edge']=repr(PE.near_edge(P,v,pl))
    for name,col in (('tray_left','red'),('tray_right','blue'),('tray_center','green')):
        d=TR.detect(P,z['rgb'],v,name,col,pl);o['tray:'+name]=None if d is None else repr(d)
    rects=[];trays=[]
    for name in ('tray_left','tray_right'):
        d=TR.detect(P,z['rgb'],v,name,{'tray_left':'red','tray_right':'blue'}[name],pl)
        if d is not None and not d['partial']:
            trays.append(d);x,y,ax,ay,a,b,_=ARM.tray_geom(d);vv=np.array([ax,ay]);u=np.array([vv[1],-vv[0]]);rects.append((x,y,u,vv,a/2+.03,b/2+.03))
    m=PE.rect_mask(P,rects) if rects else np.zeros(v.shape,bool);o['rect_mask']=H(m)
    tb=PE.table_region(P,v&~m)
    dets=PE.segment(P,z['rgb'],tb,pl,(np.array([.18,-.62,-1.]),np.array([.62,.62,1.])),min_pixels=40)
    o['segments']=H(np.concatenate([d.pixels.ravel() for d in dets])) if dets else 'none'
    o['seg_feat']=r([np.r_[d.centroid,d.lower,d.upper,d.features] for d in dets])
    for s in 'lr':A[s].set_scene(pl,trays or TRS,None)
    for i,s_ in enumerate(dets):
        cl=P[s_.pixels[:,0],s_.pixels[:,1]];others=[P[d.pixels[:,0],d.pixels[:,1]] for d in dets if d is not s_]
        for cat in ('box','jar','dish'):
            try:f,C=GR.candidates(cl,pl,cat,others,mass=.2)
            except Exception as e:o[f'cand:{i}{cat}']=type(e).__name__;continue
            o[f'cand:{i}{cat}']=repr([(g['kind'],r(g['center']),r(g['close_axis']),repr(g['width']),repr(g['z']),repr(g['score']),repr(g['clearance'])) for g in C])+repr({k:(r(v_) if isinstance(v_,np.ndarray) else repr(v_)) for k,v_ in f.items() if k!='hull'})
            for g in C:
                if not g['feasible']:continue
                side='l' if g['center'][1]>=0 else 'r';Aa=A[side];top=-(pl[1]+pl[0][:2]@g['center'])/pl[0][2]
                if g['kind']=='side':ap=np.r_[g['approach'],0.];ca=np.r_[g['close_axis'],0.];pt='grip'
                else:ap=np.array([0,0,-1.]);ca=np.asarray(g['close_axis'],float);ca=np.r_[ca,0.] if ca.size==2 else ca;pt='pinch'
                c3=np.array([g['center'][0],g['center'][1],top+g['z']])
                p=Aa.plan_pick(c3,ca,ap,g['width'],point=pt,obj=None if 'box' not in g else dict(g['box'],z0=top))
                if p is None:o[f'pick:{i}{cat}']='none';continue
                o[f'pick:{i}{cat}']=r(np.r_[p['q_pre'],p['q_grasp'],p['q_lift']])+repr(p['duration'])
                tr=TRS[0] if side=='l' else TRS[1]
                yp=N.plan_place_yaw(Aa,p['q_lift'],tr[0],tr[1],pl,trays or TRS,obj_off=N.held_offset(Aa,p['q_grasp'],g,c3))
                o[f'place:{i}{cat}']=None if yp is None else repr(yp[0])+r(yp[1][0])
                o[f'path_home:{i}{cat}']=repr(None if Aa.path_home(p['q_lift']) is None else [r(q) for q in Aa.path_home(p['q_lift'])])
                break
    out[key]=o;print(key,len(o),flush=True)
json.dump(out,open(outp,'w'))
