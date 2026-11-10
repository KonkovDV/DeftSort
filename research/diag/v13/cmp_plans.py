"""Plans before/after (two source trees) on every recorded head frame (Kaggle TRACE npz of v7, v8, v9b, v11b):
segments -> candidates for box/jar/dish -> first feasible pick (plan_pick) + place (plan_place_yaw) per segment,
with the node's own functions. Prints a JSON per tree; compare with --cmp old.json new.json.
Usage: python cmp_plans.py <tree> <out.json>"""
import sys,json,glob,os,numpy as np
if sys.argv[1]=='--cmp':
    a,b=(json.load(open(x)) for x in sys.argv[2:4]);n=d=0;plans=0
    for k in a:
        for ra,rb in zip(a[k],b[k]):
            n+=1;plans+=len(ra)>3
            if ra!=rb:d+=1;print('DIFF',k,ra[:2])
    print(json.dumps(dict(frames=len(a),records=n,plans=plans,different=d)));sys.exit()
root,outp=sys.argv[1],sys.argv[2]
sys.path.insert(0,root+'/src/team048_solution');sys.path.insert(0,root+'/src/team048_solution/test')
import test_node_stub
from team048_solution import arm as ARM, grasp as GR, perception as PE, node as N
K=np.load('/workspace/DeftSort/src/team048_solution/test/fixtures/v7_s102_start_frames.npz')['K'].reshape(3,3)
R='/workspace/DeftSort/research/kaggle-sim'
frames=sorted(glob.glob(f'{R}/out_v7/trace/*/*.npz')+glob.glob(f'{R}/out_v8/trace/*/*.npz')+glob.glob(f'{R}/out_v9b_f525ed9/trace/*/*.npz')+glob.glob(f'{R}/out_v11b_497471e/trace/*/*.npz'))
TR=[(0.348,0.69),(0.357,-0.709)]   # replay scaffolding only (approximate tray centres for the place planner)
A={s:ARM.Arm(s) for s in 'lr'};out={}
for fn in frames:
    z=np.load(fn);key=fn.split('kaggle-sim/')[1]
    try:
        T=ARM.fk_link('camera_head_link',{'head_pitch_joint':float(z['head_pitch'])});dep=z['depth'].astype(float)
        P=PE.deproject(dep,K,T);v=dep>0;reg=PE.table_region(P,v);pl=PE.table_plane(P,reg)
        dets=PE.segment(P,z['rgb'],reg,pl,(np.array([.18,-.62,-1.]),np.array([.62,.62,1.])),min_pixels=40)
    except Exception as e:out[key]=[['frame',type(e).__name__]];continue
    for s in 'lr':A[s].set_scene(pl,TR)
    recs=[]
    for i,s_ in enumerate(dets):
        cl=P[s_.pixels[:,0],s_.pixels[:,1]];others=[P[d.pixels[:,0],d.pixels[:,1]] for d in dets if d is not s_]
        for cat in ('box','jar','dish'):
            try:f,C=GR.candidates(cl,pl,cat,others)
            except Exception as e:recs.append([i,cat,'err',type(e).__name__]);continue
            rec=[i,cat,[[g['kind'],np.round(g['center'],5).tolist(),round(g['score'],6),bool(g['feasible'])] for g in C]]
            for g in C:
                if not g['feasible']:continue
                side='l' if g['center'][1]>=0 else 'r';Aa=A[side];top=-(pl[1]+pl[0][:2]@g['center'])/pl[0][2]
                if g['kind']=='side':ap=np.r_[g['approach'],0.];ca=np.r_[g['close_axis'],0.];pt='grip'
                else:ap=np.array([0,0,-1.]);ca=np.asarray(g['close_axis'],float);ca=np.r_[ca,0.] if ca.size==2 else ca;pt='pinch'
                c3=np.array([g['center'][0],g['center'][1],top+g['z']])
                p=Aa.plan_pick(c3,ca,ap,g['width'],point=pt,obj=None if 'box' not in g else dict(g['box'],z0=top))
                if p is None:continue
                tx,ty=TR[0] if side=='l' else TR[1]
                yp=N.plan_place_yaw(Aa,p['q_lift'],tx,ty,pl,TR,obj_off=N.held_offset(Aa,p['q_grasp'],g,c3))
                rec.append([g['kind'],np.round(p['q_grasp'],6).tolist(),None if yp is None else [round(yp[0],6),np.round(yp[1][0],6).tolist()]]);break
            recs.append(rec)
    out[key]=recs;print(key,len(dets),flush=True)
json.dump(out,open(outp,'w'))
