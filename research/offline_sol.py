import sys,json,glob,numpy as np,time,types
sys.path.insert(0,'.')
from team048_solution import arm as ARMMOD
from team048_solution.perception import table_plane,segment
from team048_solution.grasp import candidates
src=open('team048_solution/node.py').read()
ns={};exec("import json,os,numpy as np\nfrom scipy.optimize import linear_sum_assignment\nPRI=json.load(open('team048_solution/data/priors.json'))\n",ns)
idsrc=src[src.index('    def identify'):src.index('    # ---------------- task')]
exec('class S:\n'+idsrc,ns);S=ns['S']()
sys.path.insert(0,'/workspace/fixes/perception');from eval_perception import deproject
A={k:ARMMOD.Arm(k) for k in 'lr'};stats=dict(obj=0,ided=0,correct=0,planned=0);t0=time.time();fails=[]
N=int(sys.argv[1]) if len(sys.argv)>1 else 12
for js in sorted(glob.glob('/workspace/kaggle/team048-render-hard/out/set/render/p3_s*.json'))[:N]:
    m=json.load(open(js));z=np.load(js[:-5]+'.npz');P,valid=deproject(z['depth_mm'].astype(float),m['intrinsics'],m['camera'])
    P=P.copy();P[...,2]+=ARMMOD.TABLE_TOP-.75
    plane=table_plane(P,valid);dets=segment(P,z['rgb'],valid,plane,(np.array([.18,-.62,-1.]),np.array([.62,.62,1.])),min_pixels=40)
    names=[o['name'] for o in m['objects']];cat={o['name']:o['category'] for o in m['objects']}
    out=S.identify(dets,P,plane,names);stats['obj']+=len(names)
    for det,name,c,conf,cloud,cost in out:
        if conf<=1 or cost>=2.5:continue
        stats['ided']+=1;stats['correct']+=c==cat[name]
        if '--noplan' in sys.argv:continue
        f,C=candidates(cloud,plane,c,[],W_MAX=.095,robot_xy=(0,0))
        top=-(plane[1]+plane[0][:2]@f['center'])/plane[0][2]
        ok=False
        for g,side in [(g,sd) for g in C for sd in (('l','r') if g['center'][1]>=0 else ('r','l'))]:
            c3=np.array([*g['center'],top+g['z']]);ap=np.r_[g['approach'],0.] if g['kind']=='side' else np.array([0,0,-1.])
            if A[side].plan_pick(c3,np.r_[g['close_axis'],0.],ap,g['width']):stats['planned']+=1;ok=True;break
        if not ok:fails.append((c,name,np.round(f['center'],2).tolist(),round(f['width'],3),len(C)))
print(fails);print(stats,round(time.time()-t0),'s')
