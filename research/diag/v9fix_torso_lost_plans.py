"""Plans the torso envelope removes: re-plan (base = master d96da87 code) the objects that are 'ok' on base but not on
v9fix in the 100-layout pool-3 sort3 reach study, and measure the arm-to-torso MESH distance along
home -> pre -> grasp -> lift (joint-linear, as commanded). Control: objects ok on both (v9fix plans)."""
import sys,os,json,collections,numpy as np
which=sys.argv[1]   # base|fix
SRC={'base':'/tmp/wt_m/src/team048_solution','fix':'/workspace/solution/nssr-sol-team-048/src/team048_solution'}[which]
os.environ['STUDY_CFG']=open('/tmp/cfg_v9_%s.json'%('base' if which=='base' else 'fix')).read()
sys.path.insert(0,'/workspace/fixes/kaggle-tpu/study');import study_core as S
S.winit(SRC)
from scipy.spatial import cKDTree
ARM=S.W['ARM'];sys.path.insert(0,'/workspace/solution/nssr-sol-team-048/tools')
from fit_spheres import read_stl,rpy_matrix
import xml.etree.ElementTree as ET
U='/workspace/competitors/UBTECH-Robot__tiangong2pro_tf/src/tiangong2pro_urdf';R=ET.parse(U+'/urdf/tiangong2.0_pro_with_hands.urdf').getroot()
def mesh(link):
    l=R.find(f"link[@name='{link}']");c=l.find('collision') or l.find('visual');m=c.find('geometry/mesh');o=c.find('origin')
    V=np.unique(np.round(read_stl(os.path.join(U,'meshes',os.path.basename(m.get('filename')))),4),axis=0)
    if len(V)>3000:V=V[np.random.default_rng(0).choice(len(V),3000,replace=False)]
    xyz=np.array([float(x) for x in (o.get('xyz') if o is not None else '0 0 0').split()]);rpy=[float(x) for x in (o.get('rpy') if o is not None else '0 0 0').split()]
    return V@rpy_matrix(*rpy).T+xyz
TORSO=cKDTree(mesh('waist_yaw_link'))
LINKS={s:[ARM.J[f'{n}_{s}_joint']['child'] for n in ARM.ARM[2:]]+['L_base_link' if s=='l' else 'R_base_link'] for s in 'lr'}
MM={l:mesh(l) for s in 'lr' for l in LINKS[s]}
def dmin(side,q):
    qd={f'{n}_{side}_joint':float(v) for n,v in zip(ARM.ARM,q)};best=9
    for l in LINKS[side]:
        T=ARM.fk_link(l,qd);best=min(best,float(TORSO.query(MM[l]@T[:3,:3].T+T[:3,3])[0].min()))
    return best
base={(r['layout'],r['name']):r for r in map(json.loads,open('/workspace/fixes/kaggle-tpu/study/out/v9fix-r3s3-base/results.jsonl'))}
fix={(r['layout'],r['name']):r for r in map(json.loads,open('/workspace/fixes/kaggle-tpu/study/out/v9fix-r3s3-fix/results.jsonl'))}
lost=[k for k in base if base[k]['result']=='ok' and fix[k]['result']!='ok']
both=[k for k in base if base[k]['result']=='ok' and fix[k]['result']=='ok']
want=set(lost) if which=='base' else set(both[::8])
cap={}
orig=ARM.Arm.plan_pick
def pp(self,*a,**k):
    p=orig(self,*a,**k)
    if p is not None:cap['p']=(self.side,p)
    return p
ARM.Arm.plan_pick=pp
# rebuild the study's jobs (same generator as study_core.run)
C=S.CFG;trays=S.TASK_TRAYS[C['task']];cats=C['cats'];cat2tray={c:trays[i] for i,c in enumerate(cats)}
names=[n for c in cats for n in C['pool'][c]];dims={n:C['meta'][n][1] for n in names};out=[]
for li in range(C['l0'],C['l1']):
    seed=C.get('seed_base',70000)+li;lay=S.sample_layout(names,dims,seed,trays)
    objs={n:dict(cat=C['meta'][n][0],dims=dims[n],pose=lay[n]) for n in names if lay[n]!='park'}
    for n in objs:
        if (li,n) not in want:continue
        job=dict(layout=li,seed=seed*31+S.zlib.crc32(n.encode())%1000,name=n,objs=objs,trays=trays,tray=cat2tray[objs[n]['cat']])
        cap.clear();r=S.eval_obj(job)
        if r['result']!='ok' or 'p' not in cap:continue
        side,p=cap['p'];A=S.W['arms'][side];path=[A.home,p['q_pre'],p['q_grasp'],p['q_lift']]
        d=min(dmin(side,a+(b-a)*t) for a,b in zip(path[:-1],path[1:]) for t in np.linspace(0,1,9))
        out.append((li,n[:20],r['cat'],side,round(d,4)))
d=np.array([x[-1] for x in out])
print(which,'plans',len(out),'min torso mesh distance: <0 mm n/a; <5 mm',int((d<.005).sum()),'<10 mm',int((d<.010).sum()),'<15 mm',int((d<.015).sum()),'median %.3f'%np.median(d))
print(sorted(out,key=lambda x:x[-1])[:12])
