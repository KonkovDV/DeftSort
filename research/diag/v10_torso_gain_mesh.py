"""Arm-to-torso MESH distance (UBTECH visual meshes) along home->pre->grasp->lift and lift->carry->place->up->home of
the plans GAINED by shrinking the torso spheres by SH (study_v10_torso) vs the current model; control = sample of
current plans. Usage: python v10_torso_gain_mesh.py 005"""
import sys,os,json,numpy as np
tag=sys.argv[1];SH=int(tag)/1000
os.environ['STUDY_CFG']=open(f'/tmp/cfg_v10_torso{tag}.json').read();os.environ['V10_CLEAR']='1';os.environ['V10_TORSO_SHRINK']=str(SH)
sys.path.insert(0,'/workspace/fixes/kaggle-tpu/study')
import study_v10_torso as VT, study_v10 as V, study_core as S
S.winit('/workspace/solution/nssr-sol-team-048/src/team048_solution')
ARM=S.W['ARM'];NODE=S.W['NODE']
sys.path.insert(0,'/workspace/solution/nssr-sol-team-048/tools')
from fit_spheres import read_stl,rpy_matrix
from scipy.spatial import cKDTree
import xml.etree.ElementTree as ET
U='/workspace/competitors/UBTECH-Robot__tiangong2pro_tf/src/tiangong2pro_urdf';R=ET.parse(U+'/urdf/tiangong2.0_pro_with_hands.urdf').getroot()
def mesh(link):
    l=R.find(f"link[@name='{link}']");c=l.find('collision') or l.find('visual');m=c.find('geometry/mesh');o=c.find('origin')
    Vv=np.unique(np.round(read_stl(os.path.join(U,'meshes',os.path.basename(m.get('filename')))),4),axis=0)
    if len(Vv)>3000:Vv=Vv[np.random.default_rng(0).choice(len(Vv),3000,replace=False)]
    xyz=np.array([float(x) for x in (o.get('xyz') if o is not None else '0 0 0').split()]);rpy=[float(x) for x in (o.get('rpy') if o is not None else '0 0 0').split()]
    return Vv@rpy_matrix(*rpy).T+xyz
TORSO=cKDTree(mesh('waist_yaw_link'))
LINKS={s:[ARM.J[f'{n}_{s}_joint']['child'] for n in ARM.ARM[2:]]+['L_base_link' if s=='l' else 'R_base_link'] for s in 'lr'}
MM={l:mesh(l) for s in 'lr' for l in LINKS[s]}
def dmin(side,q):
    qd={f'{n}_{side}_joint':float(v) for n,v in zip(ARM.ARM,q)};best=9
    for l in LINKS[side]:
        T=ARM.fk_link(l,qd);best=min(best,float(TORSO.query(MM[l]@T[:3,:3].T+T[:3,3])[0].min()))
    return best
C={(r['layout'],r['name']):r for r in map(json.loads,open('/workspace/fixes/kaggle-tpu/study/out/v10-r3s3-clear/results.jsonl'))}
Rr={(r['layout'],r['name']):r for r in map(json.loads,open(f'/workspace/fixes/kaggle-tpu/study/out/v10-r3s3-torso{tag}/results.jsonl'))}
gain={k for k in Rr if Rr[k]['result']=='ok' and C[k]['result']!='ok'}
if os.environ.get('ALL')=='1':gain={k for k in Rr if Rr[k]['result']=='ok'}
L0,L1=int(os.environ.get('L0',Cf_l0:=0)),int(os.environ.get('L1',100))
cap={};op=ARM.Arm.plan_pick;oy=NODE.plan_place_yaw
def pp(self,*a,**k):
    p=op(self,*a,**k)
    if p is not None:cap['p']=(self.side,p)
    return p
def py(A,q_from,*a,**k):
    r=oy(A,q_from,*a,**k)
    if r is not None:cap['y']=r
    return r
ARM.Arm.plan_pick=pp;NODE.plan_place_yaw=py
Cf=S.CFG;trays=S.TASK_TRAYS[Cf['task']];cats=Cf['cats'];cat2tray={c:trays[i] for i,c in enumerate(cats)}
names=[n for c in cats for n in Cf['pool'][c]];dims={n:Cf['meta'][n][1] for n in names};out=[]
for li in range(max(Cf['l0'],L0),min(Cf['l1'],L1)):
    seed=Cf.get('seed_base',70000)+li;lay=S.sample_layout(names,dims,seed,trays)
    objs={n:dict(cat=Cf['meta'][n][0],dims=dims[n],pose=lay[n]) for n in names if lay[n]!='park'}
    for n in objs:
        if (li,n) not in gain:continue
        job=dict(layout=li,seed=seed*31+S.zlib.crc32(n.encode())%1000,name=n,objs=objs,trays=trays,tray=cat2tray[objs[n]['cat']])
        cap.clear();r=S.eval_obj(job)
        if r['result']!='ok' or 'p' not in cap or 'y' not in cap:continue
        side,p=cap['p'];A=S.W['arms'][side];yaw,(qpl,qup,qca)=cap['y']
        path=[A.home,p['q_pre'],p['q_grasp'],p['q_lift']]+[q for q in (qca,qpl,qup) if q is not None]+[A.home]
        d=[min(dmin(side,a+(b-a)*t) for t in np.linspace(0,1,9)) for a,b in zip(path[:-1],path[1:])]
        out.append((li,n[:18],side,round(min(d),4),round(min(d[:3]),4)))
d=np.array([x[3] for x in out])
print('shrink',SH,'gained plans',len(out),'min arm-torso mesh distance over pick+place paths: <5mm',int((d<.005).sum()),'<10mm',int((d<.01).sum()),'<15mm',int((d<.015).sum()),'min %.4f median %.4f'%(d.min(),np.median(d)) if len(d) else '')
print(sorted(out,key=lambda x:x[3]))
if os.environ.get('OUT'):json.dump(out,open(os.environ['OUT'],'w'))
