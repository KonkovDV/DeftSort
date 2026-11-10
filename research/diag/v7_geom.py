import sys,glob,os,json,numpy as np
sys.path.insert(0,'/workspace/solution/nssr-sol-team-048/src/team048_solution')
from team048_solution import arm as ARM
from team048_solution.perception import table_plane
K=dict(fx=343.16,cx=320.,cy=180.)
def cloud(z):
    d=z['depth'].astype(float)[...,0] if z['depth'].ndim==3 else z['depth'].astype(float)
    T=ARM.fk_link('camera_head_link',{'head_pitch_joint':float(z['head_pitch'])})
    v,u=np.mgrid[0:d.shape[0],0:d.shape[1]];Z=d/1000.;X=(u-K['cx'])*Z/K['fx'];Y=(v-K['cy'])*Z/K['fx']
    P=(np.stack([X,Y,Z],-1).reshape(-1,3)@T[:3,:3].T+T[:3,3]).reshape(d.shape+(3,));return P,d>0
if __name__=='__main__':
  for f in sorted(glob.glob('/workspace/fixes/kaggle-sim/out_v7/trace/*/*.npz'))[:40]:
    z=np.load(f);P,valid=cloud(z)
    tb=valid&(P[...,0]>.15)&(P[...,0]<.75)&(np.abs(P[...,1])<.7)
    n,d=table_plane(P,tb)
    # table height at x=.4,y=0
    zt=-(d+n[0]*.4)/n[2]
    print(os.path.basename(f)[:40],'pitch',round(float(z['head_pitch']),3),'n',n.round(3),'table z@(.4,0)',round(zt,4),'tilt deg',round(np.degrees(np.arccos(abs(n[2]))),2))
