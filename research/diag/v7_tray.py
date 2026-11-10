from v7_geom import *
from team048_solution import trays as TR
import glob
def roty(a):ca,sa=np.cos(a),np.sin(a);return np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]])
def correct(P,tb,c,top=-.18):
    n,d=table_plane(P,tb)
    # rotation taking n to z about camera origin c
    v=np.cross(n,[0,0,1.]);s=np.linalg.norm(v);cth=n[2]
    Vx=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]]);R=np.eye(3)+Vx+Vx@Vx*((1-cth)/s**2) if s>1e-9 else np.eye(3)
    Q=(P-c)@R.T+c;n2,d2=table_plane(Q,tb);dz=top-(-d2/n2[2]);Q=Q+[0,0,dz];return Q,R,dz
for f in sorted(glob.glob('/workspace/fixes/kaggle-sim/out_v7/trace/sort3_10*/*closed*.npz'))[:6]:
    z=np.load(f);P,valid=cloud(z);rgb=z['rgb']
    T=ARM.fk_link('camera_head_link',{'head_pitch_joint':float(z['head_pitch'])});c=T[:3,3]
    tb=valid&(P[...,0]>.15)&(P[...,0]<.75)&(np.abs(P[...,1])<.7)
    Q,R,dz=correct(P,tb,c)
    for name,col in (('tray_center','green'),):
        m=TR.chroma_mask(rgb,col)&valid
        for lab,A in (('raw',P),('corr',Q)):
            pts=A[m];h=pts[:,2];rim=pts[h>np.quantile(h,.7)]   # wall tops
            print(os.path.basename(f)[:28],lab,col,'n',m.sum(),'x',rim[:,0].min().round(3),rim[:,0].max().round(3),'y',rim[:,1].min().round(3),rim[:,1].max().round(3),'rim z',np.median(rim[:,2]).round(3),'dz',round(dz,4))
