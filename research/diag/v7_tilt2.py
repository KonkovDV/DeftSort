from v7_geom import *
f=sorted(glob.glob('/workspace/fixes/kaggle-sim/out_v7/trace/sort3_102/*.npz'))[0];z=np.load(f);d=z['depth'].astype(float);d=d[...,0] if d.ndim==3 else d
hp=float(z['head_pitch'])
v,u=np.mgrid[0:360,0:640];Z=d/1000.;Xc=np.stack([(u-320)*Z/343.16,(v-180)*Z/343.16,Z],-1).reshape(-1,3);valid=(d>0)
for dp in (0,.03,.05,.06,.07,.08,.10):
    T=ARM.fk_link('camera_head_link',{'head_pitch_joint':hp+dp})
    P=(Xc@T[:3,:3].T+T[:3,3]).reshape(360,640,3)
    tb=valid&(P[...,0]>.15)&(P[...,0]<.75)&(np.abs(P[...,1])<.7)
    n,dd=table_plane(P,tb);print('pitch+',dp,'n',n.round(4),'table z@.4',round(-(dd+n[0]*.4)/n[2],4))
print(ARM.J['head_pitch_joint'])
