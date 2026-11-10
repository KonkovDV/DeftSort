from v7_geom import *
f=sorted(glob.glob('/workspace/fixes/kaggle-sim/out_v7/trace/sort3_102/*.npz'))[0];z=np.load(f);P,valid=cloud(z)
T=ARM.fk_link('camera_head_link',{'head_pitch_joint':float(z['head_pitch'])});c=T[:3,3]
tb=valid&(P[...,0]>.15)&(P[...,0]<.75)&(np.abs(P[...,1])<.7)
n,d=table_plane(P,tb);print('cam origin',c.round(3),'n',n.round(4),'d',round(d,4))
def roty(a):ca,sa=np.cos(a),np.sin(a);return np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]])
def rotx(a):ca,sa=np.cos(a),np.sin(a);return np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]])
for a in np.radians([3.5,3.8,4.0,4.2,4.5]):
  for sgn in (1,-1):
    R=roty(sgn*a);Q=(P-c)@R.T+c
    n2,d2=table_plane(Q,tb);zt=-(d2+n2[0]*.4)/n2[2]
    print(round(np.degrees(sgn*a),2),'n',n2.round(4),'table z',round(zt,4))
