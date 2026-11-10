from v7_proj import *
import glob
# For every closed/lifted frame: compare FK wrist/hand-base 3D with the depth cloud near its projection (nearest 3D point within 15 px)
for f in sorted(glob.glob('/workspace/fixes/kaggle-sim/out_v7/trace/*/*.npz')):
    side=os.path.basename(f).split('_')[2];z=np.load(f);P=hand_pts(side,z['q']);C,valid=cloud(z)
    res=[]
    for k in ('base','wrist'):
        u,v,dc=proj(z,[P[k]])[0]
        if not(15<=u<625 and 15<=v<345):res.append(f'{k}:out');continue
        ui,vi=int(u),int(v);W=C[vi-15:vi+15,ui-15:ui+15].reshape(-1,3);m=valid[vi-15:vi+15,ui-15:ui+15].ravel()
        dd=np.linalg.norm(W[m]-P[k],axis=1);res.append(f'{k}: nearest {dd.min()*100:.1f} cm')
    print(os.path.basename(os.path.dirname(f))[-3:],os.path.basename(f)[:34],res)
