from v7_proj import *
import glob,json
for f in sorted(glob.glob('/workspace/fixes/kaggle-sim/out_v7/trace/*/*.npz')):
    side=os.path.basename(f).split('_')[2];z=np.load(f);rec=json.loads(str(z['rec']))
    q=dict(zip([f'{n}_{side}_joint' for n in ARM.ARM],z['q']))
    R=ARM.fk_link(f'wrist_roll_{side}_link',q)[:3,:3]
    F=np.array(rec['F']);Fp=R@F
    Rh=ARM.fk_link(ARM.SIDE[side]['tip'],q)[:3,:3]
    print(os.path.basename(os.path.dirname(f))[-3:],rec['phase'][:8].ljust(8),rec['obj'][:12].ljust(12),'F_wrist',np.round(F,2),'F_pelvis',np.round(Fp,2),'|F|',round(np.linalg.norm(F),2),'tcp z-axis',np.round(Rh[:,2],2))
