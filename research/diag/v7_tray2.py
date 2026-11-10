from v7_tray import *
for f in sorted(glob.glob('/workspace/fixes/kaggle-sim/out_v7/trace/sort3_10*/*closed*.npz'))[:4]:
    z=np.load(f);P,valid=cloud(z);rgb=z['rgb']
    T=ARM.fk_link('camera_head_link',{'head_pitch_joint':float(z['head_pitch'])});c=T[:3,3]
    tb=valid&(P[...,0]>.15)&(P[...,0]<.75)&(np.abs(P[...,1])<.7)
    Q,R,dz=correct(P,tb,c)
    for lab,A in (('raw',P),('corr',Q)):
        n,d=table_plane(A,tb);h=A@n+d
        m=TR.chroma_mask(rgb,'green',.03)&valid&(h>.05)&(h<.09)&(np.abs(A[...,1])<.3)&(A[...,0]>.3)&(A[...,0]<.8)
        pts=A[m]
        print(os.path.basename(f)[:28],lab,'n',m.sum(),'x',np.quantile(pts[:,0],[.01,.99]).round(3),'y',np.quantile(pts[:,1],[.01,.99]).round(3),'wall h',np.median(h[m]).round(3))
