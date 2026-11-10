import sys,time,numpy as np;sys.argv=['x'];sys.path.insert(0,'.')
import cycle2 as c
from team048_solution import perception as PE, _accel as AC, node as NODE
z=np.load(c.FRAMES[0]);dd=z['depth'];dd=dd[...,0] if dd.ndim==3 else dd;dmm=dd.astype(float);rgb=z['rgb']
T=c.ARM.fk_link('camera_head_link',{'head_pitch_joint':float(z['head_pitch'])})
def tm(f,n=30):
    f();t=time.perf_counter()
    for _ in range(n):f()
    return 1e3*(time.perf_counter()-t)/n
P=c.deproject(dmm,T);valid=dmm>0;tb=PE.table_region(P,valid)
r=AC.ransac_plane(P,tb,0,120,.006);p,planes,k,cnt=r
print('npts',len(p),'cands',len(planes),'best',k,cnt)
print('ransac_plane rust %.2f ms'%tm(lambda:AC.ransac_plane(P,tb,0,120,.006)))
n_=planes[k,:3].copy();o=planes[k,3]
print('inliers numpy %.2f'%tm(lambda:abs(p@n_+o)<.006))
inl=abs(p@n_+o)<.006
def ref():
    cloud=p[inl];center=cloud.mean(axis=0);np.linalg.svd(cloud-center,full_matrices=False)
print('refine (index+mean+svd) %.2f'%tm(ref))
print('table_plane total %.2f'%tm(lambda:PE.table_plane(P,tb)))
dets=PE.segment(P,rgb,tb,PE.table_plane(P,tb),(np.array([.18,-.62,-1.]),np.array([.62,.62,1.])),min_pixels=40)
S=c._S();print('identify %.2f ms (%d dets)'%(tm(lambda:NODE.Sol.identify(S,dets,P,PE.table_plane(P,tb),list(NODE.PRI))),len(dets)))
print('ransac iters=0 (compaction+array out) %.2f'%tm(lambda:AC.ransac_plane(P,tb,0,0,.006)))
print('ransac iters=120 parallel-off counts all (old kernel) %.2f'%tm(lambda:AC._ext.plane_inlier_counts(p,planes,.006,False)))
