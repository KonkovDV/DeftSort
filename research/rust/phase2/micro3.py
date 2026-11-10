import sys,time,numpy as np,os;sys.argv=['x'];sys.path.insert(0,'.')
import cycle2 as c
from team048_solution import perception as PE, _accel as AC
z=np.load(c.FRAMES[0]);dd=z['depth'];dd=dd[...,0] if dd.ndim==3 else dd;dmm=dd.astype(float)
T=c.ARM.fk_link('camera_head_link',{'head_pitch_joint':float(z['head_pitch'])})
P=c.deproject(dmm,T);valid=dmm>0;tb=PE.table_region(P,valid)
best=[]
for rep in range(5):
    t=time.perf_counter()
    for _ in range(40):AC.ransac_plane(P,tb,0,120,.006)
    best.append(1e3*(time.perf_counter()-t)/40)
print(os.environ.get('TEAM048_ACCEL_NO_AVX2','avx2'),'min %.2f ms med %.2f'%(min(best),np.median(best)))
