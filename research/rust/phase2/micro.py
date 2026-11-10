import sys,time,numpy as np;sys.argv=['x'];sys.path.insert(0,'.')
import cycle2 as c
from team048_solution import trays as TR, perception as PE, grasp as G, node as NODE
z=np.load(c.FRAMES[0]);dd=z['depth'];dd=dd[...,0] if dd.ndim==3 else dd;dmm=dd.astype(float);rgb=z['rgb']
T=c.ARM.fk_link('camera_head_link',{'head_pitch_joint':float(z['head_pitch'])})
def tm(f,n=20):
    f();t=time.perf_counter()
    for _ in range(n):f()
    return 1e3*(time.perf_counter()-t)/n
P=c.deproject(dmm,T);valid=dmm>0;plane=PE.table_plane(P,valid&(P[...,0]>.15)&(P[...,0]<.75)&(np.abs(P[...,1])<.7))
print('shape',P.shape)
print('deproject %.2f ms'%tm(lambda:c.deproject(dmm,T)))
print('tb0 mask %.2f'%tm(lambda:valid&(P[...,0]>.15)&(P[...,0]<.75)&(np.abs(P[...,1])<.7)))
print('locate1 %.2f'%tm(lambda:TR.locate(P,rgb,valid,'tray_center','green',plane)))
nx,ny=TR.NOMINAL['tray_center']
def win():
    m=valid&np.isfinite(P).all(axis=-1);m&=(np.abs(P[...,0]-nx)<.26)&(np.abs(P[...,1]-ny)<.31);h=P@plane[0]+plane[1];m&=(h>-.01)&(h<.09);return m
print(' locate window+height %.2f'%tm(win))
print(' isfinite all %.2f'%tm(lambda:np.isfinite(P).all(axis=-1)))
print(' P@n %.2f'%tm(lambda:P@plane[0]+plane[1]))
roi=(np.array([.18,-.62,-1.]),np.array([.62,.62,1.]))
tb=valid&(P[...,0]>.15)&(P[...,0]<.75)&(np.abs(P[...,1])<.7)
print('segment %.2f'%tm(lambda:PE.segment(P,rgb,tb,plane,roi,min_pixels=40)))
def segmask():
    n,d=plane;height=P@n+float(d);return tb&(height>.012)&(height<.35)&np.all((P>=roi[0])&(P<=roi[1]),axis=2)
print(' seg mask %.2f'%tm(segmask))
dets=PE.segment(P,rgb,tb,plane,roi,min_pixels=40);print(' ndets',len(dets),[len(d.pixels) for d in dets])
S=c._S()
print('identify %.2f'%tm(lambda:NODE.Sol.identify(S,dets,P,plane,list(NODE.PRI))))
print('table_plane %.2f'%tm(lambda:PE.table_plane(P,tb)))
