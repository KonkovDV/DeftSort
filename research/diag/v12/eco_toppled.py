import json,sys,glob,os,numpy as np
exec(open('replay_rim.py').read().split('out=[]')[0])
tr='/workspace/kaggle-out/team048-sim-v11b/trace/sort2_101'
pc=np.array([0.0,0.0])
L=open('/workspace/kaggle-out/team048-sim-v11b/logs/run_sort2_101.log').read().split('\n')
st=json.loads(L[1384].split('TRACE ',1)[1]);pc=np.array(st['center'][:2]);print('start centre',pc)
for f in sorted(glob.glob(tr+'/*.npz')):
    P,pl,dets=scene(f);best=None
    for s_ in dets:
        cl=P[s_.pixels[:,0],s_.pixels[:,1]]
        try:f0=GR.footprint(cl,pl);c0,r0=GR.circle_fit(f0['hull'])
        except Exception:continue
        dd=np.linalg.norm(cl[:,:2].mean(0)-pc)
        if dd<.12 and (best is None or dd<best[0]):best=(dd,cl,c0,r0,f0)
    if best:
        _,cl,c0,r0,f0=best;fc,C=GR.candidates(cl,pl,'dish',(),mass=.1)
        print(os.path.basename(f)[:20],'r0',round(r0,4),'2r/wmax',round(2*r0/.10095,2),'h',round(f0['height'],3),'w',round(f0['width'],3),'L',round(f0['length'],3),'closed',fc['rim_closed'])
