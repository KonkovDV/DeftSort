import json,sys,glob,os,numpy as np
exec(open('replay_rim.py').read().split('out=[]')[0])
KS='/workspace/DeftSort/research/kaggle-sim/'
want=[('utana_v7s102',KS+'out_v7/trace/sort3_102/0018.44_lifted_l_Utana_5_Porcelain_Ra.npz',KS+'out_v7/logs/run_sort3_102.log',1010),
      ('ecoforms_v7s103',glob.glob(KS+'out_v7/trace/sort3_103/0030.81*')[0],KS+'out_v7/logs/run_sort3_103.log',1010),
      ('threshold_v7s102',glob.glob(KS+'out_v7/trace/sort3_102/0087.28*')[0],KS+'out_v7/logs/run_sort3_102.log',1036),
      ('utana_v11b','/workspace/kaggle-out/team048-sim-v11b/trace/sort2_101/0098.15_lifted_r_Perricone_MD_AcylGlu.npz','/workspace/kaggle-out/team048-sim-v11b/logs/run_sort2_101.log',1236)]
Z={};meta={}
for key,fr,log,ln in want:
    L=open(log).read().split('\n');st=json.loads(L[ln-1].split('TRACE ',1)[1]);pc=np.array(st['center'][:2])
    P,pl,dets=scene(fr);best=None
    for s_ in dets:
        cl=P[s_.pixels[:,0],s_.pixels[:,1]]
        try:f0=GR.footprint(cl,pl);c0,r0=GR.circle_fit(f0['hull'])
        except Exception:continue
        e=abs(np.linalg.norm(pc-c0)-r0)
        if e<.03 and (best is None or e<best[0]):best=(e,cl,c0,r0)
    _,cl,c0,r0=best
    Z[key+'_cloud']=cl.astype(np.float32);Z[key+'_n']=np.asarray(pl[0],float);Z[key+'_d']=np.float64(pl[1])
    meta[key]=dict(obj=st['obj'],frame=os.path.basename(fr),start_line=f"{os.path.basename(log)}:{ln}",r0=float(r0),c0=[float(x) for x in c0])
    # at_grasp q for the FK check
    for l2 in L[ln:ln+40]:
        if 'TRACE ' in l2:
            d2=json.loads(l2.split('TRACE ',1)[1])
            if d2['phase']=='at_grasp' and d2.get('q'):meta[key].update(at_grasp_q=d2['q'],side=d2['side'],t=d2['t']);break
    print(key,len(cl),round(r0,4),'q' in str(meta[key].keys()))
np.savez_compressed('/workspace/DeftSort/src/team048_solution/test/fixtures/v12_rim_dishes.npz',**Z)
json.dump(meta,open('/workspace/DeftSort/src/team048_solution/test/fixtures/v12_rim_dishes.json','w'),indent=1)
