from v7_proj import *
import glob,json
for f in sorted(glob.glob('/workspace/fixes/kaggle-sim/out_v7/trace/*/*closed*.npz')):
    side=os.path.basename(f).split('_')[2];z=np.load(f);rec=json.loads(str(z['rec']))
    C,valid=cloud(z);tb=valid&(C[...,0]>.15)&(C[...,0]<.75)&(np.abs(C[...,1])<.7);n,d=table_plane(C,tb)
    q=dict(zip([f'{k}_{side}_joint' for k in ARM.ARM],z['q']));pre='left' if side=='l' else 'right'
    links=[f'{pre}_{a}_{b}' for a in ('index','middle','ring','little') for b in (1,2)]+[f'{pre}_thumb_{i}' for i in (1,2,3,4)]+[ARM.SIDE[side]['hand']]
    hs=[];
    for L in links:
        p=ARM.fk_link(L,q)[:3,3];hs.append((p@n+d,L))
    # fingertips estimates
    for fn in ('thumb','index','middle','ring','little'):
        try:
            t=ARM.fingertip(side,fn,q);H=ARM.fk_link(ARM.SIDE[side]['hand'],q);p=(H@np.r_[t,1])[:3];hs.append((p@n+d,fn+'_tip'))
        except Exception as e:pass
    hs.sort()
    c=np.array(rec.get('tcp'))
    print(os.path.basename(os.path.dirname(f))[-3:],rec['obj'][:12].ljust(12),'lowest above table (cm):',[(round(h*100,1),L.replace(pre+'_','')) for h,L in hs[:3]],'| table-plane flat model error at hand x: %.1f cm'%(100*((-d-n[0]*c[0])/n[2]+.18)))
