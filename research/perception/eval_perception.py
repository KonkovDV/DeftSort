"""Pool-3/6 perception evaluation on the render set using v4 sensor-only plane+segment.
Features from depth+RGB only. Ground-truth masks used ONLY for labelling/evaluation. Seed-split train/test."""
import sys,json,glob,os,numpy as np
sys.path.insert(0,'/workspace/project/workspace/team048-standalone/src/team048_manipulation')
from team048_manipulation.perception import table_plane,segment
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)));from merge import merge_fragments
MERGE=os.environ.get('MERGE','1')=='1'
D=sys.argv[1]
def deproject(dmm,intr,cam):
    h,w=dmm.shape;fx=intr['fx'];u,v=np.meshgrid(np.arange(w),np.arange(h));Z=dmm/1000.
    X=(u-intr['cx'])*Z/fx;Y=(v-intr['cy'])*Z/fx
    e=np.array(cam['eye']);t=np.array(cam['target']);f=(t-e)/np.linalg.norm(t-e);r=np.cross(f,[0,0,1]);r/=np.linalg.norm(r);up=np.cross(r,f)
    P=e+X[...,None]*r+Y[...,None]*(-up)+Z[...,None]*f
    return P,dmm>0
def feats(rgb,cloud_pix,P,plane):
    pass
if __name__=='__main__':
    rows=[]
    for js in sorted(glob.glob(D+'/p*_s*.json')):
        m=json.load(open(js));z=np.load(js[:-5]+'.npz');rgb,dmm,mask=z['rgb'],z['depth_mm'],z['mask']
        P,valid=deproject(dmm.astype(float),m['intrinsics'],m['camera'])
        try:plane=table_plane(P,valid)
        except ValueError as e:rows.append({'file':js,'plane_error':str(e)});continue
        roi=(np.array([0.15,-0.65,0.70]),np.array([0.65,0.65,1.2]))
        dets=segment(P,rgb,valid,plane,roi,min_pixels=40)
        if MERGE:dets=merge_fragments(dets,P,rgb,plane)
        id2obj={o['mask_id']:o for o in m['objects']}
        for d in dets:
            ids=mask[d.pixels[:,0],d.pixels[:,1]];ids=ids[ids>0]
            if len(ids)==0:lab,pur=None,0.
            else:
                k=np.bincount(ids).argmax();pur=float((ids==k).mean());lab=id2obj[int(k)]
            hsv=None
            px=rgb[d.pixels[:,0],d.pixels[:,1]].astype(float)/255
            hist=np.histogramdd(px,bins=(4,4,4),range=((0,1),)*3)[0].ravel();hist/=hist.sum()
            rows.append({'file':os.path.basename(js),'pool':m['pool'],'seed':m['seed'],'f6':d.features.tolist(),'hist':hist.tolist(),'npix':len(d.pixels),
                         'name':lab['name'] if lab else None,'category':lab['category'] if lab else None,'purity':pur,'centroid':d.centroid.tolist(),'gt_xy':lab['xy_yaw'][:2] if lab else None})
        n_gt=sum(1 for o in m['objects'] if o['pixels']>40);rows.append({'file':os.path.basename(js),'pool':m['pool'],'seed':m['seed'],'n_gt_visible':n_gt,'n_det':len(dets)})
    json.dump(rows,open(os.environ.get('OUTF','features.json'),'w'))
    print('rows',len(rows))
    