import json,glob,sys,os,numpy as np,collections
sys.path.insert(0,'/workspace/project/workspace/team048-standalone/src/team048_manipulation');sys.path.insert(0,'/workspace/fixes/perception');sys.path.insert(0,'.')
from team048_manipulation.perception import table_plane,segment
from eval_perception import deproject
from grasp import candidates
PRI=json.load(open('object_priors.json'))
def gt_width(name,yaw_deg,axis):
    h=np.array(PRI[name]['hull_xy']);a=np.radians(yaw_deg);R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]]);w=(h@R.T)@axis;return w.max()-w.min()
res=collections.defaultdict(list);files=sorted(glob.glob(sys.argv[1]+'/p3_s*.json'))[:int(sys.argv[2]) if len(sys.argv)>2 else None]
for js in files:
    m=json.load(open(js));z=np.load(js[:-5]+'.npz');rgb,dmm,mask=z['rgb'],z['depth_mm'].astype(float),z['mask']
    P,valid=deproject(dmm,m['intrinsics'],m['camera']);plane=table_plane(P,valid)
    dets=segment(P,rgb,valid,plane,(np.array([.15,-.65,.7]),np.array([.65,.65,1.2])),min_pixels=40)
    id2={o['mask_id']:o for o in m['objects']}
    clouds=[P[d.pixels[:,0],d.pixels[:,1]] for d in dets]
    for i,d in enumerate(dets):
        ids=mask[d.pixels[:,0],d.pixels[:,1]];ids=ids[ids>0]
        if not len(ids):continue
        k=np.bincount(ids).argmax()
        if (ids==k).mean()<.9:continue
        o=id2[int(k)];cat=o['category']
        f,C=candidates(clouds[i],plane,cat,[c for j,c in enumerate(clouds) if j!=i],robot_xy=m['camera']['eye'][:2])
        gw=gt_width(o['name'],o['xy_yaw'][2],f['axis'])
        res[cat].append(dict(name=o['name'],w_est=f['width'],w_gt=gw,w_min_gt=PRI[o['name']]['w_min'],n=len(C),feasible=sum(g['feasible'] for g in C),
                              best=C[0]['kind'] if C else None,h_est=f['height'],h_gt=PRI[o['name']]['height']))
json.dump(res,open('grasp_eval.json','w'))
for cat,R in res.items():
    e=np.array([r['w_est']-r['w_gt'] for r in R]);eh=np.array([r['h_est']-r['h_gt'] for r in R]);ax=np.array([r['w_gt']-r['w_min_gt'] for r in R])
    print(f"{cat:5s} n={len(R)} width err (est-gt along chosen axis) median {np.median(e)*100:.1f} cm, |p95| {np.percentile(abs(e),95)*100:.1f} cm; axis subopt (gt w - gt wmin) median {np.median(ax)*100:.1f} cm; height err median {np.median(eh)*100:.1f} cm; no-candidate {sum(r['n']==0 for r in R)}, no-feasible {sum(r['feasible']==0 for r in R)}")
