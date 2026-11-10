import json,sys,glob,os,numpy as np
exec(open('s102.py').read().split('for i,s_')[0])
Z={};meta={}
for key,i,obj in (('utana_v7s102',8,'Utana_5_Porcelain_Ramekin_Large'),('ecoforms_v7s102',11,'Ecoforms_Plant_Container_B4_Har'),('threshold_v7s102',9,'Threshold_Ramekin_White_Porcelain')):
    s_=dets[i];cl=P[s_.pixels[:,0],s_.pixels[:,1]];f0=GR.footprint(cl,pl);c0,r0=GR.circle_fit(f0['hull'])
    Z[key+'_cloud']=cl.astype(np.float32);Z[key+'_n']=np.asarray(pl[0],float);Z[key+'_d']=np.float64(pl[1])
    meta[key]=dict(obj=obj,frame='v7 sort3 s102 start head frame (test/fixtures/v7_s102_start_frames.npz), segment by colour/size',r0=float(r0),c0=[float(x) for x in c0])
old=np.load('/workspace/DeftSort/src/team048_solution/test/fixtures/v12_rim_dishes.npz');om=json.load(open('/workspace/DeftSort/src/team048_solution/test/fixtures/v12_rim_dishes.json'))
for k in ('cloud','n','d'):Z['utana_v11b_'+k]=old['utana_v11b_'+k]
meta['utana_v11b']=om['utana_v11b'];meta['utana_v11b']['frame']='v11b sort2 s101 trace 0098.15 (last head frame before the Utana attempt at log :1236)'
meta['utana_v11b']['start_line']='Kaggle v11b run_sort2_101.log:1236, at_grasp :1276'
np.savez_compressed('/workspace/DeftSort/src/team048_solution/test/fixtures/v12_rim_dishes.npz',**Z)
json.dump(meta,open('/workspace/DeftSort/src/team048_solution/test/fixtures/v12_rim_dishes.json','w'),indent=1)
print({k:round(v['r0'],4) for k,v in meta.items()})
