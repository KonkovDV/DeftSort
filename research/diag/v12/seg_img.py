import json,sys,glob,os,numpy as np
from PIL import Image
exec(open('replay_rim.py').read().split('out=[]')[0])
f='/workspace/kaggle-out/team048-sim-v11b/trace/sort2_101/0130.37_released_l_Utana_5_Porcelain_Ra.npz'
P,pl,dets=scene(f);img=np.load(f)['rgb'][...,:3].copy();pc=np.array([0.381,0.413])
for s_ in dets:
    cl=P[s_.pixels[:,0],s_.pixels[:,1]];f0=GR.footprint(cl,pl);c0,r0=GR.circle_fit(f0['hull'])
    print('seg px',len(cl),'centroid',np.round(cl[:,:2].mean(0),3),'r0',round(r0,3),'w',round(f0['width'],3),'L',round(f0['length'],3),'h',round(f0['height'],3))
    if np.linalg.norm(cl[:,:2].mean(0)-pc)<.12:img[s_.pixels[:,0],s_.pixels[:,1]]=[255,0,255]
Image.fromarray(img).save('/workspace/v12/seg130.png')
