import json,sys,glob,os,numpy as np
exec(open('replay_rim.py').read().split('out=[]')[0])
z=np.load(R+'/test/fixtures/v7_s102_start_frames.npz')
T=ARM.fk_link('camera_head_link',{'head_pitch_joint':float(z['head_pitch'])});P=PE.deproject(z['head_depth'],K,T);v=z['head_depth']>0
reg=PE.table_region(P,v);pl=PE.table_plane(P,reg)
dets=PE.segment(P,z['head_rgb'],reg,pl,(np.array([.18,-.62,-1.]),np.array([.62,.62,1.])),min_pixels=40)
for i,s_ in enumerate(dets):
    cl=P[s_.pixels[:,0],s_.pixels[:,1]];f0=GR.footprint(cl,pl);c0,r0=GR.circle_fit(f0['hull']);col=z['head_rgb'][s_.pixels[:,0],s_.pixels[:,1]].mean(0)
    print(i,len(cl),np.round(c0,3),'r0',round(r0,3),'w',round(f0['width'],3),'L',round(f0['length'],3),'h',round(f0['height'],3),'rgb',np.round(col).astype(int))
