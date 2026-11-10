"""CPU replay of the v7 s102 START frames (test fixture, head pitch 0.4, K from camera_info) through the node pipeline
without actuation: measured plane + near edge (torso cam) + trays (centre from the camera; the side trays are cut by the
image border at yaw 0 -> synthetic scene_fixture positions, offline only) -> segment -> identify -> candidates ->
plan_pick (open-hand clearance) -> plan_place_yaw -> timing (plan duration, place_time, hold bound)."""
import sys,os,json,time,numpy as np
R='/workspace/solution/nssr-sol-team-048/src/team048_solution';sys.path.insert(0,R);sys.path.insert(0,R+'/test')
import test_node_stub  # noqa
import scene_fixture as SF
from team048_solution import node as N, arm as ARM, trays as TR, perception as PE, grasp as GR
f=np.load(R+'/test/fixtures/v7_s102_start_frames.npz');K=f['K']
def cloud(T,dep,K):
    v,u=np.mgrid[0:dep.shape[0],0:dep.shape[1]];Z=dep/1000.;X=(u-K[2])*Z/K[0];Y=(v-K[5])*Z/K[4]
    return (np.stack([X,Y,Z],-1).reshape(-1,3)@T[:3,:3].T+T[:3,3]).reshape(dep.shape+(3,))
P=cloud(ARM.fk_link('camera_head_link',{'head_pitch_joint':float(f['head_pitch'])}),f['head_depth'].astype(float),K);valid=f['head_depth']>0
pl=PE.table_plane(P,PE.table_region(P,valid));assert ARM.plane_plausible(pl)
Pt=cloud(ARM.fk_link('camera_body_front_link'),f['torso_depth'].astype(float),K);edge=PE.near_edge(Pt,f['torso_depth']>0,pl)
trays={'tray_center':TR.detect(P,f['head_rgb'],valid,'tray_center','green',pl)}
for n in ('tray_left','tray_right'):trays[n]=(*SF.TRAYS[n],)
s=N.Sol();s.trays={'tray_center':trays['tray_center']};mask=s.tray_mask(P);v2=valid&~mask
dets=PE.segment(P,f['head_rgb'],PE.table_region(P,v2),pl,(np.array([.18,-.62,-1.]),np.array([.62,.62,1.])),min_pixels=40)
pool=['Utana_5_Porcelain_Ramekin_Large','Threshold_Ramekin_White_Porcelain','Ecoforms_Plant_Container_B4_Har','Crunch_Girl_Scouts_Candy_Bars_Peanut_Butter_Creme_78_oz_box',
      'Crayola_Bonus_64_Crayons','Perricone_MD_AcylGlutathione_Eye_Lid_Serum','006_mustard_bottle','Saccharomyces_Boulardii_MOS_Value_Size','Hyaluronic_Acid']
cats={'dish':'tray_left','box':'tray_right','jar':'tray_center'}
ids=s.identify(dets,P,pl,pool)
arms={k:ARM.Arm(k) for k in 'lr'}
TL=[trays['tray_left'],trays['tray_right'],trays['tray_center']]
for A in arms.values():A.set_scene(pl,TL,(edge,ARM.TABLE_X_FAR))
print('plane n',np.round(pl[0],3),'edge',round(edge,3),'dets',len(dets),'identified',len(ids))
rows=[]
for det,name,cat,conf,cl,cost in ids:
    if not(conf>N.ID_CONF_MIN and cost<N.ID_COST_MAX):print('  skip low conf',name[:20],round(conf,2),round(cost,2));continue
    others=[P[d.pixels[:,0],d.pixels[:,1]] for d in dets if d is not det]
    t0=time.time();fp,C=GR.candidates(cl,pl,N.PRI[name]['category'],others,W_MAX=min(arms['l'].hand['opposed_open']-.01,.11))
    tr=ARM.tray_geom(trays[cats[N.PRI[name]['category']]] if not isinstance(trays[cats[N.PRI[name]['category']]],tuple) else trays[cats[N.PRI[name]['category']]])
    tx,ty=tr[0],tr[1];res=None
    for side in ('lr' if ty>.1 else 'rl' if ty<-.1 else ('lr' if fp['center'][1]>=0 else 'rl')):
        A=arms[side]
        for g in C:
            if not g['feasible']:continue
            top=-(pl[1]+pl[0][:2]@g['center'])/pl[0][2]
            if g['kind']=='side':ap=np.r_[g['approach'],0.];ca=np.r_[g['close_axis'],0.];pt='grip'
            else:ap=np.asarray(g.get('approach3',[0,0,-1.]),float);ca=np.asarray(g['close_axis'],float);ca=np.r_[ca,0.] if ca.size==2 else ca;pt='pinch'
            p=None
            for dz in (N.SIDE_DZ if g['kind']=='side' else (0.,)):
                if g['kind']=='side' and dz>0 and g['z']+dz>fp['height']-.015:break
                c3=np.array([g['center'][0],g['center'][1],top+g['z']+dz])
                p=A.plan_pick(c3,ca,ap,g['width'],point=pt,obj=None if ('box' not in g or os.environ.get('NOCLEAR')) else dict(g['box'],z0=top))
                if p is not None:break
            if p is None:continue
            yp=N.plan_place_yaw(A,p['q_lift'],tx,ty,pl,TL)
            if yp is None:continue
            yaw,place=yp;need=N.place_time(A,p['q_lift'],place,yaw,q_from=p['q_grasp']);hold=N.hold_time_bound(p['q_grasp'],p['q_lift'],place,yaw)
            res=dict(side=side,kind=g['kind'],tilt=p['tilt'],clear=None if p['clear'] is None else {k:p['clear'][k] for k in ('du','dr','margin')},
                     pick_s=round(p['duration'],1),need=round(need,1),hold=round(hold,1),yaw=round(yaw,2));break
        if res:break
    print('  ',name[:22].ljust(22),'conf',round(conf,2),'w',round(fp['width'],3),'L',round(fp['length'],3),'h',round(fp['height'],3),'->',res or 'NO PLAN','%.2fs'%(time.time()-t0))
