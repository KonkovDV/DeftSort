import sys,os,json,numpy as np
T_='/workspace/DeftSort/src/team048_solution';sys.path.insert(0,T_);sys.path.insert(0,T_+'/test')
import test_node_stub as TS
from team048_solution import arm as ARM, grasp as GR, node as N, trays as TR
import scene_fixture as SF
print('trays',{k:v for k,v in SF.TRAYS.items()})
A=ARM.Arm('l');A.set_scene(SF.plane(),list(SF.TRAYS.values()))
r0=.05;c0=np.array([.36,.25]);tw=-c0/np.linalg.norm(c0);p=c0+r0*tw;rad=tw
g=dict(kind='rim',center=p,close_axis=rad,width=GR.RIM_W,z=.072-.015,obj_center=c0)
top=SF.TABLE_TOP;c3=np.array([p[0],p[1],top+g['z']])
pk=A.plan_pick(c3,np.r_[rad,0.],np.array([0,0,-1.]),g['width'],point='pinch')
print('pick',pk is not None)
o=N.held_offset(A,pk['q_grasp'],g,c3);print('o',np.round(o,3))
Tg=A.tcp(pk['q_grasp']);print('centre at pick',np.round((Tg[:3,:3]@o+Tg[:3,3])[:2],3),'c0',c0)
tr=SF.TRAYS['tray_left'];tx,ty,ax,ay,ts,tl,th=ARM.tray_geom(tr);print('tray',tx,ty,ax,ay,ts,tl,th)
for k in range(3):
    dl=TR.slot_offset(k,r0,size=(ts,tl),ty=ty);sx,sy=tx+dl*ax,ty+dl*ay
    for oo in (None,o):
        yp=N.plan_place_yaw(A,pk['q_lift'],sx,sy,SF.plane(),list(SF.TRAYS.values()),fallback=(tx,ty) if dl else None,tray_h=th,obj_off=oo)
        if yp is None:print(k,'none');continue
        yaw,(qp,qu,qc)=yp;Tp=A.tcp(qp);cen=(Tp[:3,:3]@o+Tp[:3,3])[:2]
        if yaw:cen=(N.rotz(yaw)@np.r_[cen,0])[:2]
        # dish in tray frame
        d=cen-np.array([tx,ty]);u=np.array([ax,ay]);v=np.array([-ay,ax]);a_=abs(d@u);b_=abs(d@v)
        inner_l=tl/2-TR.TRAY_WALL;inner_s=ts/2-TR.TRAY_WALL
        print(k,'off' if oo is None else 'on','yaw',round(yaw,2),'slot',np.round([sx,sy],3),'dish',np.round(cen,3),'margin long',round(inner_l-a_-r0,3),'short',round(inner_s-b_-r0,3))
