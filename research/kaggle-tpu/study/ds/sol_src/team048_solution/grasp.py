"""Sensor-only grasp candidate generation for a segmented tabletop object (pelvis frame, z up).
Strategies (chosen by recognised category, never by ground-truth pose):
  side  : horizontal approach, closing across the minimal caliper axis of the visible footprint (boxes, jars)
  rim   : top-down pinch on the rim of a wide low object (dishes), thumb inside / fingers outside
  top   : vertical approach onto a box/electronics, closing across a footprint axis that fits the hand

Hand opening W_MAX and finger geometry are PARAMETERS: the Inspire hand limits must come from the robot
assets/SDK (not guessed here).

Offline evidence (fixes/kaggle-tpu/study, 100 layouts, body_yaw place, Rust accel, 2026-11-09):
  dish base rim-only ok=0.880; UNION of rim angles at 30/60/90 deg -> 1.000 (Ecoforms 0.69->1.00).
  pool3 box side-only 0.639; topdown_pinch 0.913 (+104 new). pool6 box topdown_wide 0.947 (gelatin 0->0.84).
  Razer mouse (min width 10 cm) remains 0: wider than opposed_open-margin and mesh pad_gap_open 7.8 cm.
"""
import numpy as np
from scipy.spatial import ConvexHull
# Rim pinch: grip centre BELOW the rim top so the finger pads overlap the wall instead of touching its edge.
# v6 (Kaggle, old code z=rim top): 0/3 rim picks ended in a tray. Value = geometric choice (pad overlap), UNVERIFIED:
# the next run logs wrist force and finger positions at every grasp to confirm or recalibrate it.
RIM_DEPTH=.015
# Extra rim samples around the fitted circle (deg). Near-robot point (0) is always first / highest score.
# Study: rim_ang +/-30 each rescued +36 of 300 that base missed; union of all angles covers every dish.
RIM_ANGLE_DEG=(0,30,-30,60,-60,90,-90)
# Top-down depth below the object top (m). Study topdown_shallow (.01) ≈ topdown (.03) for ok-rate; keep .03.
TOP_DEPTH=.03

def footprint(cloud,plane):
    n,d=plane;h=cloud@n+d;xy=cloud[:,:2]
    hull=xy[ConvexHull(xy).vertices];best=(9.,0.)
    for a in np.linspace(0,np.pi,180,endpoint=False):
        u=np.array([np.cos(a),np.sin(a)]);p=hull@u;w=p.max()-p.min()
        if w<best[0]:best=(w,a)
    w,a=best;u=np.array([np.cos(a),np.sin(a)]);v=np.array([-u[1],u[0]]);L=(hull@v).max()-(hull@v).min()
    c=np.array([(hull@u).max()+(hull@u).min(),(hull@v).max()+(hull@v).min()])/2;center=c[0]*u+c[1]*v
    return dict(width=float(w),length=float(L),axis=u,center=center,height=float(np.quantile(h,.98)),hull=hull)

def circle_fit(xy):
    A=np.c_[2*xy,np.ones(len(xy))];b=(xy**2).sum(1);c,*_=np.linalg.lstsq(A,b,rcond=None);r=np.sqrt(c[2]+c[0]**2+c[1]**2);return c[:2],r

def candidates(cloud,plane,category,others=(),W_MAX=.09,finger=.02,depth=.04,margin=.01,robot_xy=(0.,0.)):
    f=footprint(cloud,plane);out=[]
    if category=='jar':   # visible arc of a cylinder under-estimates the diameter: circle fit on mid-height slice
        h=cloud@plane[0]+plane[1];sl=cloud[np.abs(h-.5*f['height'])<.01,:2]
        if len(sl)>30:
            c,r=circle_fit(sl)
            res=np.std(np.linalg.norm(sl-c,axis=1))
            if .015<r<.06 and res<.004 and f['width']<=2*r<=1.15*f['length']:f['width']=f['length']=float(2*r);f['center']=c
    toward=np.asarray(robot_xy,float)-f['center'];toward/=np.linalg.norm(toward)+1e-9
    if category in ('box','jar','electronics'):
        z=max(.5*f['height'],min(.04,f['height']-.01))
        for ax,w in ((f['axis'],f['width']),(np.array([-f['axis'][1],f['axis'][0]]),f['length'])):
            if w+2*margin>W_MAX:continue
            appr=np.array([-ax[1],ax[0]]);appr*=-(np.sign(appr@toward) or 1)   # fingers move away from the robot (approach from the robot side)
            out.append(dict(kind='side',center=f['center'],close_axis=ax,approach=appr,width=w,z=z,
                            score=(W_MAX-w)/W_MAX-.5*float(appr@toward)))
        # Top-down: same width gate as plan_pick (opposed_open-margin). Prefer side when both exist (lower score).
        # Offline: recovers gelatin (8.9 cm) when side margin gate rejects it; Razer (10 cm) still rejected.
        if category in ('box','electronics'):
            for ax,w in ((f['axis'],f['width']),(np.array([-f['axis'][1],f['axis'][0]]),f['length'])):
                if w>W_MAX:continue
                out.append(dict(kind='top',center=f['center'],close_axis=np.r_[ax,0.],approach3=np.array([0.,0.,-1.]),
                                width=float(w),z=max(f['height']-TOP_DEPTH,.5*f['height']),
                                score=.35+(W_MAX-w)/W_MAX))   # below a typical clear side (~1.0+)
    if category=='dish':
        hull=f['hull'];c0,r0=circle_fit(hull) if len(hull)>5 else (f['center'],float(f['width'])/2)
        a0=float(np.arctan2(toward[1],toward[0]))
        z=f['height']-min(RIM_DEPTH,.3*f['height'])
        for i,deg in enumerate(RIM_ANGLE_DEG):
            a=a0+np.deg2rad(deg);p=c0+r0*np.array([np.cos(a),np.sin(a)])
            rad=(p-c0)/ (np.linalg.norm(p-c0)+1e-9)
            # close along the radius (toward the bowl centre), approach straight down
            out.append(dict(kind='rim',center=p,close_axis=rad,approach=np.array([0.,0.,-1.]),width=.025,z=z,
                            score=1.0-.02*i))   # prefer the near-robot rim point
    for g in out:   # clearance of finger corridors against other segments' points (sensor data)
        c=g['center'];ax=np.asarray(g['close_axis'],float)[:2];half=g['width']/2+margin
        tips=[c+ax*half,c-ax*half] if g['kind'] in ('side','top') else [np.asarray(g['center'],float)]
        clr=9.
        for o in others:
            for t in tips:clr=min(clr,float(np.min(np.linalg.norm(o[:,:2]-t,axis=1))) if len(o) else 9.)
        g['clearance']=clr;g['feasible']=clr>finger
        g['score']+=min(clr,.05)*10
    return f,sorted(out,key=lambda g:-g['score'])
