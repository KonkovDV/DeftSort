"""Sensor-only grasp candidate generation for a segmented tabletop object (pelvis frame, z up).
Strategies (chosen by recognised category, never by ground-truth pose):
  side  : horizontal approach, closing across the minimal caliper axis of the visible footprint (boxes, jars)
  rim   : top-down pinch on the near rim of a wide low object (dishes), thumb inside / fingers outside
Hand opening W_MAX and finger geometry are PARAMETERS: the Inspire hand limits must come from the robot
assets/SDK (not guessed here)."""
import numpy as np
from scipy.spatial import ConvexHull
# Rim pinch: grip centre BELOW the rim top so the finger pads overlap the wall instead of touching its edge.
# v6 (Kaggle, old code z=rim top): 0/3 rim picks ended in a tray. Value = geometric choice (pad overlap), UNVERIFIED:
# the next run logs wrist force and finger positions at every grasp to confirm or recalibrate it.
RIM_DEPTH=.015
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
    toward=np.asarray(robot_xy)-f['center'];toward/=np.linalg.norm(toward)+1e-9
    if category in ('box','jar','electronics'):
        z=max(.5*f['height'],min(.04,f['height']-.01))
        for ax,w in ((f['axis'],f['width']),(np.array([-f['axis'][1],f['axis'][0]]),f['length'])):
            if w+2*margin>W_MAX:continue
            appr=np.array([-ax[1],ax[0]]);appr*=-(np.sign(appr@toward) or 1)   # fingers move away from the robot (approach from the robot side)
            out.append(dict(kind='side',center=f['center'],close_axis=ax,approach=appr,width=w,z=z,
                            score=(W_MAX-w)/W_MAX-.5*float(appr@toward)))
    if category=='dish':
        r=f['hull'];k=np.argmax(r@toward);p=r[k]   # nearest rim point toward the robot
        out.append(dict(kind='rim',center=p,close_axis=toward,approach=np.array([0,0,-1.]),width=.025,z=f['height']-min(RIM_DEPTH,.3*f['height']),score=1.))
    for g in out:   # clearance of finger corridors against other segments' points (sensor data)
        c=g['center'];ax=g['close_axis'];half=g['width']/2+margin
        tips=[c+ax*half,c-ax*half] if g['kind']=='side' else [g['center']]
        clr=9.
        for o in others:
            for t in tips:clr=min(clr,float(np.min(np.linalg.norm(o[:,:2]-t,axis=1))) if len(o) else 9.)
        g['clearance']=clr;g['feasible']=clr>finger
        g['score']+=min(clr,.05)*10
    return f,sorted(out,key=lambda g:-g['score'])
