"""Figures for the README from the recorded head frame shipped in test/fixtures (run v7, sort3 seed 102, start of
the episode). Offline only; needs matplotlib (not a runtime dependency).
  docs/img/perception.png  head camera frame: measured table plane, trays found by colour + depth, segments above
                           the table (the pipeline of node.observe on this frame)
  docs/img/plan.png        the scene measured from the same frame and a planned left-arm pick: arm collision spheres
                           at the grasp pose, TCP path, table and tray-wall spheres
Usage: python3 tools/make_figures.py   (from the repository root)"""
import os,sys
import numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0]=[os.path.join(ROOT,'src','team048_solution')]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Polygon
from team048_solution import arm,perception as PE,trays as TR
OUT=os.path.join(ROOT,'docs','img');os.makedirs(OUT,exist_ok=True)
z=np.load(os.path.join(ROOT,'src','team048_solution','test','fixtures','v7_s102_start_frames.npz'))
rgb=z['head_rgb'];d=z['head_depth'].astype(float);K=z['K'].reshape(3,3)
T=arm.fk_link('camera_head_link',{'head_pitch_joint':float(z['head_pitch'])})
P=PE.deproject(d,K,T);valid=d>0
plane=PE.table_plane(P,PE.table_region(P,valid))
COL={'tray_left':'red','tray_right':'blue','tray_center':'green'}
trays={n:TR.detect(P,rgb,valid,n,c,plane) for n,c in COL.items()}
def corners(t):
    x,y=t['center'];ax=np.array(t['axis']);u=np.array([ax[1],-ax[0]]);a,b=t['size']   # size=(across, along the axis)
    return np.array([[x,y]+su*a/2*u+sv*b/2*ax for su,sv in ((1,1),(-1,1),(-1,-1),(1,-1))])
def to_px(XY,zv):
    Ti=np.linalg.inv(T);pts=np.c_[XY,zv,np.ones(len(XY))]@Ti.T;c=pts[:,:3]@K.T
    return c[:,:2]/c[:,2:3]
def height(x,y):return arm.plane_height(plane,x,y)
full={n:t for n,t in trays.items() if t is not None and not t['partial']}
R=[]
for t in full.values():
    x,y,ax,ay,a,b,_=arm.tray_geom(t);v=np.array([ax,ay]);u=np.array([v[1],-v[0]]);R.append((x,y,u,v,a/2+.03,b/2+.03))
vm=valid&~PE.rect_mask(P,R) if R else valid
roi=(np.array([.18,-.62,-1.]),np.array([.62,.62,1.]))
dets=PE.segment(P,rgb,PE.table_region(P,vm),plane,roi,min_pixels=40)
n,off=plane;h=P.reshape(-1,3)@n+off
tab=(valid.reshape(-1)&(np.abs(h)<.006)).reshape(valid.shape)&PE.table_region(P,valid)

fig,ax=plt.subplots(figsize=(8,4.6),dpi=100)
img=rgb.astype(float)/255;img[tab]=.55*img[tab]+.45*np.array([1.,.85,.2])
ax.imshow(img)
for name,t in trays.items():
    if t is None:continue
    c=corners(t);px=to_px(c,np.array([height(*p)+t['height'] for p in c]))
    ax.add_patch(Polygon(px,fill=False,lw=2,ec=COL[name],ls='-' if not t['partial'] else '--'))
    ax.text(*px.mean(0),f"{name}\n{t['size'][0]*100:.0f}x{t['size'][1]*100:.0f} cm" if not t['partial'] else f'{name} (edge of frame)',color='white',fontsize=7,ha='center',bbox=dict(fc=COL[name],alpha=.6,lw=0))
for k,D in enumerate(dets):
    r,c=D.pixels[:,0],D.pixels[:,1]
    ax.add_patch(plt.Rectangle((c.min(),r.min()),c.max()-c.min(),r.max()-r.min(),fill=False,ec='cyan',lw=1))
ax.set_title(f'Head camera, run v7 start frame: table plane (yellow), trays, {len(dets)} segments above the table (cyan)',fontsize=8)
ax.axis('off');fig.tight_layout();fig.savefig(os.path.join(OUT,'perception.png'));plt.close(fig)

A=arm.Arm('l');A.set_scene(plane,list(full.values()))
_,C,Rr,_=list(A._scenes.values())[-1]
target=np.array([.35,.25,height(.35,.25)+.04])     # an object position on the measured table, left of the robot
plan=A.plan_pick(target,np.array([0,1.,0]),np.array([1.,0,0]),.05)
fig,axs=plt.subplots(1,2,figsize=(9,4.2),dpi=100)
for ax,(i,j,lab) in zip(axs,((0,1,'top view: x forward, y left (m)'),(0,2,'side view: x forward, z up (m)'))):
    for c,r in zip(C,Rr):ax.add_patch(Circle((c[i],c[j]),r,fc='0.85',ec='0.6',lw=.3))
    if plan is not None:
        tip=np.array([A.chain.fk(q)[:3,3] for q in plan['traj'].points])
        F=A.chain.frames(plan['q_grasp'])
        for s in A.col.spheres:
            p=F[s.link][:3,:3]@np.asarray(s.center)+F[s.link][:3,3];ax.add_patch(Circle((p[i],p[j]),s.radius,fc=(.2,.4,.9,.35),ec=(.1,.2,.6),lw=.4))
        ax.plot(tip[:,i],tip[:,j],'-',c='crimson',lw=1.5,label='TCP path: home, pre-grasp, grasp, lift')
    ax.plot(target[i],target[j],'*',c='orange',ms=12,label='object grip point')
    ax.set_aspect('equal');ax.set_title(lab,fontsize=8);ax.tick_params(labelsize=7)
axs[0].set_xlim(-.3,1.0);axs[0].set_ylim(-.8,.9);axs[1].set_xlim(-.3,1.0);axs[1].set_ylim(-.6,.5)
axs[0].legend(fontsize=7,loc='lower right')
dur='' if plan is None else f", path {plan['duration']:.1f} s"
fig.suptitle(f'Scene measured from the same frame (grey: table, tray walls, body) and the left arm at the grasp pose (blue spheres){dur}',fontsize=8)
fig.tight_layout();fig.savefig(os.path.join(OUT,'plan.png'));plt.close(fig)
print('perception: trays',{k:(None if v is None else ('partial' if v['partial'] else 'full')) for k,v in trays.items()},'segments',len(dets))
print('plan:', None if plan is None else dict(tilt=plan['tilt'],duration=round(plan['duration'],2),points=len(plan['traj'].points)))
