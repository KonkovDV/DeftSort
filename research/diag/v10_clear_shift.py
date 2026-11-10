"""Pool 3 (+1,2,4,5,6 box/jar/electronics) side & top grasps: open-hand mesh overlap with the object box at zero shift
and the shift chosen by arm.grasp_clear_shift (priors dims = objects.yaml / USD extents)."""
import sys,json,numpy as np
sys.path.insert(0,'/workspace/solution/nssr-sol-team-048/src/team048_solution')
from team048_solution import arm as A, grasp as G
PRI=json.load(open('/workspace/solution/nssr-sol-team-048/src/team048_solution/team048_solution/data/priors.json'))
POOLS=__import__('yaml').safe_load(open('/workspace/polygon/hackathon/config/pools.yaml'))['pools']
arm=A.Arm('l')
W_MAX=min(arm.hand['opposed_open']-.01,.11)
def run(names,verbose=True):
    rows=[]
    for n in names:
        p=PRI[n];cat=p['category'];h=p['height'];w,L=p['w_min'],p['w_max']
        cyl=cat=='jar' and abs(w-L)<.006
        res=[]
        for kind,(cw,cl) in (('side',(w,L)),('side',(L,w)),('top',(w,L)),('top',(L,w))):
            if kind=='side' and (cat not in ('box','jar','electronics') or cw+.02>W_MAX):continue
            if kind=='top' and (cat not in ('box','electronics') or cw>W_MAX):continue
            box=dict(center=(.4,0.),u=(0.,1.),w=cw,L=cl,z0=-.17,h=h,cyl=cyl)
            if kind=='side':z=max(.5*h,min(.04,h-.01));ap=np.array([1.,0,0]);pt='grip'
            else:z=max(h-G.TOP_DEPTH,.5*h);ap=np.array([0,0,-1.]);pt='pinch'
            c=np.array([.4,0.,-.17+z])
            for t in ((0,.35,.6) if kind=='side' else (0,)):
                for sg in (1,-1):
                    a=ap*np.cos(t)+np.array([0,0,-np.sin(t)]) if kind=='side' else ap
                    Tg=arm.grasp_pose(c,sg*np.array([0,1.,0]),a,pt)
                    P,parts=A.hand_open_points('l');Wp=(Tg[:3,:3]@P.T).T+Tg[:3,3];B=A.box_coords(Wp,box)
                    ins=A.box_inside(B,box,0.);pen=sorted(set(parts[ins]))
                    cs=A.grasp_clear_shift('l',Tg,a,box)
                    res.append((kind,round(cw,3),t,sg,pen,None if cs is None else (cs['du'],cs['dr'],cs['margin'])))
        rows.append((n,p['mass'],cat,res))
        if verbose:
            print(n[:30],p['mass'],cat,'h',round(h,3))
            for r in res:print('   ',r)
    return rows
if __name__=='__main__':
    pools=sys.argv[1:] or ['3']
    names=[x for pid in pools for c,v in POOLS[pid].items() for x in v if PRI.get(x,{}).get('category') in ('box','jar','electronics')]
    run(names)
