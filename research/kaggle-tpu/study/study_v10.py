"""study_core reach mode with the node's open-hand clearance (plan_pick obj=box), toggled by V10_CLEAR=1.
Usage: STUDY_CFG=... V10_CLEAR=0|1 python study_v10.py"""
import os,sys,json
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import study_core as S, numpy as np
CLEAR=os.environ.get('V10_CLEAR','1')=='1'
_LAST={}
def try_pick(A,g,f,top,tilts=(0,.35,.6),point=None,side_dz=None,pre=.08):
    NODE=S.W['NODE']
    ap=np.r_[g['approach'],0.] if g['kind']=='side' else np.asarray(g.get('approach3',[0,0,-1.]),float)
    ca=np.r_[g['close_axis'],0.] if len(g['close_axis'])==2 else np.asarray(g['close_axis'],float)
    pt=point or ('pinch' if g['kind'] in ('rim','top') else 'grip')
    for dz in (side_dz if side_dz is not None else (NODE.SIDE_DZ if g['kind']=='side' else (0.,))):
        if g['kind']=='side' and dz>0 and g['z']+dz>f['height']-.015:break
        c3=np.array([g['center'][0],g['center'][1],top+g['z']+dz])
        obj=dict(g['box'],z0=top) if (CLEAR and 'box' in g) else None
        p=A.plan_pick(c3,ca,ap,g['width'],point=pt,tilts=tilts,pre=pre,obj=obj)
        if p is not None:
            _LAST['clear']=p.get('clear')
            if 'box' in g:   # overlap of the open hand with the measured box at the PLANNED pose (zero shift)
                t=p['tilt'];a=ap*np.cos(t)+np.array([0,0,-np.sin(t)]) if (abs(ap[2])<.9) else ap
                Tg=A.tcp(p['q_grasp']);ARM=S.W['ARM']
                _LAST['overlap0']=ARM.grasp_clear_shift(A.side,Tg,a,dict(g['box'],z0=top),pre=pre,margins=(0.,),su=(0.,),sr=(0.,)) is None
                _LAST['overlap_grasp']=ARM.grasp_clear_shift(A.side,Tg,a,dict(g['box'],z0=top),pre=0.,margins=(0.,),su=(0.,),sr=(0.,)) is None
            return p,dz
    return None,None
S.try_pick=try_pick
_eval=S.eval_obj
def eval_obj(job):
    _LAST.clear();r=_eval(job);c=_LAST.get('clear')
    if r.get('result')=='ok' and c:r.update(du=c['du'],dr=c['dr'],margin=c['margin'])
    if r.get('result')=='ok' and 'overlap0' in _LAST:r.update(overlap_sweep=_LAST['overlap0'],overlap_grasp=_LAST['overlap_grasp'])
    return r
S.eval_obj=eval_obj
if __name__=='__main__':S.run()
