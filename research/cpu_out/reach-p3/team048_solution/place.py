import numpy as np
from team048_solution.trays import TRAY_H
RETREAT_DZ=.08
PLACE_AXES=(np.array([1.,0,0]),np.array([0,1.,0]),np.array([-1.,0,0]),np.array([0,-1.,0]))
def plan_place(A,q_from,tx,ty,top,fallback=None):
    """Release pose over (tx,ty) at the lowest feasible height, plus a straight-up retreat pose (or None).
    Closing axis tried with BOTH signs: the Inspire hand is not symmetric (thumb vs fingers), the right arm reaches
    tray_right only with the mirrored axis (offline 2026-11-09: +x/+y -> IK fails for every height, -y -> ok).
    v6 root cause: no box was ever picked because tray_right had no release pose. fallback=(x,y) (tray centre)
    is tried when the slot position is out of reach."""
    for (x,y) in [(tx,ty)]+([fallback] if fallback is not None else []):
        for dz in (.20,.25,.30):            # unchanged heights; lowering needs payload-vs-tray-wall data (TODO)
            c=np.array([x,y,top+TRAY_H+dz-.1])
            for ca in PLACE_AXES:
                T=A.grasp_pose(c,ca,np.array([0,0,-1.]))
                q=A.solve(T,q_from)
                if q is None:continue
                if A.col.certify(q_from,q) and A.col.certify(q,A.home):
                    return q,A.lift(q,RETREAT_DZ)
    return None