import arm,numpy as np
def why(A,q):
    fr=A.chain.frames(q);W=[(s,fr[s.link][:3,:3]@s.center+fr[s.link][:3,3]) for s in A.col.spheres];out=[]
    for s,p in W:
        for e in A.col.environment:
            d=np.linalg.norm(e.center-p)-s.radius-e.radius-A.col.margin
            if d<=0:out.append((s.link,'ENV',np.round(e.center,2),round(d,3)))
    for i,(s,p) in enumerate(W):
        for t,r in W[i+1:]:
            if s.link==t.link or frozenset((s.link,t.link)) in A.col.ignored:continue
            d=np.linalg.norm(p-r)-s.radius-t.radius-A.col.margin
            if d<=0:out.append((s.link,t.link,round(d,3)))
    return out
