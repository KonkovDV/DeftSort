"""Robot-only URDF serial chain FK/IK. No scene/USD introspection."""
from dataclasses import dataclass,replace
import xml.etree.ElementTree as ET
import numpy as np
from .geometry import finite, pose, rotation, check_transform, rotation_error


@dataclass
class Joint:
    name: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float
    velocity: float
    movable: bool


def numbers(text, default):
    return [float(s) for s in (text if text is not None else default).split()]


class Chain:
    def __init__(self, joints, base, tip, tcp):
        self.joints = joints; self.base = base; self.tip = tip
        self.tcp = check_transform(tcp)
        self.active = [j for j in joints if j.movable]
        if not self.active:
            raise ValueError('no movable joints')
        self.names = [j.name for j in self.active]
        self.lower = np.array([j.lower for j in self.active]); self.upper = np.array([j.upper for j in self.active])
        self.velocity = np.array([min(.8, j.velocity) for j in self.active])

    @classmethod
    def from_urdf(cls, xml, base, tip, tcp):
        if '<!DOCTYPE' in xml.upper() or '<!ENTITY' in xml.upper() or len(xml) > 5000000:
            raise ValueError('unsupported XML')
        root = ET.fromstring(xml); by_child = {}; names = set()
        for j in root.findall('joint'):
            child, parent = j.find('child'), j.find('parent')
            if child is None or parent is None:
                raise ValueError('joint missing parent/child')
            key = child.attrib['link']; name = j.attrib['name']
            if key in by_child or name in names:
                raise ValueError('duplicate joint/child')
            names.add(name); by_child[key] = (j, parent.attrib['link'])
        chain = []; link = tip; visited = set()
        while link != base:
            if link in visited or link not in by_child:
                raise ValueError('cycle or disconnected tip')
            visited.add(link); j, parent = by_child[link]
            kind = j.attrib['type']
            if kind not in ('fixed', 'revolute') or j.find('mimic') is not None:
                raise ValueError('only fixed and bounded non-mimic revolute joints supported')
            o = j.find('origin'); a = j.find('axis'); lim = j.find('limit')
            origin = pose(numbers(o.get('xyz') if o is not None else None, '0 0 0'), numbers(o.get('rpy') if o is not None else None, '0 0 0'))
            axis = finite(numbers(a.get('xyz') if a is not None else None, '1 0 0'), (3,))
            if np.linalg.norm(axis) < 1e-12:
                raise ValueError('zero joint axis')
            lo = hi = vel = 0.
            if kind == 'revolute':
                if lim is None:
                    raise ValueError('joint limits mandatory')
                lo, hi, vel = float(lim.get('lower')), float(lim.get('upper')), float(lim.get('velocity'))
                if not np.isfinite([lo, hi, vel]).all() or lo >= hi or vel <= 0:
                    raise ValueError('invalid joint limits')
            chain.append(Joint(j.attrib['name'], link, origin, axis/np.linalg.norm(axis), lo, hi, vel, kind != 'fixed'))
            link = parent
        return cls(list(reversed(chain)), base, tip, tcp)

    def lock_joints(self,positions):
        """Explicit calibrated locks, never inferred zero angles.

        Caller MUST monitor these measured joints and abort on drift. This folds
        their actual reviewed posture into FK, without commanding the joint.
        """
        active={j.name:j for j in self.active}
        if set(positions)-set(active):
            raise ValueError('unknown/nonmovable locked joint')
        joints=[]
        for j in self.joints:
            if j.name in positions:
                value=float(positions[j.name])
                if not np.isfinite(value) or not j.lower<=value<=j.upper:
                    raise ValueError('locked joint outside limits')
                M=np.eye(4);M[:3,:3]=rotation(j.axis,value)
                j=replace(j,origin=j.origin@M,lower=0.,upper=0.,velocity=0.,movable=False)
            joints.append(j)
        return Chain(joints,self.base,self.tip,self.tcp)

    def validate_q(self, q):
        q = finite(q, (len(self.active),))
        if np.any(q < self.lower-1e-9) or np.any(q > self.upper+1e-9):
            raise ValueError('joint limit violation')
        return q

    def frames(self, q):
        q = self.validate_q(q); frames = {self.base: np.eye(4)}; T = np.eye(4); i = 0
        for j in self.joints:
            T = T@j.origin
            if j.movable:
                M = np.eye(4); M[:3, :3] = rotation(j.axis, q[i]); T = T@M; i += 1
            frames[j.child] = T.copy()
        return frames

    def fk(self, q):
        return self.frames(q)[self.tip]@self.tcp

    def jacobian(self, q):
        """World-frame geometric Jacobian [TCP linear m/rad; angular rad/rad]."""
        q=self.validate_q(q); T=np.eye(4); origins=[]; axes=[]; i=0
        for joint in self.joints:
            T=T@joint.origin
            if joint.movable:
                origins.append(T[:3,3].copy()); axes.append(T[:3,:3]@joint.axis)
                M=np.eye(4); M[:3,:3]=rotation(joint.axis,q[i]); T=T@M; i+=1
        point=(T@self.tcp)[:3,3]
        return np.array([np.r_[np.cross(axis,point-origin),axis] for axis,origin in zip(axes,origins)]).T

    def ik(self, target, seed, max_iter=200, position_tol=.002, angle_tol=.015,
           length_scale=.25, trust_radius=.15, q_pref=None, w_null=0.):
        """Local, bounded scaled DLS; no global reachability/completeness claim.

        The orientation error and angular Jacobian both use world coordinates.
        Backtracking checks the actual SO(3) residual, not a linearized success.
        """
        target=check_transform(target); q=self.validate_q(seed).copy()
        # Secondary objective (redundant 7-DOF arm): pull toward a preferred posture inside the task null space.
        # It cannot change the TCP residual to first order, so convergence/accuracy of the primary task is kept.
        if q_pref is not None:
            q_pref=self.validate_q(np.clip(finite(q_pref,(len(self.active),)),self.lower,self.upper))
        if not np.isfinite(w_null) or w_null<0 or w_null>1:
            raise ValueError('null-space weight must be in [0,1]')
        if type(max_iter) is not int or max_iter<=0 or not np.isfinite([position_tol,angle_tol,length_scale,trust_radius]).all() or min(position_tol,angle_tol,length_scale,trust_radius)<=0:
            raise ValueError('invalid IK parameters')
        weights=np.r_[np.ones(3),np.full(3,length_scale)]
        def residual(v):
            T=self.fk(v)
            return np.r_[target[:3,3]-T[:3,3],rotation_error(target[:3,:3],T[:3,:3])]
        for _ in range(max_iter):
            e=residual(q)
            if np.linalg.norm(e[:3])<position_tol and np.linalg.norm(e[3:])<angle_tol:
                return q
            J=weights[:,None]*self.jacobian(q); ew=weights*e
            U,s,Vt=np.linalg.svd(J,full_matrices=False)
            damping=.002+.05*max(0.,1.-float(s[-1])/.04)**2
            dq_task=Vt.T@((s/(s*s+damping*damping))*(U.T@ew))
            steps=[dq_task]
            if q_pref is not None and w_null>0:
                rank=int(np.sum(s>1e-6*s[0]))
                N=np.eye(len(q))-Vt[:rank].T@Vt[:rank]        # projector onto the null space of the weighted Jacobian
                steps.insert(0,dq_task+w_null*(N@(q_pref-q)))  # try task+posture first, pure task step as fallback
            improved=False
            for dq in steps:
                norm=np.linalg.norm(dq)
                if norm>trust_radius: dq=dq*(trust_radius/norm)
                for scale in (1.,.5,.25,.125,.0625,.015625):
                    candidate=np.clip(q+scale*dq,self.lower,self.upper)
                    if np.linalg.norm(weights*residual(candidate))<np.linalg.norm(ew)-1e-12:
                        q=candidate; improved=True; break
                if improved: break
            if not improved: break
        raise ValueError('IK did not converge within limits; no command permitted')
