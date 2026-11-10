"""Conservative enclosing-sphere collision checks, including swept-path bound.

Requires an audited complete envelope (robot + hand + payload) and environment.
A partial chain model alone is not a whole-robot collision model.
"""
from dataclasses import dataclass
import numpy as np
from .geometry import finite


@dataclass(frozen=True)
class Sphere:
    link: str
    center: np.ndarray
    radius: float

    def __post_init__(self):
        center=finite(self.center,(3,)).copy(); center.setflags(write=False)
        object.__setattr__(self,"center",center)
        if not np.isfinite(self.radius) or self.radius <= 0:
            raise ValueError('invalid radius')


class CollisionModel:
    def __init__(self, chain, spheres, environment, ignored_pairs=(), *, audited_complete=False, margin=.015):
        if not audited_complete or not spheres or not environment:
            raise ValueError('complete audited robot/hand/payload and environment envelopes mandatory')
        if not np.isfinite(margin) or margin <= 0:
            raise ValueError('positive safety margin required')
        self.chain = chain; self.spheres = tuple(Sphere(s.link,s.center,s.radius) for s in spheres); self.environment = list(environment)
        self.ignored = {frozenset(pair) for pair in ignored_pairs}; self.margin = margin
        links = set(chain.frames(np.clip(np.zeros(len(chain.active)), chain.lower, chain.upper)))
        if any(s.link not in links for s in spheres):
            raise ValueError('unknown collision link')
        # Per-sphere/per-upstream-joint global center displacement bound.
        # Origin translations BEFORE a joint do not contribute its lever arm.
        self.levers=np.zeros((len(self.spheres),len(chain.active)))
        for row,sphere in enumerate(self.spheres):
            stop=next((i for i,j in enumerate(chain.joints) if j.child==sphere.link),-1)
            active_index=0
            for index,joint in enumerate(chain.joints):
                if joint.movable:
                    if index<=stop:
                        self.levers[row,active_index]=sum(np.linalg.norm(j.origin[:3,3]) for j in chain.joints[index+1:stop+1])+np.linalg.norm(sphere.center)
                    active_index+=1
        self.reach=float(np.max(self.levers))

    @property
    def environment(self):
        return self._environment

    @environment.setter
    def environment(self,values):
        self._environment=tuple(Sphere(s.link,s.center,s.radius) for s in values)
        values=self._environment
        if not values or any(s.link!=self.chain.base for s in values): raise ValueError("nonempty base-frame environment required")
        self._env_centers=np.array([s.center for s in values],dtype=float).reshape(-1,3)
        self._env_radii=np.array([s.radius for s in values],dtype=float)
        self._env_centers.setflags(write=False); self._env_radii.setflags(write=False)

    def valid(self, q, inflation=0):
        inflation=np.broadcast_to(finite(inflation),(len(self.spheres),))
        if np.any(inflation<0): raise ValueError('invalid collision inflation')
        frames = self.chain.frames(q); world = []
        for s in self.spheres:
            T = frames[s.link]; world.append((s, T[:3, :3]@s.center+T[:3, 3]))
        for k,(s, p) in enumerate(world):
            # Vectorized immutable snapshot of world/pelvis obstacle spheres.
            if np.any(np.linalg.norm(self._env_centers-p,axis=1) <= s.radius+self._env_radii+self.margin+inflation[k]):
                return False
        for i, (s, p) in enumerate(world):
            for k,(t, r) in enumerate(world[i+1:],start=i+1):
                if s.link == t.link or frozenset((s.link, t.link)) in self.ignored:
                    continue
                if np.linalg.norm(p-r) <= s.radius+t.radius+self.margin+inflation[i]+inflation[k]:
                    return False
        return True

    def certify(self, a, b, max_step=.003):
        if not np.isfinite(max_step) or max_step <= 0:
            raise ValueError('positive finite collision resolution required')
        a = self.chain.validate_q(a); b = self.chain.validate_q(b)
        steps = max(1, int(np.ceil(np.max(abs(b-a))/max_step)))
        # Every intermediate center is within bound of a sampled center.
        inflation = self.levers@abs(b-a)/steps
        return all(self.valid(a+(b-a)*(i/steps), inflation) for i in range(steps+1))

    def certify_box(self, a, b, padding=0.):
        """Conditional sphere-envelope bound for ANY joint ordering in box.

        Padding is a calibrated per-joint tracking/latency/encoder bound in rad.
        This is not mesh CCD or a guarantee about unmodeled robot geometry.
        """
        a=self.chain.validate_q(a); b=self.chain.validate_q(b)
        pad=np.broadcast_to(finite(padding),a.shape)
        if np.any(pad<0): raise ValueError('negative tracking tube')
        low=np.minimum(a,b)-pad; high=np.maximum(a,b)+pad
        if np.any(low<self.chain.lower) or np.any(high>self.chain.upper): return False
        return self.valid((low+high)/2,self.levers@((high-low)/2))


@dataclass
class Trajectory:
    points: np.ndarray
    times: np.ndarray


def joint_path(chain, collision, waypoints, speed=.35, acceleration=.4, dt=.05):
    if not np.isfinite([speed, acceleration, dt]).all() or not 0 < speed <= .8 or acceleration <= 0 or dt <= 0:
        raise ValueError('invalid trajectory bounds')
    qs = [chain.validate_q(q) for q in waypoints]
    if len(qs) < 2:
        raise ValueError('need at least two waypoints')
    points = [qs[0]]; times = [0.]
    vmax = np.minimum(chain.velocity, speed)
    for a, b in zip(qs, qs[1:]):
        if not collision.certify(a, b):
            raise ValueError('uncertified swept path')
        delta = abs(b-a)
        # Cubic smoothstep: peak speed 1.5*d/T, peak acceleration 6*d/T^2.
        duration = max(dt, float(np.max(1.5*delta/vmax)), float(np.sqrt(np.max(6*delta/acceleration))))
        count = max(1, int(np.ceil(duration/dt))); start = times[-1]
        for i in range(1, count+1):
            u = i/count; points.append(a+(b-a)*(3*u*u-2*u*u*u)); times.append(start+duration*u)
    return Trajectory(np.array(points), np.array(times))


def cartesian_waypoints(chain, targets, seed, collision):
    qs = [chain.validate_q(seed)]
    for target in targets:
        q = chain.ik(target, qs[-1])
        if not collision.certify(qs[-1], q):
            raise ValueError('IK endpoint reachable but connecting path unsafe')
        qs.append(q)
    return qs
