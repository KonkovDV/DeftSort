"""Conservative enclosing-sphere collision checks, including swept-path bound.

Requires the complete robot and hand envelope (audited_complete=True) and a non-empty base-frame environment.
A partial chain model alone is not a whole-robot collision model. The held object is not part of the envelope.
"""
from dataclasses import dataclass
import numpy as np
from .geometry import finite
from . import _accel


@dataclass(frozen=True)
class Sphere:
    """Collision sphere: link name, centre in the link frame (m), radius (m)."""
    link: str
    center: np.ndarray
    radius: float

    def __post_init__(self):
        center=finite(self.center,(3,)).copy(); center.setflags(write=False)
        object.__setattr__(self,"center",center)
        if not np.isfinite(self.radius) or self.radius <= 0:
            raise ValueError('invalid radius')


def spheres_bulk(link, centers, radii):
    """Spheres for many centres at once: same objects as [Sphere(link,c,r) ...] (frozen, read-only float centre),
    validated in bulk (finite centres, finite positive radii) instead of per instance."""
    C=np.array(centers,dtype=float).reshape(-1,3); R=np.broadcast_to(np.asarray(radii,dtype=float),(len(C),))
    if not np.isfinite(C).all() or not np.isfinite(R).all() or np.any(R<=0): raise ValueError('invalid sphere batch')
    C.setflags(write=False); out=[]; new=object.__new__; put=object.__setattr__
    for c,r in zip(C,R.tolist()):
        s=new(Sphere); put(s,'link',link); put(s,'center',c); put(s,'radius',r); out.append(s)
    return out


ESCAPE_TOL=.001   # m, see CollisionModel.certify_escape
class CollisionModel:
    """Sphere model of an arm chain against a base-frame environment, with a safety margin (m)."""
    def __init__(self, chain, spheres, environment, ignored_pairs=(), *, audited_complete=False, margin=.015):
        if not audited_complete or not spheres or not environment:
            raise ValueError('complete audited robot and hand envelope and a non-empty environment are mandatory')
        if not np.isfinite(margin) or margin <= 0:
            raise ValueError('positive safety margin required')
        self._fast = None
        self.chain = chain; self.spheres = tuple(Sphere(s.link,s.center,s.radius) for s in spheres); self.environment = list(environment)
        self.ignored = {frozenset(pair) for pair in ignored_pairs}; self._margin = margin
        links = set(chain.frames(np.clip(np.zeros(len(chain.active)), chain.lower, chain.upper)))
        if any(s.link not in links for s in spheres):
            raise ValueError('unknown collision link')
        # Per-sphere/per-upstream-joint global center displacement bound.
        # Origin translations before a joint do not contribute its lever arm.
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
        # NumPy fast-path tables for valid_py (same per-pair expressions as the reference loop)
        self._sph_links=[s.link for s in self.spheres]
        self._sph_centers=np.array([s.center for s in self.spheres],dtype=float).reshape(-1,3)
        self._sph_radii=np.array([s.radius for s in self.spheres],dtype=float)
        S=self.spheres
        P=[(i,k) for i in range(len(S)) for k in range(i+1,len(S)) if S[i].link!=S[k].link and frozenset((S[i].link,S[k].link)) not in self.ignored]
        self._pi=np.array([p[0] for p in P],dtype=int); self._pk=np.array([p[1] for p in P],dtype=int)
        self._fast=_accel.fast_collision(self)   # Rust kernels (None -> NumPy reference)

    @property
    def margin(self):
        """Safety margin (m) added to every sphere-sphere distance check."""
        return self._margin

    @margin.setter
    def margin(self,value):
        """Safety margin (m) added to every sphere-sphere distance check."""
        if not np.isfinite(value) or value<=0: raise ValueError('positive safety margin required')
        self._margin=value
        if self._fast is not None: self._fast.margin=float(value)

    def _fallback(self,where,exc):
        _accel.guard(exc); _accel.failed(where,exc); self._fast=None

    @property
    def environment(self):
        """Environment spheres in the chain base frame (pelvis)."""
        return self._environment

    @environment.setter
    def environment(self,values):
        """Environment spheres in the chain base frame (pelvis)."""
        # Sphere is frozen with a private read-only centre copy: reuse instances, re-validate anything else
        self._environment=tuple(s if type(s) is Sphere else Sphere(s.link,s.center,s.radius) for s in values)
        values=self._environment
        if not values or any(s.link!=self.chain.base for s in values): raise ValueError("nonempty base-frame environment required")
        self._env_centers=np.array([s.center for s in values],dtype=float).reshape(-1,3)
        self._env_radii=np.array([s.radius for s in values],dtype=float)
        self._env_centers.setflags(write=False); self._env_radii.setflags(write=False)
        if getattr(self,'_fast',None) is not None:
            try: self._fast.set_env(np.ascontiguousarray(self._env_centers),[float(r) for r in self._env_radii])
            except BaseException as e: self._fallback('set_env',e)

    def install_environment(self, values, centers, radii, fast_env=None):
        """Install a pre-validated environment: `values` a tuple of base-frame Spheres, `centers`/`radii` the read-only
        arrays the `environment` setter would build from them (same rows, same order), `fast_env` an optional cached
        Rust grid of exactly those arrays. Equivalent to `self.environment = values` without re-validation/rebuild."""
        self._environment=values; self._env_centers=centers; self._env_radii=radii
        if getattr(self,'_fast',None) is not None:
            try:
                if fast_env is not None: self._fast.use_env(fast_env)
                else: self._fast.set_env(np.ascontiguousarray(centers),[float(r) for r in radii])
            except BaseException as e: self._fallback('set_env',e)

    def valid(self, q, inflation=0):
        """True when posture q (rad) keeps every checked pair apart by more than the margin (+ inflation, m)."""
        inflation=np.broadcast_to(finite(inflation),(len(self.spheres),))
        if np.any(inflation<0): raise ValueError('invalid collision inflation')
        if self._fast is not None:
            q=self.chain.validate_q(q)
            try: return self._fast.valid(np.ascontiguousarray(q,float),np.ascontiguousarray(inflation,float))
            except BaseException as e: self._fallback('valid',e)
        return self.valid_vec(q,inflation)

    def valid_vec(self, q, inflation=0):
        """Vectorised NumPy fallback: same per-pair expressions as valid_py, evaluated as arrays."""
        inflation=np.broadcast_to(finite(inflation),(len(self.spheres),))
        if np.any(inflation<0): raise ValueError('invalid collision inflation')
        frames=self.chain.frames_py(q)
        R=np.array([frames[l][:3,:3] for l in self._sph_links]); t=np.array([frames[l][:3,3] for l in self._sph_links])
        W=np.einsum('kij,kj->ki',R,self._sph_centers)+t
        D=np.sqrt(((self._env_centers[None,:,:]-W[:,None,:])**2).sum(axis=2))
        if np.any(D<=((self._sph_radii[:,None]+self._env_radii[None,:])+self.margin)+inflation[:,None]): return False
        if len(self._pi):
            d=np.sqrt(((W[self._pi]-W[self._pk])**2).sum(axis=1))
            if np.any(d<=(((self._sph_radii[self._pi]+self._sph_radii[self._pk])+self.margin)+inflation[self._pi])+inflation[self._pk]): return False
        return True

    def valid_py(self, q, inflation=0):
        """NumPy reference of valid()."""
        inflation=np.broadcast_to(finite(inflation),(len(self.spheres),))
        if np.any(inflation<0): raise ValueError('invalid collision inflation')
        frames = self.chain.frames_py(q); world = []
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
        """True when the straight joint-space path a -> b (rad) is proven collision-free."""
        if not np.isfinite(max_step) or max_step <= 0:
            raise ValueError('positive finite collision resolution required')
        a = self.chain.validate_q(a); b = self.chain.validate_q(b)
        if self._fast is not None:
            try: return self._fast.certify(np.ascontiguousarray(a,float),np.ascontiguousarray(b,float),float(max_step))
            except BaseException as e: self._fallback('certify',e)
        steps = max(1, int(np.ceil(np.max(abs(b-a))/max_step)))
        # Every intermediate center is within bound of a sampled center.
        inflation = self.levers@abs(b-a)/steps
        return all(self.valid_vec(a+(b-a)*(i/steps), inflation) for i in range(steps+1))

    def gaps(self, q):
        """Signed clearances at q (NumPy, same expressions as valid_vec with zero inflation): (env (n_spheres, n_env),
        self (n_pairs,)); a pair is in collision (within the margin) when its gap <= 0."""
        frames=self.chain.frames_py(self.chain.validate_q(q))
        R=np.array([frames[l][:3,:3] for l in self._sph_links]); t=np.array([frames[l][:3,3] for l in self._sph_links])
        W=np.einsum('kij,kj->ki',R,self._sph_centers)+t
        D=np.sqrt(((self._env_centers[None,:,:]-W[:,None,:])**2).sum(axis=2))
        ge=D-((self._sph_radii[:,None]+self._env_radii[None,:])+self.margin)
        if len(self._pi):
            d=np.sqrt(((W[self._pi]-W[self._pk])**2).sum(axis=1))
            gs=d-((self._sph_radii[self._pi]+self._sph_radii[self._pk])+self.margin)
        else: gs=np.zeros(0)
        return ge,gs

    def certify_escape(self, a, b, max_step=.003):
        """Start-state-collision handling (the MoveIt 'FixStartStateCollision' situation): the start `a` may lie
        within the safety margin of some pairs (v9a: the organiser start posture has the forearm 3 mm from the hip
        mesh, inside the 1.5 cm margin). The segment a->b is accepted iff
          * every pair that is clear at `a` stays clear along the whole swept segment (same inflation bound as certify),
          * every pair that is not clear at `a` never gets more than ESCAPE_TOL closer than it was at the start
            (no deeper penetration than the start posture itself; 1 mm < the 2.9 mm real mesh clearance of the
            start posture), and
          * `b` is fully valid.
        If `a` is valid this is exactly certify(a, b). Environment spheres within one sample of the start are removed
        so certify's inflated first sample can pass, and checked here: a pair already inside the margin may close by
        at most ESCAPE_TOL; a pair that is still clear may not cross into the margin anywhere on the segment
        (endpoint gaps plus the per-step travel bound, subdivided when that bound does not prove it)."""
        a=self.chain.validate_q(a); b=self.chain.validate_q(b)
        if self.valid(a): return self.certify(a,b,max_step)
        if not self.valid(b): return False
        ge0,gs0=self.gaps(a)
        if len(gs0) and np.any(gs0<=0): return self._certify_escape_full(a,b,max_step,ge0,gs0)
        steps=max(1,int(np.ceil(np.max(abs(b-a))/max_step))); inf=self.levers@abs(b-a)/steps
        # Within one sample of the start the inflated certify cannot accept its first sample. Those spheres come
        # out of certify and are checked below. Spheres already clear and further than that stay in certify.
        J=np.unique(np.nonzero(ge0<=inf[:,None])[1]); keep=np.ones(len(self._environment),bool); keep[J]=False
        saved=(self._environment,self._env_centers,self._env_radii)
        try:
            if keep.any():
                self.environment=[e for e,k in zip(self._environment,keep) if k]
                if not self.certify(a,b,max_step): return False
            elif not self._certify_self(a,b,max_step): return False   # no env left in certify: self-pairs still count
        finally:
            self.install_environment(*saved)
        if not len(J): return True
        g0=ge0[:,J]; pen=g0<=0; near=(g0>0)&(g0<=inf[:,None]); far=g0>inf[:,None]; prev=g0
        for i in range(1,steps+1):
            g=self._env_gap(a+(b-a)*(i/steps),J)
            if np.any((g<=inf[:,None])&far): return False
            if np.any(g[pen]<g0[pen]-ESCAPE_TOL): return False
            if np.any(near&(g<=0)): return False
            unsure=near&(prev+g<=inf[:,None])
            if np.any(unsure) and not self._near_stays_clear(a,b,steps,(i-1)/steps,i/steps,inf,J,unsure,prev,g): return False
            left=(pen|near)&(g>inf[:,None]); far=far|left; pen=pen&~left; near=near&~left; prev=g
        return True

    def _env_gap(self, q, J):
        """Signed env clearances (n_spheres, len(J)) at q, same expression as gaps()."""
        frames=self.chain.frames_py(q)
        R=np.array([frames[l][:3,:3] for l in self._sph_links]); t=np.array([frames[l][:3,3] for l in self._sph_links])
        W=np.einsum('kij,kj->ki',R,self._sph_centers)+t
        return np.sqrt(((self._env_centers[J][None,:,:]-W[:,None,:])**2).sum(axis=2))-((self._sph_radii[:,None]+self._env_radii[J][None,:])+self.margin)

    def _near_stays_clear(self, a, b, steps, u0, u1, inf, J, mask, g0, g1, depth=0):
        """Pairs in `mask` (clear, but within a sample of the margin) stay strictly clear on the joint segment
        u in [u0, u1]. A sphere moves at most inf*(u1-u0)*steps on that segment; endpoint gaps that sum to more
        than this cannot have crossed 0. Otherwise the segment is halved, at most eight times."""
        if not np.any(mask): return True
        um=.5*(u0+u1); gm=self._env_gap(a+(b-a)*um,J)
        if np.any(gm[mask]<=0): return False
        motion=inf[:,None]*((u1-u0)*steps*.5); left=mask&(g0+gm<=motion); right=mask&(gm+g1<=motion)
        if not np.any(left|right): return True
        if depth>=8: return False
        return (self._near_stays_clear(a,b,steps,u0,um,inf,J,left,g0,gm,depth+1)
                and self._near_stays_clear(a,b,steps,um,u1,inf,J,right,gm,g1,depth+1))

    def _certify_self(self, a, b, max_step):
        """Self-pairs along a->b at certify's samples and inflation, when every environment sphere was removed."""
        if not len(self._pi): return True
        steps=max(1,int(np.ceil(np.max(np.abs(b-a))/max_step))); inf=self.levers@np.abs(b-a)/steps
        thr=(self._sph_radii[self._pi]+self._sph_radii[self._pk])+self.margin+inf[self._pi]+inf[self._pk]
        for i in range(steps+1):
            frames=self.chain.frames_py(a+(b-a)*(i/steps))
            R=np.array([frames[l][:3,:3] for l in self._sph_links]); t=np.array([frames[l][:3,3] for l in self._sph_links])
            W=np.einsum('kij,kj->ki',R,self._sph_centers)+t
            if np.any(np.sqrt(((W[self._pi]-W[self._pk])**2).sum(axis=1))<=thr): return False
        return True

    def _certify_escape_full(self, a, b, max_step, ge0, gs0):
        """certify_escape reference for starts with self-pair contacts too (all pairs in NumPy)."""
        steps=max(1,int(np.ceil(np.max(abs(b-a))/max_step)))
        inf=self.levers@abs(b-a)/steps
        be=ge0<=0; bs=gs0<=0
        pe,ps=ge0-ESCAPE_TOL,gs0-ESCAPE_TOL
        ie=inf[:,None]; is_=(inf[self._pi]+inf[self._pk]) if len(self._pi) else np.zeros(0)
        for i in range(1,steps+1):
            ge,gs=self.gaps(a+(b-a)*(i/steps))
            if np.any((ge<=ie)&~be) or (len(gs) and np.any((gs<=is_)&~bs)): return False
            if np.any(ge[be]<pe[be]) or (len(gs) and np.any(gs[bs]<ps[bs])): return False
            be=be&(ge<=ie); bs=bs&(gs<=is_) if len(gs) else bs
        return True

    def certify_py(self, a, b, max_step=.003):
        """NumPy reference of certify() (samples valid_py)."""
        a = self.chain.validate_q(a); b = self.chain.validate_q(b)
        steps = max(1, int(np.ceil(np.max(abs(b-a))/max_step)))
        inflation = self.levers@abs(b-a)/steps
        return all(self.valid_py(a+(b-a)*(i/steps), inflation) for i in range(steps+1))

    def certify_box(self, a, b, padding=0.):
        """Conditional sphere-envelope bound for any joint ordering in box.

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
    """Time-parametrised joint path: points (rad) and times (s)."""
    points: np.ndarray
    times: np.ndarray


def joint_path(chain, collision, waypoints, speed=.35, acceleration=.4, dt=.05):
    """Joint trajectory through certified waypoints, cubic smoothstep per segment (speed rad/s, acceleration rad/s^2, dt s)."""
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

