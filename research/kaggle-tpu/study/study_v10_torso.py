"""Reach study (study_v10 = node logic incl. open-hand clearance) with the torso spheres shrunk by V10_TORSO_SHRINK m
(equivalent to a torso-only margin of 0.015 - shrink). Offline A/B only."""
import os,sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import study_v10 as V, study_core as S
SH=float(os.environ.get('V10_TORSO_SHRINK','0'))
_w=S.winit
def winit(src):
    sys.path.insert(0,src);sys.path.insert(0,src+'/test')
    import test_node_stub  # noqa
    from team048_solution import arm as ARM
    f=ARM.torso_spheres
    def ts(side,_f=f):return [ARM.Sphere(s.link,s.center,max(1e-3,s.radius-SH)) for s in _f(side)]
    ARM.torso_spheres=ts;ARM._ENV_CACHE.clear()
    _w(src)
S.winit=winit
if __name__=='__main__':S.run()
