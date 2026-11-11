"""Rim pinch gated by the dish's gravity torque, and rim-held dishes placed by their centre.
Fixtures: v12_rim_dishes.npz/.json (v7 sort3 s102 start head frame: Utana, Ecoforms, Threshold segments; v11b
trace 0098.15: Utana before the attempt that slipped during the carry, at_grasp joint angles from the log)."""
import os,sys,json,unittest
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__));sys.path.insert(0,os.path.join(HERE,'..'));sys.path.insert(0,HERE)
import test_node_stub as TS  # noqa: F401  (installs the ROS stubs)
from team048_solution import arm as ARM, grasp as GR, node as N, trays as TR
import scene_fixture as SF
FX=os.path.join(HERE,'fixtures');Z=np.load(os.path.join(FX,'v12_rim_dishes.npz'));META=json.load(open(os.path.join(FX,'v12_rim_dishes.json')))
def seg(k):return Z[k+'_cloud'].astype(float),(Z[k+'_n'],float(Z[k+'_d']))
def mass(k):return N.PRI[META[k]['obj']]['mass']

class RimTorque(unittest.TestCase):
    def test_utana_gets_no_rim_candidate(self):
        for k in ('utana_v7s102','utana_v11b'):
            cl,pl=seg(k);f,C=GR.candidates(cl,pl,'dish',mass=mass(k))
            self.assertTrue(f['rim_heavy'],k);self.assertEqual(C,[],k)
            self.assertGreater(f['rim_torque'],.2);self.assertLess(f['rim_torque'],.23)
            f0,C0=GR.candidates(cl,pl,'dish');self.assertEqual(sum(g['kind']=='rim' for g in C0),7,k)   # without mass: no torque gate
    def test_light_dishes_unchanged(self):
        for k in ('ecoforms_v7s102','threshold_v7s102'):
            cl,pl=seg(k);f,C=GR.candidates(cl,pl,'dish',mass=mass(k));f0,C0=GR.candidates(cl,pl,'dish')
            self.assertFalse(f['rim_heavy'],k);self.assertLess(f['rim_torque'],GR.RIM_TORQUE_MAX)
            self.assertEqual([np.round(g['center'],6).tolist() for g in C],[np.round(g['center'],6).tolist() for g in C0])
    def test_box_candidates_ignore_mass(self):
        cl,pl=seg('ecoforms_v7s102');_,C=GR.candidates(cl,pl,'box',mass=9.);_,C0=GR.candidates(cl,pl,'box')
        self.assertEqual(len(C),len(C0));self.assertTrue(all('obj_center' not in g for g in C))
    def test_node_reason(self):
        import inspect;src=inspect.getsource(N.Sol.pick_place)
        self.assertIn("mass=PRI.get(name,{}).get('mass')",src);self.assertIn('rim_heavy',src)

class RimPlace(unittest.TestCase):
    @classmethod
    def setUpClass(c):
        c.A=ARM.Arm('l');c.A.set_scene(SF.plane(),list(SF.TRAYS.values()))
        r0=.05;c0=np.array([.36,.25]);tw=-c0/np.linalg.norm(c0);p=c0+r0*tw   # Ecoforms-size dish, near-robot rim point
        c.g=dict(kind='rim',center=p,close_axis=tw,width=GR.RIM_W,z=.057,obj_center=c0);c.r0=r0
        c.c3=np.array([p[0],p[1],SF.TABLE_TOP+c.g['z']])
        c.pk=c.A.plan_pick(c.c3,np.r_[tw,0.],np.array([0,0,-1.]),GR.RIM_W,point='pinch')
    def test_offset_recovers_dish_centre_at_pick(self):
        o=N.held_offset(self.A,self.pk['q_grasp'],self.g,self.c3);T=self.A.tcp(self.pk['q_grasp'])
        self.assertLess(np.linalg.norm((T[:3,:3]@o+T[:3,3])[:2]-self.g['obj_center']),1e-9)
        self.assertIsNone(N.held_offset(self.A,self.pk['q_grasp'],dict(kind='side'),self.c3))
    def test_fk_offset_on_v11b_utana(self):
        m=META['utana_v11b'];A=ARM.Arm(m['side']);q=np.clip(np.array(m['at_grasp_q']),A.chain.lower,A.chain.upper)
        T=A.tcp(q);Am,gt=A._hand_frame('pinch');pin=T[:3,:3]@gt+T[:3,3];u=T[:3,:3]@Am[:,0]
        self.assertLess(np.linalg.norm(pin[:2]+m['r0']*u[:2]-np.array(m['c0'])),.015)   # 11.1 mm: inside TCP_TOL + fit
    def margins(s,obj_off):
        tx,ty,ax,ay,ts,tl,th=ARM.tray_geom(SF.TRAYS['tray_left']);o=N.held_offset(s.A,s.pk['q_grasp'],s.g,s.c3);out=[]
        for k in range(3):
            dl=TR.slot_offset(k,s.r0,size=(ts,tl),ty=ty)
            yp=N.plan_place_yaw(s.A,s.pk['q_lift'],tx+dl*ax,ty+dl*ay,SF.plane(),list(SF.TRAYS.values()),
                                fallback=(tx,ty) if dl else None,tray_h=th,obj_off=o if obj_off else None)
            s.assertIsNotNone(yp);yaw,(qp,_,_)=yp;s.assertEqual(yaw,0.)
            T=s.A.tcp(qp);d=(T[:3,:3]@o+T[:3,3])[:2]-np.array([tx,ty])
            out.append(min(tl/2-TR.TRAY_WALL-abs(d@[ax,ay]),ts/2-TR.TRAY_WALL-abs(d@[-ay,ax]))-s.r0)
        return out
    def test_dish_stays_inside_the_tray(self):
        old=self.margins(False);new=self.margins(True)
        self.assertLess(min(old),0.)            # slot 1: the hanging dish crosses the short wall (-1.5 cm)
        self.assertGreater(min(new),.02)        # every slot: the dish rim at least 2 cm inside the walls
    def test_side_place_unchanged(self):
        q=self.pk['q_lift'];tx,ty=.34,.69
        a=N.plan_place(self.A,q,tx,ty,SF.TABLE_TOP);b=N.plan_place(self.A,q,tx,ty,SF.TABLE_TOP,obj_off=None)
        self.assertTrue(all(np.array_equal(x,y) for x,y in zip(a,b) if x is not None))

if __name__=='__main__':unittest.main()
