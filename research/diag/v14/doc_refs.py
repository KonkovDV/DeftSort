"""file:line references in docs/MATH_PHYSICS.md and docs/ORG_ANSWERS.md, each tied to the code it names (anchor =
regex of the referenced line). --check: every reference must point at its anchor; --fix: rewrite the numbers.
Usage: python doc_refs.py [--fix]"""
import re,sys,os
R=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','..')
PKG='src/team048_solution/team048_solution/';T='src/team048_solution/test/'
def line(f,rx,nth=0,after=0):
    L=open(os.path.join(R,f)).read().split('\n');hits=[i+1 for i,l in enumerate(L) if re.search(rx,l) and i+1>after]
    if len(hits)<=nth:raise SystemExit(f'anchor not found: {f} {rx}')
    return hits[nth]
# (doc, text before the number in the doc, file, anchor regex, [end anchor for a range])
M=[('MATH_PHYSICS','node.py:',PKG+'node.py',r'def tf\(','tf'),('MATH_PHYSICS','node.py:',PKG+'node.py',r'^def quat_to_T','quat'),
 ('MATH_PHYSICS','arm.py:',PKG+'arm.py',r'^def rotate_trays','rotate'),('MATH_PHYSICS','node.py:',PKG+'node.py',r'P=deproject\(dmm,K,T\)','deproj'),
 ('MATH_PHYSICS','perception.py:',PKG+'perception.py',r'^def deproject_py','deproj_py'),('MATH_PHYSICS','perception.py:',PKG+'perception.py',r'^def table_plane\(','plane'),
 ('MATH_PHYSICS','perception.py:',PKG+'perception.py',r'^def _refine_plane','refine'),('MATH_PHYSICS','arm.py:',PKG+'arm.py',r'^def plane_plausible','plaus'),
 ('MATH_PHYSICS','arm.py:',PKG+'arm.py',r'^def plane_height','ph'),('MATH_PHYSICS','perception.py:',PKG+'perception.py',r'^def near_edge','edge'),
 ('MATH_PHYSICS','trays.py:',PKG+'trays.py',r'^def canonical_axis','canon'),
 ('MATH_PHYSICS','perception.py:',PKG+'perception.py',r'^def segment\(','seg'),('MATH_PHYSICS','node.py:',PKG+'node.py',r'C\[:,:len\(N\)\]=np.abs\(h-ph\)','cost'),
 ('MATH_PHYSICS','grasp.py:',PKG+'grasp.py',r'^def candidates','cand'),('MATH_PHYSICS','grasp.py:',PKG+'grasp.py',r'^def footprint','fp'),
 ('MATH_PHYSICS','grasp.py:',PKG+'grasp.py',r'^def circle_fit','circ'),('MATH_PHYSICS','grasp.py:',PKG+'grasp.py',r'^def rim_closed','rimc'),
 ('MATH_PHYSICS','grasp.py:',PKG+'grasp.py',r'^RIM_TORQUE_MAX','torque'),('MATH_PHYSICS','arm.py:',PKG+'arm.py',r'def grasp_pose','gpose'),
 ('MATH_PHYSICS','arm.py:',PKG+'arm.py',r'^def grasp_clear_shift','clear'),('MATH_PHYSICS','node.py:',PKG+'node.py',r'^def held_offset','held_off'),
 ('MATH_PHYSICS','grip.py:',PKG+'grip.py',r'^HAND_FORCE_RATIO =','force',r'^FULL_STROKE_S'),('MATH_PHYSICS','grip.py:',PKG+'grip.py',r'^def contact_target','ct'),
 ('MATH_PHYSICS','grip.py:',PKG+'grip.py',r'^def held_rule','held'),('MATH_PHYSICS','grip.py:',PKG+'grip.py',r'^def approach_deviation','guard'),
 ('MATH_PHYSICS','kinematics.py:',PKG+'kinematics.py',r'def _ik_loop','ik'),('MATH_PHYSICS','arm.py:',PKG+'arm.py',r'def correct\(','corr'),
 ('MATH_PHYSICS','arm.py:',PKG+'arm.py',r'^def arm_spheres','sph'),('MATH_PHYSICS','planning.py:',PKG+'planning.py',r'def valid_py','valid'),
 ('MATH_PHYSICS','planning.py:',PKG+'planning.py',r'def certify\(','cert'),('MATH_PHYSICS','node.py:',PKG+'node.py',r'spd=np.maximum\(vmax\*dq/m','spd'),
 ('MATH_PHYSICS','budget.py:',PKG+'budget.py',r'^def start_time','st'),('MATH_PHYSICS','budget.py:',PKG+'budget.py',r'def can_start','cs'),
 ('MATH_PHYSICS','node.py:',PKG+'node.py',r'^def lift_allowed','lift'),('MATH_PHYSICS','node.py:',PKG+'node.py',r'^def place_time','pt'),
 ('MATH_PHYSICS','node.py:',PKG+'node.py',r'^def hold_time_bound','hold'),('MATH_PHYSICS','node.py:',PKG+'node.py',r'def end_safe','end'),
 ('MATH_PHYSICS','node.py:',PKG+'node.py',r'def recover','rec'),('MATH_PHYSICS','sensing.py:',PKG+'sensing.py',r'def snapshot','snap')]
# MATH_PHYSICS: references in document order (one per occurrence); the table above lists them in that order except
# where the doc names two on one line -- resolved below by matching the order of appearance.
ORDER_MP=['tf','quat','rotate','deproj','deproj_py','plane','refine','plaus','ph','edge','canon','seg','cost','cand','fp','circ','rimc','torque',
 'gpose','clear','held_off','force','ct','held','guard','ik','corr','sph','valid','cert','spd','st','cs','lift','pt','hold','rec','end','snap']
def ref(key):
    for e in M:
        if e[4]==key:
            a=line(e[2],e[3])
            return f'{os.path.basename(e[2])}:{a}'+(f'-{line(e[2],e[5],after=a)}' if len(e)>5 else '')
    raise KeyError(key)
def mp(text):
    keys=iter(ORDER_MP);out=[];last=0
    for m in re.finditer(r'(?<![\w/])([a-z_]+\.py):(\d+)(?:-(\d+))?',text):
        k=next(keys);new=ref(k)
        if os.path.basename(new.split(':')[0])!=m.group(1):raise SystemExit(f'order mismatch at {m.group(0)} vs {k}')
        out.append(text[last:m.start()]+new);last=m.end()
    return ''.join(out)+text[last:]
O=dict(capture=(PKG+'node.py',r'def capture\(',r'return P,rgb,dmm>0'),task_sub=(PKG+'node.py',r"create_subscription\(String,'/sim/task'"),
 subs_test=(T+'test_compliance.py',r'def test_subscriptions'),stub_topics=(T+'test_node_stub.py',r'def test_subscribed_topics'),
 empty=(PKG+'node.py',r"if gr\['empty'\]:",r'return None$'),place_time=(PKG+'node.py',r'^def place_time'),no_tray=(PKG+'node.py',r'if tray not in s.trays:'),
 t_nolift=(T+'test_compliance.py',r'def test_no_lift_when_place'),t_close=(T+'test_compliance.py',r'def test_grasp_close_sequence'),
 wall=(PKG+'node.py',r'^WALL_LAUNCH='),t0=(PKG+'node.py',r't0=BUD.start_time\('),t_age=(T+'test_compliance.py',r'def test_launch_age'),
 t_simt=(T+'test_compliance.py',r'def test_run_sh_uses_sim_time'),rd_sub=(PKG+'node.py',r"create_subscription\(String,'/robot_description'"),
 use_rd=(PKG+'node.py',r'def use_robot_description'),load=(PKG+'arm.py',r'^def load_urdf'),t_rd=(T+'test_compliance.py',r'def test_robot_description_bundled'),
 K=(PKG+'node.py',r'K=np.array\(info.k\)'),torso_info=(PKG+'node.py',r"/cam_torso/depth/camera_info'"),ign=(PKG+'arm.py',r'ign=\[\(a,b\)'),
 body=(PKG+'arm.py',r'^def body_env'),other=(PKG+'arm.py',r'def set_other'),home=(PKG+'arm.py',r'def path_home'),go_home=(PKG+'node.py',r'def go_home'),
 pp_other=(PKG+'node.py',r"A=s.arms\[side\];A.set_other"),run_home=(PKG+'node.py',r"for k in 'lr':s.go_home\(k\)"),
 t_home=(T+'test_arm.py',r'def test_home_valid_and_self_collision'),t_walls=(T+'test_ideas.py',r'def test_tray_walls_are_obstacles'),
 t_done=(T+'test_node_stub.py',r'def test_run_retries_empty_observation_then_done'),run=(PKG+'node.py',r'    def run\(s\):',r"hand settings at end"),
 search=(PKG+'trays.py',r'^SEARCH'),detect=(PKG+'trays.py',r'^def detect'),survey=(PKG+'node.py',r'def survey'),edge=(PKG+'node.py',r'def measure_near_edge'),
 band=(PKG+'node.py',r'if not ARMMOD.plane_plausible'),tconst=(PKG+'arm.py',r'^TABLE_Z_BAND',r'^TABLE_X_NEAR_MIN'),targs=(PKG+'arm.py',r'^def _table_args'),
 tbuild=(PKG+'arm.py',r'^def table_env_build'),c17=(T+'test_compliance.py',r'def test_no_scene_yaml'),c26=(T+'test_compliance.py',r'def test_scene_fixture_not'),
 c31=(T+'test_compliance.py',r'def test_plane_band'),c35=(T+'test_compliance.py',r'def test_table_obstacle'),c41=(T+'test_compliance.py',r'def test_v7_near_edge'),
 c52=(T+'test_compliance.py',r'def test_v7_trays'),c68=(T+'test_compliance.py',r'def test_missing_tray'),grip=(PKG+'grip.py',r'^HAND_FORCE_RATIO =',r'^CLOSE_PAST'),
 thumb=(PKG+'node.py',r'def wait_thumb'),close=(PKG+'node.py',r'def grasp_close'),fing=(PKG+'node.py',r'def wait_fingers'),req=(PKG+'node.py',r'def hand_requests'),
 c98=(T+'test_compliance.py',r'def test_defaults'),c102=(T+'test_compliance.py',r'def test_effort_threshold'),c106=(T+'test_compliance.py',r'def test_contact_target_and'),
 c124=(T+'test_compliance.py',r'def test_thumb_first'),cmd=(PKG+'node.py',r'c.spd=float\(min\(.8,max\(v,1e-3\)\)\)'),t_spd=(T+'test_node_stub.py',r'def test_speed_cap_in_command'))
def o(k):
    e=O[k];a=line(e[0],e[1]);return str(a)+(f'-{line(e[0],e[2],after=a)}' if len(e)>2 else '')   # range end: first hit after the start
def org(text):
    S=lambda f,*ks:f+':'+','.join(o(k) for k in ks)
    rows={
     '`pkg/node.py:248-262` (`capture`':'`'+S('pkg/node.py','capture')+'` (`capture`',
     '`pkg/node.py:71` (только `/sim/task`)':'`'+S('pkg/node.py','task_sub')+'` (только `/sim/task`)',
     '`T/test_node_stub.py:66`, `T/test_compliance.py:87`':'`'+S('T/test_node_stub.py','stub_topics')+'`, `'+S('T/test_compliance.py','subs_test')+'`',
     '`pkg/node.py:508-521` (пустой захват':'`'+S('pkg/node.py','empty')+'` (пустой захват',
     '`pkg/node.py:622` (`place_time`), `pkg/node.py:408`':'`'+S('pkg/node.py','place_time')+'` (`place_time`), `'+S('pkg/node.py','no_tray')+'`',
     '`T/test_compliance.py:142` (`no_lift_when_place_does_not_fit`), `:115`':'`'+S('T/test_compliance.py','t_nolift')+'` (`no_lift_when_place_does_not_fit`), `:'+o('t_close')+'` (`grasp_close_sequence`)',
     '`pkg/node.py:14,376`':'`'+S('pkg/node.py','wall','t0')+'`',
     '`T/test_compliance.py:180,185`':'`'+S('T/test_compliance.py','t_age','t_simt')+'`',
     '`pkg/node.py:83` (подписка latched), `:423` (`use_robot_description`), `pkg/arm.py:25`':'`'+S('pkg/node.py','rd_sub')+'` (подписка latched), `:'+o('use_rd')+'` (`use_robot_description`), `'+S('pkg/arm.py','load')+'`',
     '| `T/test_compliance.py:92`. `[sim]`':'| `'+S('T/test_compliance.py','t_rd')+'`. `[sim]`',
     '`pkg/node.py:262` (K из `info.k`), торс `:80`':'`'+S('pkg/node.py','K')+'` (K из `info.k`), торс `:'+o('torso_info')+'`',
     '`pkg/arm.py:236` (внутрицепные пары), `:137` (`body_env`), `:251` (`set_other`), `:261` (`path_home`), `pkg/node.py:444,473,388`':
       '`'+S('pkg/arm.py','ign')+'` (внутрицепные пары), `:'+o('body')+'` (`body_env`), `:'+o('other')+'` (`set_other`), `:'+o('home')+'` (`path_home`), `'+S('pkg/node.py','go_home','pp_other','run_home')+'` (`go_home`, `set_other` в `pick_place`, путь домой после обзора)',
     '`T/test_arm.py:32`, `T/test_ideas.py:229`, `T/test_compliance.py:92`':'`'+S('T/test_arm.py','t_home')+'`, `'+S('T/test_ideas.py','t_walls')+'`, `'+S('T/test_compliance.py','t_rd')+'`',
     '`pkg/node.py:372-422` (`run`)':'`'+S('pkg/node.py','run')+'` (`run`)',
     '`T/test_node_stub.py:145` (`done` публикуется)':'`'+S('T/test_node_stub.py','t_done')+'` (`done` публикуется)',
     '`pkg/trays.py:24,75` (`SEARCH`, `detect`), `pkg/node.py:309` (`survey`), `:338` (ближний край), `:267` (окно высот), `pkg/arm.py:17-20,168,172`':
       '`'+S('pkg/trays.py','search','detect')+'` (`SEARCH`, `detect`), `'+S('pkg/node.py','survey')+'` (`survey`), `:'+o('edge')+'` (ближний край), `:'+o('band')+'` (окно высот), `'+S('pkg/arm.py','tconst','targs','tbuild')+'` (окна, `_table_args`, `table_env_build`)',
     '`T/test_compliance.py:17,26,31,35,41,52,68`':'`'+S('T/test_compliance.py','c17','c26','c31','c35','c41','c52','c68')+'`',
     '`pkg/grip.py:45-48` (сила 0.4':'`'+S('pkg/grip.py','grip')+'` (сила 0.4',
     '`pkg/node.py:143,153,176,187`':'`'+S('pkg/node.py','thumb','close','fing','req')+'` (`wait_thumb`, `grasp_close`, `wait_fingers`, `hand_requests`)',
     '`T/test_compliance.py:98,102,106,115,124`':'`'+S('T/test_compliance.py','c98','c102','c106','t_close','c124')+'`',
     '`pkg/node.py:108` (`min(.8, max(v, 1e-3))`) | — | `T/test_node_stub.py:56`':'`'+S('pkg/node.py','cmd')+'` (`min(.8, max(v, 1e-3))`) | — | `'+S('T/test_node_stub.py','t_spd')+'`',
     '`T/test_compliance.py:17` |\n':'`'+S('T/test_compliance.py','c17')+'` |\n',
     'Номера строк указаны на тот момент и с тех пор сдвинулись; в текущем коде место находится по имени функции или константы, указанному в скобках.':
       'Номера строк сверены с кодом v14 скриптом сверки рабочего репозитория (`research/diag/v14/doc_refs.py`, в сдачу не входит): каждая ссылка привязана к строке, которую она называет; имя функции или константы указано в скобках.'}
    return rows
if __name__=='__main__':
    mpf=os.path.join(R,'docs','MATH_PHYSICS.md');s=open(mpf).read();n=mp(s)
    print('MATH_PHYSICS changed refs:',sum(a!=b for a,b in zip(re.findall(r'[a-z_]+\.py:\d+(?:-\d+)?',s),re.findall(r'[a-z_]+\.py:\d+(?:-\d+)?',n))),'of',len(re.findall(r'(?<![\w/])[a-z_]+\.py:\d+',s)))
    of=os.path.join(R,'docs','ORG_ANSWERS.md');t=open(of).read();rows=org(t);miss=[k for k in rows if k not in t]
    print('ORG_ANSWERS old anchors still in doc:',[k for k in rows if k in t and k!=rows[k]]);print('ORG_ANSWERS current refs missing:',[v[:80] for v in rows.values() if v not in t])
    if '--fix' in sys.argv:
        open(mpf,'w').write(n)
        for k,v in rows.items():t=t.replace(k,v)
        open(of,'w').write(t)
