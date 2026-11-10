"""v9 kernels (one task per session, fix 1 of fixes/V8_ANALYSIS_RU.md). Builds k<A|B>/ (code + kernel-metadata.json)
from a COMMITTED git ref of the solution (git archive: no untracked/dirty files). Does NOT push anything.
  python3 make_v9.py A [--ref master] [--repo /workspace/solution/nssr-sol-team-048] [--force]
Session A: sort3 seed 103 (same layout as v8/v9a), HACK_SELF_COLLISION=1, wall cap RUNTO 2000 s.
Session B: sort2 seed 101 (never completed before), wall cap RUNTO 1400 s.
Budget (v9fix, calibrated on v8/v9a/v9b, fixes/V9_ANALYSIS_RU.md): setup cap 2400 s (v9a setup 648 s, v9b > 1500 s:
pip isaaclab), Isaac boot SIMBOOT_MAX 1500 s (v8 22 min, v9a 18.3 min), run at RTF 0.21-0.245 (sort3 405 s sim ->
1650-1930 s wall, sort2 270 s -> 1100-1290 s), session cap 100 min (--session-min). run_sort.sh shortens RUNTO to the
time left before the session cap (logged), so a slow setup costs run time, never the scoring/output step."""
import json,base64,io,os,subprocess,sys,argparse
ap=argparse.ArgumentParser();ap.add_argument('variant',choices=['A','B']);ap.add_argument('--ref',default='master')
ap.add_argument('--repo',default='/workspace/solution/nssr-sol-team-048');ap.add_argument('--force',action='store_true')
ap.add_argument('--session-min',type=int,default=100);ap.add_argument('--setup-cap',type=int,default=2400)
ap.add_argument('--simboot',type=int,default=1500);ap.add_argument('--out',default=None);ap.add_argument('--runto',type=int,default=None)
a=ap.parse_args();H=os.path.dirname(os.path.abspath(__file__))
VAR={'A':dict(TASK='sort3',SEEDS='103',RUNTO='2000',HACK_SELF_COLLISION='1',slug='team048-sim-v9a'),
     'B':dict(TASK='sort2',SEEDS='101',RUNTO='1400',HACK_SELF_COLLISION='0',slug='team048-sim-v9b')}[a.variant]
if a.runto:VAR['RUNTO']=str(a.runto)
if a.setup_cap+a.simboot+int(VAR['RUNTO'])+300>a.session_min*60:print('NOTE: worst case setup+boot+run+300 s exceeds the session cap; run_sort.sh will shorten RUNTO')
git=lambda *c:subprocess.run(['git','-C',a.repo,*c],capture_output=True,text=True)
sha=git('rev-parse',a.ref).stdout.strip()
if not sha:sys.exit(f'unknown ref {a.ref}')
# the merged master must contain: compliance (4cb594b), Rust phase 2 (68d6b61), v8 fixes (v8fixes head)
need={'compliance':'4cb594b','rust-accel2':'68d6b61','v8fixes':git('rev-parse','v8fixes').stdout.strip(),
      'v9fix':git('rev-parse','v9fix').stdout.strip()}   # v9fix: start path / torso envelope / guard (V9_ANALYSIS_RU.md)
miss=[k for k,c in need.items() if git('merge-base','--is-ancestor',c,sha).returncode!=0]
if miss and not a.force:sys.exit(f'{a.ref}={sha[:7]} does not contain {miss}; merge first (or --force for a diagnostic build)')
# runtime only: Rust bench data (8 MB) and test fixtures are not needed on Kaggle (wheel is vendored; NO_BUILD=1)
EXCL=[':(exclude)rust/team048_accel/benches/data',':(exclude)src/team048_solution/test/fixtures',
      ':(exclude)research',':(exclude)HANDOFF.md']   # v11: DeftSort carries research/ and HANDOFF.md, the kernel must not
tar=subprocess.run(['git','-C',a.repo,'archive','--format=tar.gz','--prefix=solution/',sha,'--','.',*EXCL],capture_output=True).stdout
names=subprocess.run(['tar','tzf','-'],input=tar,capture_output=True).stdout.decode().split()
for f in ('solution/run.sh','solution/setup.sh','solution/src/team048_solution/team048_solution/node.py'):
    if f not in names:sys.exit('missing in archive: '+f)
whl=[n for n in names if n.startswith('solution/vendor/team048_accel-') and n.endswith('.whl')]
print('ref',a.ref,sha[:10],'wheel',whl or 'NONE (NumPy reference only)','missing',miss,'session_min',a.session_min,'setup_cap',a.setup_cap,'simboot',a.simboot,'RUNTO',VAR['RUNTO'])
files={k:open(os.path.join(H,k)).read() for k in ['setup_env.sh','grab_frames.py','run_sort.sh','score.py','wait_task.py']}
sol=base64.b64encode(tar).decode()
env=dict(TASK=VAR['TASK'],SEEDS=VAR['SEEDS'],RUNTO=VAR['RUNTO'],HACK_SELF_COLLISION=VAR['HACK_SELF_COLLISION'],HACK_POOL='3',SEED0='100',SIMBOOT_MAX=str(a.simboot))
src=f"""import subprocess,os,time,base64,io,tarfile,threading,signal,glob,datetime,json
# team048 v9 session {a.variant}: {VAR['TASK']} seeds {VAR['SEEDS']}, solution {a.ref} {sha}
W='/kaggle/working';os.makedirs(W+'/logs',exist_ok=True)
F={files!r}
for k,v in F.items():open(W+'/'+k,'w').write(v)
tarfile.open(fileobj=io.BytesIO(base64.b64decode({sol!r})),mode='r:gz').extractall(W)
T0=time.time();DEADLINE=T0+{a.session_min}*60;cur=['init']
os.environ.update(dict({env!r},DEADLINE_EPOCH=str(int(DEADLINE))))
def msk():return (datetime.datetime.utcnow()+datetime.timedelta(hours=3)).strftime('%H:%M:%S MSK')
def log(*a):
    s=msk()+' +%ds '%(time.time()-T0)+' '.join(map(str,a));print(s,flush=True);open(W+'/progress.log','a').write(s+'\\n')
def hb():
    while True:
        time.sleep(60)
        try:
            fs=sorted(glob.glob(W+'/logs/*')+glob.glob(W+'/*.log'),key=os.path.getmtime)[-1:]
            tl=open(fs[0],errors='ignore').read()[-300:].replace('\\n',' | ') if fs else ''
            g=subprocess.run('nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader',shell=True,capture_output=True,text=True,timeout=20).stdout.strip()
            log('HB step=%s gpu=[%s] last=%s :: %s'%(cur[0],g,os.path.basename(fs[0]) if fs else '-',tl))
        except Exception as e:log('HB err',e)
threading.Thread(target=hb,daemon=True).start()
def step(name,script,to):
    cur[0]=name;to=max(1,min(to,DEADLINE-time.time()));log('START',name,'timeout',int(to))
    p=subprocess.Popen(['bash',W+'/'+script],stdout=open(W+'/'+name+'.log','w'),stderr=subprocess.STDOUT,start_new_session=True)
    try:rc=p.wait(timeout=to)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid,signal.SIGKILL);rc='TIMEOUT'
    log('END',name,'rc',rc);return rc
def rd(f):
    try:return open(W+'/logs/'+f,errors='ignore').read()
    except Exception as e:return ''
log('SESSION {a.variant}',{env!r},'solution {sha[:10]}')
log('GPU',subprocess.run('nvidia-smi -L',shell=True,capture_output=True,text=True).stdout.strip())
step('setup','setup_env.sh',{a.setup_cap})
ok={{'ros':'ros_env_rc=0' in rd('ros_env.log'),'isaac':'isaac_env_rc=0' in rd('isaac_env.log'),'bodyctrl':'bodyctrl import OK' in rd('bodyctrl_build.log'),'nssr':'nssr_sim OK' in rd('nssr_pkg.log')}}
log('SETUP_CHECK',ok)
if not all(ok.values()):
    for f in ['ros_env.log','isaac_env.log','bodyctrl_build.log','nssr_pkg.log']:log('TAIL',f,rd(f)[-800:])
    log('FAIL-FAST: setup incomplete');raise SystemExit(1)
step('sort','run_sort.sh',DEADLINE-time.time()-60)
log('ACCEL_SETUP',rd('accel_setup.log')[-600:])
log('BUILD',rd('sol_build.log')[-300:])
try:log('RESULTS',open(W+'/results.jsonl').read()[-6000:])
except Exception as e:log('RESULTS missing',e)
KEY=('accel','tray sources','no plan','guard:','jammed','re-observe','grasp check','pick ','done at','deadline','skip ','run rc=')
for f in sorted(glob.glob(W+'/logs/run_sort*_*.log')):
    L=[l for l in open(f,errors='ignore') if 'team048_solution' in l or 'run rc=' in l]
    log('KEYLINES',os.path.basename(f),''.join(l for l in L if any(k in l for k in KEY))[-6000:])
log('DONE')
"""
k=a.out or os.path.join(H,'k'+a.variant);os.makedirs(k,exist_ok=True);code=VAR['slug']+'.py'
open(os.path.join(k,code),'w').write(src)
json.dump({"id":"dmitrijkv/"+VAR['slug'],"title":VAR['slug'],"code_file":code,"language":"python","kernel_type":"script",
           "is_private":True,"enable_gpu":True,"enable_internet":True,"dataset_sources":["dmitrijkv/team048-polygon-assets"],
           "kernel_sources":[],"competition_sources":[],"machine_shape":"NvidiaTeslaT4"},open(os.path.join(k,'kernel-metadata.json'),'w'))
print('wrote',k,'src',len(src),'solution_tar_b64',len(sol))
