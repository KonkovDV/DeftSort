import json,base64,io,tarfile,os
# v8: setup -> sort: sort3 seed 103 + sort2 seed 101, pool 3; hard deadline 50 min after start.
files={k:open(k).read() for k in ['setup_env.sh','grab_frames.py','run_sort.sh','score.py']}
SKIP=('__pycache__','/build','/install','/log/','.git','.bak','.pytest_cache')
b=io.BytesIO()
with tarfile.open(fileobj=b,mode='w:gz') as t:t.add('/workspace/solution/nssr-sol-team-048',arcname='solution',filter=lambda i:None if any(x in i.name for x in SKIP) else i)
sol=base64.b64encode(b.getvalue()).decode()
src=f"""import subprocess,os,time,base64,io,tarfile,threading,signal,glob,datetime,json
W='/kaggle/working';os.makedirs(W+'/logs',exist_ok=True)
F={files!r}
for k,v in F.items():open(W+'/'+k,'w').write(v)
tarfile.open(fileobj=io.BytesIO(base64.b64decode({sol!r})),mode='r:gz').extractall(W)
T0=time.time();DEADLINE=T0+50*60;cur=['init']
os.environ.update(dict(TASKS='sort3 sort2',HACK_POOL='3',SEED0='100',SEEDS_sort3='103',SEEDS_sort2='101',RUNTO_sort3='840',RUNTO_sort2='600',DEADLINE_EPOCH=str(int(DEADLINE))))
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
log('GPU',subprocess.run('nvidia-smi -L',shell=True,capture_output=True,text=True).stdout.strip())
step('setup','setup_env.sh',1500)
ok={{'ros':'ros_env_rc=0' in rd('ros_env.log'),'isaac':'isaac_env_rc=0' in rd('isaac_env.log'),'bodyctrl':'bodyctrl import OK' in rd('bodyctrl_build.log'),'nssr':'nssr_sim OK' in rd('nssr_pkg.log')}}
log('SETUP_CHECK',ok)
if not all(ok.values()):
    for f in ['ros_env.log','isaac_env.log','bodyctrl_build.log','nssr_pkg.log']:log('TAIL',f,rd(f)[-800:])
    log('FAIL-FAST: setup incomplete');raise SystemExit(1)
step('sort','run_sort.sh',DEADLINE-time.time())
log('BUILD',rd('sol_build.log')[-300:])
try:log('RESULTS',open(W+'/results.jsonl').read()[-6000:])
except Exception as e:log('RESULTS missing',e)
for f in sorted(glob.glob(W+'/logs/run_sort*_*.log')):
    L=[l for l in open(f,errors='ignore') if 'team048_solution' in l]
    log('RUNLOG',os.path.basename(f),len(L),''.join(L[-40:])[-4000:])
log('DONE')
"""
os.makedirs('k',exist_ok=True);open('k/team048-sim-stage34.py','w').write(src)
json.dump({"id":"dmitrijkv/team048-sim-stage34","title":"team048-sim-stage34","code_file":"team048-sim-stage34.py","language":"python","kernel_type":"script","is_private":True,"enable_gpu":True,"enable_internet":True,"dataset_sources":["dmitrijkv/team048-polygon-assets"],"kernel_sources":[],"competition_sources":[],"machine_shape":"NvidiaTeslaT4"},open('k/kernel-metadata.json','w'))
print('src',len(src),'sol_tar_b64',len(sol))
