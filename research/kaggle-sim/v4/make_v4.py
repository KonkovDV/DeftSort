import json,base64,io,tarfile,os
files={k:open(k).read() for k in ['setup_env.sh','run_stage34.sh','grab_frames.py','run_sort.sh','score.py']}
b=io.BytesIO()
with tarfile.open(fileobj=b,mode='w:gz') as t:t.add('/workspace/solution/nssr-sol-team-048',arcname='solution',filter=lambda i:None if '__pycache__' in i.name or '/build' in i.name or '/install' in i.name else i)
sol=base64.b64encode(b.getvalue()).decode()
src=f"""import subprocess,os,time,base64,io,tarfile,threading,signal,glob,datetime
W='/kaggle/working';os.makedirs(W+'/logs',exist_ok=True)
F={files!r}
for k,v in F.items():open(W+'/'+k,'w').write(v)
tarfile.open(fileobj=io.BytesIO(base64.b64decode({sol!r})),mode='r:gz').extractall(W)
os.environ.update(dict(TASKS='sort3',HACK_POOL='3',SEED0='100',NLAYOUTS='3',RUNTO='600'))
T0=time.time();DEADLINE=T0+4*3600;cur=['init']
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
            d=subprocess.run('df -h /tmp|tail -1',shell=True,capture_output=True,text=True,timeout=20).stdout.strip()
            log('HB step=%s gpu=[%s] disk=[%s] last=%s :: %s'%(cur[0],g,d,os.path.basename(fs[0]) if fs else '-',tl))
        except Exception as e:log('HB err',e)
threading.Thread(target=hb,daemon=True).start()
def step(name,script,to):
    cur[0]=name;to=min(to,DEADLINE-time.time());log('START',name,'timeout',int(to))
    p=subprocess.Popen(['bash',W+'/'+script],stdout=open(W+'/'+name+'.log','w'),stderr=subprocess.STDOUT,start_new_session=True)
    try:rc=p.wait(timeout=to)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid,signal.SIGKILL);rc='TIMEOUT'
    log('END',name,'rc',rc);return rc
def rd(f):
    try:return open(W+'/logs/'+f,errors='ignore').read()
    except Exception as e:return ''
log('GPU',subprocess.run('nvidia-smi -L',shell=True,capture_output=True,text=True).stdout.strip(),'internet',subprocess.run('timeout 10 curl -sI https://pypi.org -o /dev/null -w %{{http_code}}',shell=True,capture_output=True,text=True).stdout)
step('setup','setup_env.sh',5400)
ok={{'ros':'ros_env_rc=0' in rd('ros_env.log'),'isaac':'isaac_env_rc=0' in rd('isaac_env.log'),'bodyctrl':'bodyctrl import OK' in rd('bodyctrl_build.log'),'nssr':'nssr_sim OK' in rd('nssr_pkg.log')}}
log('SETUP_CHECK',ok)
for f in ['ros_env.log','isaac_env.log','bodyctrl_build.log','nssr_pkg.log']:log('TAIL',f,rd(f)[-800:])
if not all(ok.values()):log('FAIL-FAST: setup incomplete');raise SystemExit(1)
step('stage34','run_stage34.sh',2700)
log('SIM_TASK',rd('sim_task.txt')[-500:]);log('CHECK',rd('check_interfaces.log')[-3000:])
if 'ALL OK' not in rd('check_interfaces.log'):
    log('TAIL sim log',''.join(rd(os.path.basename(f))[-3000:] for f in glob.glob(W+'/logs/sim_*.log')));log('FAIL-FAST: check_interfaces not ALL OK');raise SystemExit(2)
step('sort','run_sort.sh',3*900+1500)
try:log('RESULTS',open(W+'/results.jsonl').read()[-4000:])
except Exception as e:log('RESULTS missing',e)
log('DONE')
"""
os.makedirs('k',exist_ok=True);open('k/team048-sim-stage34.py','w').write(src)
json.dump({"id":"dmitrijkv/team048-sim-stage34","title":"team048-sim-stage34","code_file":"team048-sim-stage34.py","language":"python","kernel_type":"script","is_private":True,"enable_gpu":True,"enable_internet":True,"dataset_sources":["dmitrijkv/team048-polygon-assets"],"kernel_sources":[],"competition_sources":[],"machine_shape":"NvidiaTeslaT4"},open('k/kernel-metadata.json','w'))
print(len(src))
