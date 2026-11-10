import json,base64,io,tarfile,sys,os
slug=sys.argv[1];steps=sys.argv[2].split(',')
files={k:open(k).read() for k in ['setup_env.sh','run_stage34.sh','grab_frames.py','run_sort.sh','score.py']}
b=io.BytesIO()
with tarfile.open(fileobj=b,mode='w:gz') as t:t.add('/workspace/solution/nssr-sol-team-048',arcname='solution',filter=lambda i:None if '__pycache__' in i.name or '/build' in i.name or '/install' in i.name else i)
sol=base64.b64encode(b.getvalue()).decode()
env=json.loads(sys.argv[3]) if len(sys.argv)>3 else {}
src=f"""import subprocess,os,time,base64,io,tarfile
F={files!r}
for k,v in F.items():open('/kaggle/working/'+k,'w').write(v)
tarfile.open(fileobj=io.BytesIO(base64.b64decode({sol!r})),mode='r:gz').extractall('/kaggle/working')
os.environ.update({env!r})
t=time.time()
def step(name,script,to):
    r=subprocess.run(['bash','/kaggle/working/'+script],stdout=open('/kaggle/working/'+name+'.log','w'),stderr=subprocess.STDOUT,timeout=to)
    print(name,'rc',r.returncode,round(time.time()-t),flush=True)
for st in {steps!r}:
    if st=='setup':
        step('setup','setup_env.sh',5400)
        for f in ['gpu.txt','bodyctrl_build.log','nssr_pkg.log','isaac_env.log']:
            try:print(f,open('/kaggle/working/logs/'+f).read()[-1200:],flush=True)
            except Exception as e:print(f,e)
    if st=='stage34':
        step('stage34','run_stage34.sh',3000)
        try:print(open('/kaggle/working/logs/check_interfaces.log').read()[-5000:],flush=True)
        except Exception as e:print(e)
    if st=='sort':
        step('sort','run_sort.sh',30000)
        try:print(open('/kaggle/working/results.jsonl').read()[-5000:],flush=True)
        except Exception as e:print(e)
"""
os.makedirs(slug,exist_ok=True);open(f'{slug}/{slug}.py','w').write(src)
json.dump({"id":f"dmitrijkv/{slug}","title":slug,"code_file":f"{slug}.py","language":"python","kernel_type":"script","is_private":True,"enable_gpu":True,"enable_internet":True,"dataset_sources":["dmitrijkv/team048-polygon-assets"],"kernel_sources":[],"competition_sources":[],"machine_shape":"NvidiaTeslaT4"},open(f'{slug}/kernel-metadata.json','w'))
print(len(src))
