import json,os,make_cfg as M
from pathlib import Path
core=Path('study_core.py').read_text()
shards=[]
def add(sid,cfg):shards.append((sid,cfg))
for i,(a,b) in enumerate([(0,20),(20,40),(40,60),(60,80),(80,100)]):add(f'r3s3-{i}',M.cfg('reach',3,'sort3',a,b))
for i,(a,b) in enumerate([(0,34),(34,67),(67,100)]):add(f'r3s2-{i}',M.cfg('reach',3,'sort2',a,b))
for i,(a,b) in enumerate([(0,34),(34,67),(67,100)]):add(f'r6s2-{i}',M.cfg('reach',6,'sort2',a,b))
for i,(a,b) in enumerate([(0,15),(15,30),(30,45)]):add(f'g3d-{i}',M.cfg('grasp',3,'sort3',a,b,grasp_cats=['dish']))
for i,(a,b) in enumerate([(0,8),(8,16),(16,24),(24,32)]):add(f'g6-{i}',M.cfg('grasp',6,'sort2',a,b,grasp_cats=['electronics','box']))
for i,(a,b) in enumerate([(0,12),(12,24)]):add(f'g3b-{i}',M.cfg('grasp',3,'sort3',a,b,grasp_cats=['box']))
add('r6x-0',M.cfg('reach',6,'sort2',100,200))
add('tubes',dict(mode='tubes',jobs=M.tubes_jobs(),meta={}))
root=Path('kernels');root.mkdir(exist_ok=True)
for sid,cfg in shards:
    cfg.setdefault('deadline_s',48*60)
    d=root/sid;d.mkdir(exist_ok=True)
    slug=f'team048-cpu-s-{sid}'
    head=f"import os,json\nos.environ['STUDY_CFG']=json.dumps(json.loads(r'''{json.dumps(cfg)}'''))\nos.environ.setdefault('OMP_NUM_THREADS','1')\n"
    (d/f'{slug}.py').write_text(head+core)
    (d/'kernel-metadata.json').write_text(json.dumps(dict(id=f'dmitrijkv/{slug}',title=slug,code_file=f'{slug}.py',language='python',kernel_type='script',
        is_private=True,enable_gpu=False,enable_tpu=False,enable_internet=False,dataset_sources=['dmitrijkv/team048-tpu-cpu-inputs'],kernel_sources=[],competition_sources=[],model_sources=[]),indent=1))
print(len(shards),[s for s,_ in shards])
