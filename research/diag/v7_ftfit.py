from v7_proj import *
import glob,json
rows=[]
for f in sorted(glob.glob('/workspace/fixes/kaggle-sim/out_v7/trace/*/*.npz')):
    side=os.path.basename(f).split('_')[2];z=np.load(f);rec=json.loads(str(z['rec']))
    q=dict(zip([f'{n}_{side}_joint' for n in ARM.ARM],z['q']));R=ARM.fk_link(f'wrist_roll_{side}_link',q)[:3,:3]
    gw=R.T@np.array([0,0,-9.81]);fing=np.array(rec['fing']);eff=np.array(rec['eff'])
    ph=rec['phase'];obj=rec['obj']
    if ph=='released':lab='empty'
    elif ph=='lifted' and (fing[:4].max()<.15 or (fing[:5]<.25).all()):lab='empty'
    elif ph=='lifted' and eff[2:4].min()>.9 and fing[2:4].min()>.5:lab='loaded'
    else:lab='?'
    if 'Perricone' in obj and ph=='released':lab='collision'
    rows.append(dict(f=os.path.basename(os.path.dirname(f))[-3:]+' '+ph+' '+obj[:10],side=side,gw=gw,F=np.array(rec['F']),lab=lab,fing=fing,eff=eff))
E=[r for r in rows if r['lab']=='empty']
for side in 'lr':
    S=[r for r in E if r['side']==side]
    if len(S)<3:continue
    A=np.vstack([np.c_[r['gw'][:,None],np.eye(3)] for r in S]);b=np.concatenate([r['F'] for r in S])
    x,*_=np.linalg.lstsq(A,b,rcond=None);res=b-A@x
    print(side,'n',len(S),'fit m_err(kg)',round(x[0],3),'offset',x[1:].round(2),'resid rms',round(np.sqrt((res**2).mean()),3))
    for r in rows:
        if r['side']!=side:continue
        pred=x[0]*r['gw']+x[1:];r['dev']=np.linalg.norm(r['F']-pred)
for r in rows:print(r['f'].ljust(28),r['side'],r['lab'].ljust(9),'|F|',round(np.linalg.norm(r['F']),2),'|F-pred_empty|',round(r.get('dev',np.nan),2),'fing',np.round(r['fing'][:5],2),'eff',np.round(r['eff'][:5],2))
