# Decision analysis attempt vs abstain from OOF predictions (GroupKFold by layout)
exec(open('stats_eval.py').read().split('OUT={}')[0])
G=.7;D=(0,.1,.2,.3);ths=np.linspace(.34,.99,66);R={}
fig,axs=plt.subplots(1,2,figsize=(10,4))
for ax,pool in zip(axs,(3,6)):
    rows=[r for r in ALL if r['pool']==pool];pr,pred,y,g=oof(rows,'f6+hist+npix',True)
    single=np.array([r['purity']>=.9 for r in rows]);tc=np.array([r['category'] for r in rows])
    ok=single&(pred==tc);nl=len(np.unique(g));R[pool]={}
    for d in D:
        ev=[]
        for t in ths:
            acc=(pr>=t)&(pred!='MERGED')
            s=10*G*(((1-d)-d)*(acc&ok).sum()-(acc&~ok).sum())
            ev.append(s/nl)
        i=int(np.argmax(ev));R[pool][d]=dict(best_th=float(ths[i]),ev_per_layout=float(ev[i]),ev_at_08=float(ev[np.argmin(abs(ths-.8))]),theory_th=1/(2*(1-d)))
        ax.plot(ths,ev,label=f'd={d}')
    ax.set_title(f'Пул {pool}: E[очки]/раскладка, g={G}');ax.set_xlabel('порог уверенности');ax.legend()
fig.tight_layout();fig.savefig('fig/decision.png',dpi=110);json.dump(R,open('decision.json','w'),indent=1);print(json.dumps(R,indent=1))
