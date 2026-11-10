"""Offline Monte-Carlo of full sort2 (270 s) / sort3 (405 s) episodes with the node's loop structure, to choose object
order, skip and budget policies. Inputs: measured v7 phase times (phases.py, sim s), per-object feasibility and planned
pick durations from the reach study (results.jsonl of the new planner), execution success SCENARIOS (not measured yet
for the new code -- clearly labelled). Score per SCORING.md: +10 in own tray, -10 lifted and left outside / in hand at the
limit, 0 untouched."""
import json,sys,glob,collections,numpy as np
sys.path.insert(0,'/workspace/fixes/budget');import phases as PH
C=[c for r in PH.rows for c in r['cycles']]
PLAN=np.array([c['plan'] for c in C if c.get('plan') is not None])
PICK=np.array([c['pick'] for c in C if c.get('pick') is not None])
PLACE=np.array([c['place'] for c in C if c.get('place') is not None])
TOPRE=np.array([c['to_pre']+c['approach_close'] for c in C if c.get('approach_close') is not None])
STARTUP=3.6; FINAL=4.0   # measured startup (s102/s103); final retreat+done (go_home from home ~0, waist back)
YAW_T=lambda y:2*abs(y)/.8+1.0   # waist out and back at 0.8 rad/s + settle (node.waist)
def load_reach(pattern):
    R=[json.loads(l) for f in glob.glob(pattern) for l in open(f)]
    by=collections.defaultdict(list)
    for r in R:by[r['layout']].append(r)
    return by
SCEN={  # P(held after lift | feasible plan), P(lands in tray | held). v7 old planner: side 0/6 held, rim 2/5 held, 1 of 3 held rims landed.
 'pessimistic(v7-like)':dict(side=(.35,.7),rim=(.4,.6)),
 'mid':dict(side=(.65,.85),rim=(.55,.75)),
 'optimistic':dict(side=(.85,.95),rim=(.75,.9))}
def episode(objs,limit,pol,sc,rng,plan_scale=1.0):
    """objs: list of dict(cat,result,kind,yaw,pick_dur,plan_s). Returns score, n_ok, n_bad, t_end."""
    t=STARTUP;score=0;ok=0;bad=0;tout=0;fails=collections.Counter();handled=set();infeas_seen={}
    pend=list(range(len(objs)))
    if pol['order']=='ev':
        def key(i):   # NON-oracle: class prior feasibility (reach study) x scenario success / expected time with PREDICTED yaw
            o=objs[i];pf=pol['prior'].get(o['cat'],.5)
            ps,pp=sc['rim' if o['cat']=='dish' else 'side']
            yt=YAW_T(2.22) if o['yaw_pred'] else 0.
            return -(pf*(ps*pp*10-ps*(1-pp)*10))/(np.mean(PLAN)+pf*(np.mean(PICK)+np.mean(PLACE)+yt))
        pend.sort(key=key)
    else:rng.shuffle(pend)   # identification confidence ~ unrelated to feasibility
    cycle_est=35.0
    while True:
        progressed=False;stop=False
        for i in list(pend):
            o=objs[i]
            if fails[i]>=2:continue
            if pol.get('cache') and o['result']!='ok' and infeas_seen.get(i)==len(handled):continue
            # planning cost (sim s): v7 measured observe->start, scaled; infeasible searches cost as much (all candidates tried)
            if t+cycle_est+8>limit and pol['budget']=='ema':stop=True;break
            t+=rng.choice(PLAN)*plan_scale
            if o['result']!='ok':fails[i]+=1;infeas_seen[i]=len(handled);continue
            ps,pp=sc['rim' if o['kind']=='rim' else 'side']
            yt=YAW_T(o['yaw']) if o['yaw'] else 0.
            if pol['budget']=='node':need=o['pick_dur']+(yt)+12+8
            else:need=pol['k']*o['pick_dur']+np.quantile(PLACE,.9)+yt+8
            if limit-t<need:stop=True;break       # node: return None -> stop the episode
            t_pick=rng.choice(PICK);t_place=rng.choice(PLACE)+yt
            if rng.random()>ps:                    # empty after lift: retreat, 0 points
                t+=t_pick+5.;fails[i]+=1
                if t>limit:break
                continue
            if t+t_pick+t_place>limit:score-=10;bad+=1;tout+=1;t=limit;stop=True;break   # still in hand at the limit
            t+=t_pick+t_place;pend.remove(i);handled.add(i);progressed=True
            cyc=t_pick+t_place;cycle_est=max(.5*cyc+.5*cycle_est,cyc) if cyc>cycle_est else .5*cyc+.5*cycle_est
            if rng.random()<pp:score+=10;ok+=1
            else:score-=10;bad+=1
            break
        if stop or not progressed and all(fails[i]>=2 or (pol.get('cache') and objs[i]['result']!='ok' and infeas_seen.get(i)==len(handled)) for i in pend):break
        if not pend:break
        if t>limit:break
    return score,ok,bad,tout
POLICIES={
 'P0 node as is (confidence order, tail 12 s)':dict(order='rand',budget='node'),
 'P1 calibrated budget (1.7x planned + p90 tail)':dict(order='rand',budget='cal',k=1.7),
 'P2 P1 + EV/time order':dict(order='ev',budget='cal',k=1.7),
 'P3 P2 + infeasible cache':dict(order='ev',budget='cal',k=1.7,cache=True),
 'P4 P1 + infeasible cache':dict(order='rand',budget='cal',k=1.7,cache=True)}
def run(pattern,limit,N=4000,plan_scales=(1.0,2.0)):
    by=load_reach(pattern);lays=list(by.values());rng=np.random.default_rng(0);out={}
    allr=[r for L in lays for r in L];prior={c:np.mean([r['result']=='ok' for r in allr if r['cat']==c]) for c in set(r['cat'] for r in allr)}
    print('class feasibility prior',{k:round(v,3) for k,v in prior.items()})
    for ps_ in plan_scales:
        for sn,sc in SCEN.items():
            for pn,pol in POLICIES.items():
                S=[];O=[];B=[];T=[]
                for k in range(N):
                    L=lays[k%len(lays)]
                    objs=[dict(cat=r['cat'],result=r['result'],kind=r.get('kind'),yaw=r.get('yaw',0.) or 0.,pick_dur=r.get('pick_dur',10.),
                               yaw_pred=(r['tray']=='tray_left' and r['pose'][1]<0) or (r['tray']=='tray_right' and r['pose'][1]>0)) for r in L]
                    pol=dict(pol,prior=prior)
                    s,o,b,to=episode(objs,limit,pol,sc,rng,ps_);S.append(s);O.append(o);B.append(b);T.append(to)
                out[(ps_,sn,pn)]=(np.mean(S),np.mean(O),np.mean(B),np.quantile(S,.1),np.mean(T),np.std(S)/np.sqrt(N))
    return out,len(lays)
if __name__=='__main__':
    pattern,limit=sys.argv[1],float(sys.argv[2])
    out,n=run(pattern,limit)
    print('layouts',n,'limit',limit)
    for (ps_,sn,pn),(m,o,b,q,to,se) in out.items():print(f'plan x{ps_:.0f} | {sn:20s} | {pn:46s} | E {m:5.1f}+-{se:3.1f} ok {o:4.2f} bad {b:4.2f} (timeout-in-hand {to:4.2f}) p10 {q:4.0f}')
