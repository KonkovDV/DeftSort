import sys,numpy as np,struct,xml.etree.ElementTree as ET
sys.path.insert(0,'/workspace/solution/nssr-sol-team-048/src/team048_solution')
from team048_solution import arm as A
M='/workspace/competitors/UBTECH-Robot__tiangong2pro_tf/src/tiangong2pro_urdf/meshes/'
def stl(p):
    b=open(p,'rb').read()
    if b[:5]==b'solid' and b'facet' in b[:300]:
        v=[list(map(float,l.split()[1:4])) for l in b.decode(errors='ignore').splitlines() if l.strip().startswith('vertex')];return np.unique(np.array(v),axis=0)
    n=struct.unpack('<I',b[80:84])[0];a=np.frombuffer(b[84:84+n*50],dtype=np.dtype([('n','<3f4'),('v','<9f4'),('a','<u2')]))
    return np.unique(a['v'].reshape(-1,3).astype(float),axis=0)
U=ET.parse('/workspace/competitors/UBTECH-Robot__tiangong2pro_tf/src/tiangong2pro_urdf/urdf/tiangong2.0_pro_with_hands.urdf').getroot()
for l in U.findall('link'):
    if l.get('name') in ('left_index_2','left_thumb_4'):
        v=l.find('visual');o=v.find('origin');print(l.get('name'),'visual origin',None if o is None else (o.get('xyz'),o.get('rpy')),v.find('geometry/mesh').get('filename'))
def pts(side,link,q):
    V=stl(M+link+'.STL');T=A.fk_link(link,q);H=A.fk_link(A.SIDE[side]['hand'],q)
    return (np.linalg.inv(H)@T@np.c_[V,np.ones(len(V))].T).T[:,:3]
def q_of(p,cf,ct):
    q={f'{p}_thumb_1_joint':1.246165,f'{p}_thumb_2_joint':ct*.48,f'{p}_thumb_3_joint':ct*.48*1.1425,f'{p}_thumb_4_joint':ct*.48*1.1425*.7508}
    for f in ('index','middle','ring','little'):q[f'{p}_{f}_1_joint']=cf*1.333;q[f'{p}_{f}_2_joint']=cf*1.333*1.1169
    return q
for side in 'l':
    p='left';q0=q_of(p,0,0)
    th=pts(side,f'{p}_thumb_4',q0);ix=pts(side,f'{p}_index_2',q0)
    print('open: thumb4 mesh extent',th.min(0).round(3),th.max(0).round(3));print('index2',ix.min(0).round(3),ix.max(0).round(3))
    est_t=A.fingertip(side,'thumb',q0);est_i=A.fingertip(side,'index',q0)
    print('est tips thumb',est_t.round(3),'nearest mesh pt dist',np.linalg.norm(th-est_t,axis=1).min().round(3),'index',est_i.round(3),np.linalg.norm(ix-est_i,axis=1).min().round(3))
    # opening at open: min distance thumb mesh to index/middle meshes
    from scipy.spatial import cKDTree
    res=[]
    for c in np.linspace(0,1,21):
        q=q_of(p,c,c);th=pts(side,f'{p}_thumb_4',q);F=np.vstack([pts(side,f'{p}_{f}_2',q) for f in ('index','middle')])
        d,i=cKDTree(F).query(th);k=np.argmin(d);res.append((c,d[k],(th[k]+F[i[k]])/2))
    for c,d,m in res[::2]:print('closure',round(c,2),'min thumb-finger gap',round(d,3),'at',m.round(3))
    first=[r for r in res if r[1]<.004]
    print('first contact',first[0][0] if first else None,first[0][2].round(3) if first else None)
