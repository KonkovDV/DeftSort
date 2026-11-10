from v7_geom import *
from PIL import Image,ImageDraw
import sys
def proj(z,pts):
    T=ARM.fk_link('camera_head_link',{'head_pitch_joint':float(z['head_pitch'])});Ti=np.linalg.inv(T)
    out=[]
    for p in pts:
        c=Ti@np.r_[p,1];out.append((343.16*c[0]/c[2]+320,343.16*c[1]/c[2]+180,c[2]))
    return out
def hand_pts(side,qarm,fing=None):
    A=ARM.Arm.__new__(ARM.Arm);names=[f'{n}_{side}_joint' for n in ARM.ARM]
    q=dict(zip(names,qarm));pre='left' if side=='l' else 'right'
    P={}
    P['tcp']=ARM.fk_link(ARM.SIDE[side]['tip'],q)[:3,3]
    P['base']=ARM.fk_link(ARM.SIDE[side]['hand'],q)[:3,3]
    for fn in ('thumb_4','index_2','middle_2','little_2'):
        P[fn]=ARM.fk_link(f'{pre}_{fn}',q)[:3,3]
    h=ARM.hand_opening(side);H=ARM.fk_link(ARM.SIDE[side]['hand'],q)
    P['grip']=(H@np.r_[h['grip_center_hand'],1])[:3]
    P['wrist']=ARM.fk_link(f'wrist_pitch_{side}_link',q)[:3,3]
    P['elbow']=ARM.fk_link(f'elbow_pitch_{side}_link',q)[:3,3]
    return P
if __name__=='__main__':
    f=sys.argv[1];side=sys.argv[2];z=np.load(f);P=hand_pts(side,z['q'])
    im=Image.fromarray(z['rgb']).convert('RGB');dr=ImageDraw.Draw(im)
    col=dict(tcp='red',base='blue',thumb_4='magenta',index_2='cyan',middle_2='lime',little_2='orange',grip='yellow',wrist='white',elbow='black')
    for k,(u,v,d) in zip(P,proj(z,P.values())):
        dr.ellipse([u-4,v-4,u+4,v+4],outline=col[k],width=2);print(k,np.round(P[k],3),'px',round(u),round(v))
    # also the depth point at those pixels
    out=sys.argv[3];im.save(out)
