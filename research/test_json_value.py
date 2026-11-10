import json,math,numpy as np,ast
src=open('/workspace/kaggle/isaac500-lifecycle-native-dc2ccbf7c2/out/executed_receiver.py').read()
ns={};exec(src.split('def atomic_report')[0],ns);jv=ns['json_value']
d={'b':np.bool_(True),'i':np.int64(3),'f':np.float32(1.5),'nan':float('nan'),'arr':np.arange(3),'nested':{1:(np.bool_(False),)}}
s=json.dumps(jv(d),allow_nan=False);print(s)
assert json.loads(s)=={'b':True,'i':3,'f':1.5,'nan':None,'arr':[0,1,2],'nested':{'1':[False]}}
try: jv(object()); raise SystemExit('should fail closed')
except TypeError: print('fail-closed ok')
