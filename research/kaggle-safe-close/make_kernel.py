import re,json,pathlib
src=pathlib.Path('/workspace/kaggle/isaac500-native-stacks-0c44bed3f3/isaac500-native-stacks-0c44bed3f3.py').read_text()
lines=src.split('\n')
ns={};idx=[i for i,l in enumerate(lines) if l.startswith('SCENE=')][0]
exec(lines[idx],ns);scene=ns['SCENE']
PRE='''
def safe_pre_close(app):
 # Fix 3 (workaround): release ROS2 bridge + simulation_manager event subscriptions
 # while carb.events is still alive, i.e. before SimulationApp.close()->unload_all_plugins().
 import omni.kit.app
 em=omni.kit.app.get_app().get_extension_manager();res=[]
 for name in ['isaacsim.ros2.bridge','isaacsim.core.simulation_manager']:
  try:
   was=em.is_extension_enabled(name)
   if was:em.set_extension_enabled_immediate(name,False)
   for _ in range(5):app.update()
   res.append({'ext':name,'was_enabled':was,'enabled_after':em.is_extension_enabled(name)})
  except Exception as e:res.append({'ext':name,'error':str(e)})
 r['safe_pre_close']=res;save()
'''
anchor="try:\n if mode=='never_bare':"
assert anchor in scene
scene=scene.replace(anchor,PRE+'\n'+anchor,1)
old="sys.settrace(close_trace);t=time.monotonic();app.close();sys.settrace(None)"
assert old in scene
scene=scene.replace(old,"safe_pre_close(app);"+old,1)
old2=" sys.exit(code)"
assert scene.rstrip().endswith(old2.strip())
scene=scene.rstrip()[:-len('sys.exit(code)')]+"r['post_close_exit']='os._exit after flush (skip interpreter finalization of unloaded native plugins)';save();sys.stdout.flush();sys.stderr.flush();os._exit(code)\n"
lines[idx]='SCENE='+repr(scene)
out='\n'.join(lines)
out=out.replace("modes=['bridge_inactive','clock_only','rclpy_only','both']","modes=['clock_only','both']")
out=out.replace("if mode=='both' and all_clean and","if mode=='both' and")  # run joint even if partial, to collect full acceptance
pathlib.Path('isaac500-safe-close-v1.py').write_text(out)
