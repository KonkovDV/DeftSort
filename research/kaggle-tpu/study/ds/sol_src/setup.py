from setuptools import setup
setup(name='team048_solution',version='0.1.0',packages=['team048_solution'],
      package_data={'team048_solution':['data/*']},
      data_files=[('share/ament_index/resource_index/packages',['resource/team048_solution']),('share/team048_solution',['package.xml'])],
      install_requires=['setuptools'],zip_safe=False,
      entry_points={'console_scripts':['solution=team048_solution.node:main']})
