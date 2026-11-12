from setuptools import setup
setup(name='team048_solution',version='0.1.0',packages=['team048_solution'],
      description='Team 048 NSSR hackathon solution',license='Apache-2.0',maintainer='team048',maintainer_email='team048@example.com',
      package_data={'team048_solution':['data/*']},
      data_files=[('share/ament_index/resource_index/packages',['resource/team048_solution']),('share/team048_solution',['package.xml'])],
      install_requires=['setuptools'],tests_require=['pytest'],zip_safe=False,
      entry_points={'console_scripts':['solution=team048_solution.node:main']})
