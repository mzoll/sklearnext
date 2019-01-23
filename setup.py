from setuptools import setup

setup(name='sklearnext',
      version='0.1.3',
      description='pandas compatibility and pipelining extensions to scikit-learn (sklearn)',
      url='http://github.com/mzoll/sklearnext',
      author='Marcel Zoll',
      author_email='marcel.zoll.physics@gmail.com',
      license='',
      packages=['sklearnext'],
      zip_safe=False,
      keywords='sklearn pipeline data analysis',
      install_requires=['pytest, scikit-learn, numpy, pandas'],
      #classifieres=['Programming Language :: Python :: 3.6']
)