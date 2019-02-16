import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name='sklearnext',
      version='0.1.3',
      description='pandas compatibility and pipelining extensions to scikit-learn (sklearn)',
      url='http://github.com/mzoll/sklearnext',
      author='Marcel Zoll',
      author_email='marcel.zoll.physics@gmail.com',
      #license='',
      packages=setuptools.find_packages(),  # ['sklearnext'],
      zip_safe=False,
      long_description=long_description,
      long_description_content_type="text/markdown",
      keywords='sklearn pipeline data analysis',
      install_requires=['pytest', 'scikit-learn', 'numpy', 'pandas'],
      classifieres=[
            'Programming Language :: Python :: 3',
            "License :: None",
            "Operating System :: OS Independent"]
)