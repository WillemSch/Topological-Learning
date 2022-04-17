from setuptools import setup

with open("README.md", "r") as file:
    long_desc = file.read()

setup(
    name='TopologicalLearningTechniques',
    version='0.1.1',
    description='Package created for INF367AII at UiB',
    py_modules=["gng", "graph", "homology", "neural_net", "reeb", "som", "util"],  # All modules here
    package_dir={'': 'src'},
    long_description=long_desc,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy~=1.21.3",
        "torch~=1.11.0",
        "scikit-learn~=1.0.2",
        "scipy~=1.7.2",
    ],
    extra_requires={
        "dev": [
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.8",
    ],
    license_files=('LICENSE.txt',),
)
