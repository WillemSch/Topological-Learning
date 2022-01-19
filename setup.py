from setuptools import setup

with open("README.md", "r") as file:
    long_desc = file.read()

setup(
    name='TopologicalLearningTechniques',
    vesrion='0.0.1',
    description='Package created for INF367AII at UiB',
    py_modules=["main"],  # All modules here
    package_dir={'': 'src'},
    long_description=long_desc,
    long_description_content_type="text/markdown",
    install_requires=[
        # "numpy>=1.0",     <- Example
    ],
    extra_requires={
        "dev": [
            # "numpy>=1.0",     <- Example
        ]
    },
)