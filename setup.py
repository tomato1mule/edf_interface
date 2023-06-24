from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="edf_interface",
    version="0.0.1",
    author="Hyunwoo Ryu",
    author_email="tomato1mule@gmail.com",
    description="Network interface for EDFs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tomato1mule/edf_interface",
    project_urls={
        "Bug Tracker": "https://github.com/tomato1mule/edf_interface/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Ubuntu 22.04",
    ],
    packages=['edf_interface'],
    python_requires="<3.9",
    install_requires=[
        'beartype',
        'Pyro5==5.14',
        'plotly==5.13',
        'pyyaml',
        'dash==2.7.1',
        'dash_vtk==0.0.9',
        'dash_daq==0.5.0',
    ]
)