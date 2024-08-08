from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="xai4chem",
    version="0.0.1",
    author="Hellen Namulinda",
    author_email="hellennamulinda@gmail.com",
    description="Explainable AI for Chemistry", 
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ersilia-os/xai4chem",
    license="GPLv3",
    python_requires=">=3.10",
    install_requires=install_requires,
    packages=find_packages(exclude=("utilities")),
    py_modules=['cmd'],
    entry_points={'console_scripts': ['xai4chem = xai4chem.cli:cli',],
    },
    classifiers=[  
        "Programming Language :: Python :: 3.10", 
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ], 
    keywords="xai, chemistry, machine-learning, drug-discovery",
    project_urls={
        "Documentation": "",
        "Source Code": "https://github.com/ersilia-os/xai4chem",
    },
)