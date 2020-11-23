import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyCFOFiSAX",
    version="0.1.0",
    author="Lucas Foulon",
    author_email="lucas.foulon@gmail.com",
    description="Calculate CFOF score on stream context with iSAX trees",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.liris.cnrs.fr/lfoulon/icfof",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
