from setuptools import setup,find_packages


setup(
    name = "vector_cuda",
    version = "0.1.0",
    author = "Sahil Rajwar",
    description = "Vector lib with strong GPU support using CUDA",
    long_description = open("README.md","r",encoding = "utf-8").read(),
    long_description_content_type = "text/markdown",
    license = "MIT",
    install_requires = ["numpy","cupy-cuda12x"],
    url = "https://github.com/Sahil-Rajwar-2004/vector.py-cuda",
    packages = find_packages(),
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires = ">=3.6",
)


