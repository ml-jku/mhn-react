import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mhnreact",
    version="1.0",
    author="Philipp Seidl and Philipp Renz",
    author_email="ph.seidl92@gmail.com",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ml-jku/mhn-react",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD-2-Clause License",
        "Operating System :: linux-64",
    ],
    python_requires='>=3.7',
)
