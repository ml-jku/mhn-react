import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="temprel",
    version="1.0",
    author="Mike Fortunato",
    author_email="mef231@gmail.com",
    description="Reaction template relevance training pipeline code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/mefortunato/template-relevance/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)