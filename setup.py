from setuptools import setup, find_packages

# Read the requirements from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="interview",
    version="1.0.0",
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
    description="Data science interview questions and answers",
    long_description=open("README.md").read(),
    author="Omid Solari",
    author_email="omid.solari@cerebras.net",
    url="https://github.com/omids-cerebras/interview",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
