from setuptools import setup, find_packages #type: ignore

with open("requirements.txt") as f:
    requirements = f.read().splitlines() 
setup(
    name="medical_assistant",
    packages=find_packages(),
    install_requires=requirements,
)