from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="audio_toolbox",
    version="0.1.0",
    author="AI Audio Toolbox Team",
    author_email="example@example.com",
    description="A comprehensive audio analysis and transformation toolbox",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/audio_toolbox",
    packages=find_packages(where="src"),
    package_dir={"":"src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)