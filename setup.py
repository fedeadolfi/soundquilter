from setuptools import setup, find_packages

setup(
    name="sound_quilter",
    version="0.1",
    author="Federico Adolfi",
    author_email="fedeadolfi@gmail.com",
    description="Generalization and implementation of sound quilting algorithms",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "soundfile",
        ],
    )

