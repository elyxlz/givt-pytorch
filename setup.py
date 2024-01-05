from setuptools import setup, find_packages

setup(
    name="xxx",
    version="0.0.1",
    author="xxx",
    author_email="xxx",
    description="xxx",
    url="",
    classifiers=[
        "Framework :: Pytorch",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "torch",
        "python-dotenv",
    ],
    extras_require={"train": ["accelerate", "wandb", "tqdm"]},
    license="MIT",
)
