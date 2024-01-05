from setuptools import setup, find_packages

setup(
    name="givt_pytorch",
    version="0.0.1",
    author="Elio Pascarelli",
    author_email="elio@pascarelli.com",
    description="A partial implementation of Generative Infinite Vocabulary Transformer (GIVT) from Google Deepmind, in PyTorch.",
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
