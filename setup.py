from setuptools import setup, find_packages

setup(
    name="givt_pytorch",
    version="0.0.3",
    author="Elio Pascarelli",
    author_email="elio@pascarelli.com",
    description="A partial implementation of Generative Infinite Vocabulary Transformer (GIVT) from Google Deepmind, in PyTorch.",
    long_description=open('README.md').read(),
    long_description_content_type = 'text/markdown',
    url="https://github.com/elyxlz/givt-pytorch",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "torch",
        "python-dotenv",
        "einops",
        "ema-pytorch>=0.3.2",
    ],
    extras_require={"train": ["accelerate", "wandb", "tqdm"]},
    license="MIT",
)
