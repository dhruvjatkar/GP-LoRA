import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gplora",
    version="0.1.0",
    author="Dhruv Jatkar",
    author_email="",
    description="GP-LoRA: Gauge-Projected Low-Rank Adaptation. Extends LoRA with gauge-fixing projections to exploit symmetry for accelerated fine-tuning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dhruvjatkar/GP-LoRA",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    python_requires='>=3.6',
    install_requires=[
        "torch>=1.6.0",
    ],
    keywords=[
        "deep learning",
        "fine-tuning",
        "parameter-efficient",
        "lora",
        "gauge symmetry",
        "transformers",
        "nlp",
    ],
)
