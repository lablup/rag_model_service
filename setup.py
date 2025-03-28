from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f.readlines() if line.strip()]

setup(
    name="rag_model_service",
    version="0.1.0",
    description="A RAG (Retrieval-Augmented Generation) service for document search and generation",
    author="Lablup",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=requirements + [
        "python-dotenv",
    ],
    entry_points={
        "console_scripts": [],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
