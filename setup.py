from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f.readlines() if line.strip()]

setup(
    name="dosirag",
    version="0.1.0",
    description="DosiRAG is a service that creates specialized documentation assistants from GitHub repositories using Backend.AI, Model Service and Backend.AI CLI Client",
    author="Sergey Leksikov @Lablup",
    author_email="lexikovs@lablup.com",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.12",
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
        "Programming Language :: Python :: 3.12",
    ],
)
