"""
Setup script for Chimera: Neuroplastic Autonomous Red-Team Organism
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = requirements_path.read_text().strip().split('\n')
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

setup(
    name="chimera-redteam",
    version="1.0.0",
    description="Neuroplastic Autonomous Red-Team Organism for Ethical Security Testing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Chimera Development Team",
    author_email="dev@chimera-redteam.org",
    url="https://github.com/your-org/chimera",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "chimera": [
            "reporter/templates/*.md",
            "reporter/templates/*.html",
        ]
    },
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "gpu": [
            "torch>=2.0.0",
            "transformers[torch]>=4.30.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "chimera=main:cli",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Information Technology",
        "Topic :: Security",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Environment :: Console",
    ],
    python_requires=">=3.8",
    keywords="security redteam pentesting bugbounty automation ai neuroplastic",
    project_urls={
        "Bug Reports": "https://github.com/your-org/chimera/issues",
        "Source": "https://github.com/your-org/chimera",
        "Documentation": "https://chimera-redteam.readthedocs.io/",
    },
)