from setuptools import setup, find_packages

config = {
    'description': 'Automated Modelling in Biological Evidence-based Research',
    'download_url': 'https://github.com/zhanglab-aim/AMBER2',
    'version': '2.0.0b0',
    'packages': find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    'include_package_data': True,
    'setup_requires': [],
    'install_requires': [
        'numpy',
        'pandas',
        'tqdm',
        'expecttest',
        'packaging',
        ],
    'dependency_links': [],
    'name': 'amber-automl',
}

if __name__== '__main__':
    setup(**config)
