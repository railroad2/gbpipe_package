from setuptools import setup, find_packages

setup_requires = []

install_requires = []

dependency_links = []

setup(
    name = 'gbpipe',
    version = '0.0',
    description = 'Package for GroundBIRD pipeline',
    author = 'Kyungmin Lee',
    author_email = 'kmlee@hep.korea.ac.kr',
    packages = find_packages(),
    include_package_data = True,
    install_requires = install_requires,
    dependency_links = dependency_links,
    entry_points = {},
)

