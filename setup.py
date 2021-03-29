from setuptools import setup, find_packages

requirements = [
    'pandas',
    'netcdf4',
    'numpy',
    'cartopy',
    'matplotlib',
]

setup_requirements = [
    'pytest-runner',
]

test_requirements = [
    'pytest-cov',
]

extras = {
    'test': test_requirements,
}

packages = find_packages(include=['python_tamer'],exclude=['test','doc'])

package_dir = {'python-TAMER': 'python_tamer'}

package_data = {'test': ["UV_test_data_2018.nc"]}

setup(
    name='python-TAMER',
    version="0.3.1",
    author="Todd C. Harris",
    author_email='todd.harris@meteoswiss.ch',
    description="Toolkit for Analysis and Maps of Exposure Risk",
    url='https://github.com/tch521/python-TAMER',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
    keywords='python-TAMER UV',
    entry_points={},
    scripts=[],
    license="BSD-3-Clause license",
    long_description=open('README.rst').read() + '\n\n' +
    open('HISTORY.rst').read(),
    include_package_data=True,
    zip_safe=False,
    test_suite='test',
    packages=packages,
    install_requires=requirements,
    package_dir=package_dir,
    package_data=package_data,
    setup_requires=setup_requirements,
    tests_require=test_requirements,
    extras_require=extras,
)
