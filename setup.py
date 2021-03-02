from setuptools import setup, find_packages

requirements = [
    'yaconfigobject',
]

setup_requirements = [
    'setuptools_scm',
    'pytest-runner',
]

test_requirements = [
    'pytest-cov',
]

extras = {
    'test': test_requirements,
}

packages = find_packages(include=['python_tamer'])

package_dir = {}

package_data = {}

setup(
    name='python-tamer',
    use_scm_version=True,
    author="Todd C. Harris",
    author_email='todd.harris@meteoswiss.ch',
    description="Python library.",
    url='https://gitlab.meteoswiss.ch/htc/python-tamer',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD-3-Clause License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6-3.7-3.8',
    ],
    keywords='python_tamer',
    entry_points={},
    scripts=[],
    license="BSD-3-Clause license",
    long_description=open('README.rst').read() + '\n\n' +
    open('HISTORY.rst').read(),
    include_package_data=True,
    zip_safe=False,
    test_suite='test',
    py_modules=['python-tamer'],
    packages=packages,
    install_requires=requirements,
    package_dir=package_dir,
    package_data=package_data,
    setup_requires=setup_requirements,
    tests_require=test_requirements,
    extras_require=extras,
)
