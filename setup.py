from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='price_microservice',
    version='0.17.0',
    author='Mike Casale',
    author_email='mike.casale@blockchainclimate.org',
    description='Price Microservice is a customized version of EvalML for use with the Blockchain Climate Institute and its price_microservice pipeline.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/BlockchainClimateInstitute/price_microservice',
    install_requires=open('core-requirements.txt').readlines() + open('requirements.txt').readlines()[1:],
    tests_require=open('test-requirements.txt').readlines(),
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        'console_scripts': [
          'evalml = evalml.__main__:cli'
        ]
    },
    data_files=[('evalml/demos/data', ['evalml/demos/data/fraud_transactions.csv.tar.gz', 'evalml/demos/data/churn.csv']),
                ('evalml/tests/data', ['evalml/tests/data/tips.csv', 'evalml/tests/data/titanic.csv'])],
)
