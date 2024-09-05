from setuptools import setup, find_packages

setup(
    name='project_modules',  # This is the name of your package
    version='0.1',  # Version of your package
    packages=find_packages(),  # Automatically finds all packages in your directory
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'streamlit'
    ],
)

