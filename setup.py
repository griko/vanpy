from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='voice_characterizer',
    version='0.52',
    description='Voice characterization framework',
    author='Gregory Koushnir',
    author_email='koushgre@post.bgu.ac.il',
    packages=find_packages(include=['audio_pipeline.core', 'audio_pipeline.core.*', 'audio_pipeline.utils', 'audio_pipeline.utils.*']),
    install_requires=[
        'gdown>=4.4.0',
        'huggingface-hub>=0.5.1',
        'inaSpeechSegmenter>=0.7.3',
        'pandas>=1.3.2',
        'pyannote.audio @ https://github.com/pyannote/pyannote-audio/archive/develop.zip',
        'PyYAML>=6.0',
        'speechbrain>=0.5.11',
        'transformers==4.19.2',
        'xgboost==1.5.0'
    ],
    #install_requires=required,
    setup_requires=['flake8'],
    # extras_require={
    #     'interactive': ['matplotlib>=2.2.0', 'jupyter'],
    # }
)
