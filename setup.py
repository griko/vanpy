from setuptools import setup, find_packages
# import src.vanpy as vanpy

# with open('requirements.txt') as f:
#     required = f.read().splitlines()

setup(
    name='vanpy',
    version='0.76.09',
    #vanpy.__version__,
    description='Voice ANalysis framework',
    author='Gregory Koushnir',
    author_email='koushgre@post.bgu.ac.il',
    package_dir={'': 'src'},
    packages=find_packages('vanpy'),
    # packages=find_packages(include=['vanpy.core', 'vanpy.core.*', 'vanpy.utils', 'vanpy.utils.*']),
    install_requires=[
        'gdown>=4.4.0',
        'huggingface-hub>=0.5.1',
        'pandas>=1.3.2',
        'PyYAML>=6.0',
        'scikit-learn==1.0.2',
        'speechbrain>=0.5.11',
        'transformers>=4.19.2',
        'xgboost>=1.5.0'
    ],
    #install_requires=required,
    setup_requires=['flake8'],
    extras_require={
        'whisper': ['openai-whisper'],
        'ina': ['inaSpeechSegmenter>=0.7.3'],
        'pyannote': ['pyannote.audio>=2.1.1'], # pyannote.audio @ https://github.com/pyannote/pyannote-audio/archive/develop.zip'
        'all': ['openai-whisper', 'inaSpeechSegmenter>=0.7.3', 'pyannote.audio>=2.1.1']
        # 'voice-filter': ['https://github.com/nguyenvulebinh/voice-filter.git']
    },
    zip_safe=True
)
