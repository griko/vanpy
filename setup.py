from setuptools import setup, find_packages
import vanpy
# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

whisper = ['openai-whisper==20230314']
ina = ['inaSpeechSegmenter==0.7.3']
pyannote = ['pyannote.audio==2.1.1', 'soundfile==0.10.3.post1']
yamnet = ['tensorflow==2.8.0', 'tensorflow-hub==0.13.0']
wav2vec2 = ['transformers==4.19.2']
speech_brain_iemocap_emotion = ['speechbrain==0.5.13']


# with open('requirements.txt') as f:
#     required = f.read().splitlines()

setup(
    name='vanpy',
    version=vanpy.__version__,
    description='VANPY - Voice Analysis framework in Python',
    author='Gregory Koushnir',
    author_email='koushgre@post.bgu.ac.il',
    long_description=long_description,
    long_description_content_type='text/markdown',
    package_dir={'': 'src'},
    packages=find_packages('vanpy'),
    # packages=find_packages(include=['vanpy.core', 'vanpy.core.*', 'vanpy.utils', 'vanpy.utils.*']),

    install_requires=[
        'gdown==4.4.0',
        'huggingface-hub==0.8.1',
        'keras==2.8.0',
        'librosa==0.9.1',
        'pandas==1.3.5',
        'pydub==0.25.1',
        'PyYAML==6.0',
        'scikit-learn>=1.0.2',
        'speechbrain==0.5.13',
        'torch==1.13.1',
        'torchaudio==0.13.1'
    ],
    dependency_links=[
        'https://pypi.anaconda.org/scipy-wheels-nightly/simple/scikit-learn/'
    ],
    #install_requires=required,
    setup_requires=['flake8'],
    extras_require={
        'whisper': whisper,
        'ina': ina,
        'pyannote': pyannote,
        'yamnet': yamnet,
        'wav2vec2': wav2vec2,
        'speech_brain_iemocap_emotion': speech_brain_iemocap_emotion + wav2vec2,
        'all': whisper + ina + pyannote + yamnet + wav2vec2 + speech_brain_iemocap_emotion
    },
    zip_safe=True
)
