from setuptools import setup, find_packages

print('Found packages:', find_packages())
setup(
    description='AniMer as a package',
    name='AniMer',
    packages=find_packages(),
    install_requires=[
        'gdown',
        'numpy',
        'opencv-python',
        'pyrender',
        'pytorch-lightning',
        'scikit-image',
        'smplx==0.1.28',
        'yacs',
        'detectron2 @ git+https://github.com/facebookresearch/detectron2.git',
        'chumpy @ git+https://github.com/mattloper/chumpy.git',
        'mmcv==1.3.9',
        'timm',
        'einops',
        'xtcocotools',
        'pandas',
        'open3d',
        'gradio==5.9.0',
    ],
    extras_require={
        'all': [
            'hydra-core',
            'hydra-submitit-launcher',
            'hydra-colorlog',
            'pyrootutils',
            'rich',
        ],
    },
)
