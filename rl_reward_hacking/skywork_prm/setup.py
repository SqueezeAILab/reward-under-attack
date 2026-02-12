from setuptools import setup

setup(
    name='skywork_prm',
    version='0.1',
    packages=['.'],
    entry_points={
        'vllm.general_plugins': [
            "register_prm_model = prm_model:register"
        ]
    }
)