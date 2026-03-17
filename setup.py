from setuptools import find_packages, setup

package_name = 'collision_ai'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['resource/best_collision_model.h5']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shivanshmishra',
    maintainer_email='shivanshm413@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': ['fake_camera_node = collision_ai.fake_camera_node:main',
        'collision_ai_node = collision_ai.collision_ai_node:main',
        ],
    },
)
