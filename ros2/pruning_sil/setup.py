from setuptools import find_packages, setup

package_name = "pruning_sil"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/sil_replay.launch.py"]),
        ("share/" + package_name + "/config", ["config/pruning_sil.rviz"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Jose Sanchez",
    maintainer_email="joseszgz2021@gmail.com",
    description="Software-in-the-loop replay of recorded pruning captures.",
    license="BSD-3-Clause",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "servo_node = pruning_sil.servo_node:main",
            "export_rosbag2 = pruning_sil.export_rosbag2:main",
            "parity = pruning_sil.parity:main",
            "replay_player = pruning_sil.replay_player:main",
        ],
    },
)
