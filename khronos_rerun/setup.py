from setuptools import find_packages, setup
import os
from glob import glob

package_name = "khronos_rerun"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Juell Sprott",
    maintainer_email="juell.sprott@student.uva.nl",
    description="Rerun visualization bridge for Khronos ROS2 topics",
    license="MIT",
    entry_points={
        "console_scripts": [
            "rerun_bridge = khronos_rerun.rerun_bridge:main",
        ],
    },
)
