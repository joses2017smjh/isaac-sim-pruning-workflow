#!/usr/bin/env python3
from isaaclab_sensor_learning import CFG_DIR
from isaaclab_sensor_learning.sensor.yaml_to_cfg import load_sensor_yaml, rig_dict_to_sensor_cfgs
from isaaclab_sensor_learning.sensor.spherical_sensor_layout_generator import LloydSphereSensorLayout
from isaaclab_sensor_learning.sensor.planar_sensor_layout_generator import PlaneSensorLayout
import isaaclab_sensor_learning.sensor.camera_utils as camera_utils
import isaaclab_sensor_learning.sensor.lidar_utils as lidar_utils
import isaaclab_sensor_learning.sensor.rig_utils as rig_utils
import isaaclab_sensor_learning.utils.quaternion_utils as qutils


class RigGenerator:
    def __init__(self, sensor_cfgs: list[dict]):
        self._sensor_cfgs = sensor_cfgs
        self._sensor_groups = self.group_sensors_by_type()
        print(f"[INFO]: Sensor groups: {self._sensor_groups}")
        # self.lloyd_layout_generator: LloydSphereSensorLayout
        # self.planar_layout_generator: PlaneSensorLayout
        return
    
    def group_sensors_by_type(self) -> dict[str, list[str]]:
        """Group sensors by type. Return a dictionary of sensor type to list of sensor names."""
        sensor_groups = {
            "camera": [],
            "lidar": [],
            "tof": [],
        }
        for sensor_cfg in self._sensor_cfgs:
            sensor_name = sensor_cfg["name"]
            sensor_type = sensor_cfg["sensor_type"]
            if sensor_type not in sensor_groups:
                raise ValueError(f"Unsupported sensor type: {sensor_type}")
            sensor_groups[sensor_type].append(sensor_name)
        return sensor_groups

    def generate_rig(self, layout_type: str) -> dict:
        """Generate a rig based on sensor configs. Small form-factor sensors (i.e. tofs) are arranged together. All sensors must adhere to allowed sensor positions"""



        # # get plane/sphere layout
        # if layout_type == "sphere":
        #     self.lloyd_layout_generator = LloydSphereSensorLayout(sensor_cfgs=self._sensor_cfgs)
        #     sensor_layout = self.lloyd_layout_generator.generate_sensor_layout()
        # elif layout_type == "plane":
        #     self.planar_layout_generator = PlaneSensorLayout(sensor_cfgs=self._sensor_cfgs)
        #     sensor_layout = self.planar_layout_generator.generate_sensor_layout()
        # else:
        #     raise ValueError(f"Invalid rig type {layout_type}. Must be 'sphere' or 'plane'.")

        # update sensor prims and offsets based on layout
        for sensor_name, sensor_cfg in self._sensor_cfgs.items():

            print(sensor_cfg)

        return {}


def main():
    rig_yaml_path = CFG_DIR / "rigs" / "test_rig0.yaml"

    sensor_cfgs = load_sensor_yaml(rig_yaml_path)
    rig_generator = RigGenerator(sensor_cfgs=sensor_cfgs)
    rig_cfg = rig_generator.generate_rig(layout_type="sphere")
    

    return


if __name__ == "__main__":
    main()