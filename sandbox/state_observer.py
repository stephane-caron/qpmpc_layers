#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Inria

import numpy as np
import pink
import pinocchio as pin
import upkie_description
from numpy.typing import NDArray
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask
from pink.visualization import start_meshcat_visualizer
from upkie.utils.rotations import rotation_matrix_from_quaternion


class StateObserver:
    def __init__(
        self,
        wheel_radius: float,
        visualize: bool,
        solver: str,
    ):
        """Initialize observer.

        Args:
            wheel_radius Wheel radius in [m].
            visualize: If set, start a MeshCat visualization.
            solver: QP solver to solve the underlying optimization.
        """
        imu_task = FrameTask(
            "imu",
            position_cost=0.0,  # [cost] / [m]
            orientation_cost=1.0,  # [cost] / [rad]
        )
        left_anchor_task = FrameTask(
            "left_anchor",
            position_cost=1.0,  # [cost] / [m]
            orientation_cost=0.0,  # [cost] / [rad]
        )
        right_anchor_task = FrameTask(
            "right_anchor",
            position_cost=1.0,  # [cost] / [m]
            orientation_cost=0.0,  # [cost] / [rad]
        )
        posture_task = PostureTask(
            cost=1.0,  # [cost] / [rad]
        )
        robot = upkie_description.load_in_pinocchio(
            root_joint=pin.JointModelFreeFlyer()
        )
        visualizer = start_meshcat_visualizer(robot) if visualize else None

        q_0 = pin.neutral(robot.model)
        configuration = pink.Configuration(robot.model, robot.data, q_0)
        posture_task.set_target_from_configuration(configuration)
        left_anchor_task.set_target_from_configuration(configuration)
        right_anchor_task.set_target_from_configuration(configuration)
        left_anchor_task.transform_target_to_world.translation[2] = 0.0
        right_anchor_task.transform_target_to_world.translation[2] = 0.0

        self.anchor_tasks = {
            "left": left_anchor_task,
            "right": right_anchor_task,
        }
        self.configuration = configuration
        self.ground_position = {"left": 0.0, "right": 0.0}
        self.imu_task = imu_task
        self.model = robot.model
        self.posture_task = posture_task
        self.solver = solver
        self.tasks = (
            imu_task,
            left_anchor_task,
            posture_task,
            right_anchor_task,
        )
        self.velocity = np.zeros(robot.model.nv)
        self.visualizer = visualizer
        self.wheel_radius = wheel_radius

    def reset(self):
        self.ground_position["left"] = 0.0
        self.ground_position["right"] = 0.0

    @property
    def q(self) -> NDArray[float]:
        return self.configuration.q

    @property
    def v(self) -> NDArray[float]:
        return self.velocity

    def __update_anchor_tasks(self, observation: dict, dt: float) -> None:
        for side in ("left", "right"):
            wheel_velocity = observation["servo"][f"{side}_wheel"]["velocity"]
            sign = 1.0 if side == "left" else -1.0
            ground_velocity = sign * self.wheel_radius * wheel_velocity
            self.ground_position[side] += ground_velocity * dt
            anchor_task = self.anchor_tasks[side]
            anchor_task.transform_target_to_world.translation[0] = (
                self.ground_position[side]
            )

    def __update_imu_task(self, observation: dict) -> None:
        quat_imu_to_ars = observation["imu"]["orientation"]
        rotation_imu_to_ars = rotation_matrix_from_quaternion(quat_imu_to_ars)
        # The attitude reference system frame has +x forward, +y right and +z
        # down, whereas our world frame has +x forward, +y left and +z up:
        # https://github.com/mjbots/pi3hat/blob/ab632c82bd501b9fcb6f8200df0551989292b7a1/docs/reference.md#orientation
        rotation_ars_to_world = np.diag([1.0, -1.0, -1.0])
        rotation_imu_to_world = rotation_ars_to_world @ rotation_imu_to_ars
        self.imu_task.transform_target_to_world = pin.SE3(
            rotation=rotation_imu_to_world,
            translation=np.zeros(3),
        )

    def __update_posture_task(self, observation: dict) -> None:
        q_posture = self.posture_task.target_q
        q_posture[7] = observation["servo"]["left_hip"]["position"]
        q_posture[8] = observation["servo"]["left_knee"]["position"]
        q_posture[9] = observation["servo"]["left_wheel"]["position"]
        q_posture[10] = observation["servo"]["right_hip"]["position"]
        q_posture[11] = observation["servo"]["right_knee"]["position"]
        q_posture[12] = observation["servo"]["right_wheel"]["position"]

    def cycle(self, observation: dict, dt: float) -> None:
        self.__update_anchor_tasks(observation, dt)
        self.__update_imu_task(observation)
        self.__update_posture_task(observation)
        velocity = solve_ik(
            self.configuration,
            self.tasks,
            dt,
            solver=self.solver,
        )
        self.configuration.integrate_inplace(velocity, dt)
        self.velocity = velocity
        if self.visualizer is not None:
            self.visualizer.display(self.configuration.q)
