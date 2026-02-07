#!/usr/bin/env python3

import time
from enum import Enum

import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool

from nemo_interfaces.msg import BalloonCoOrd
from nemo_interfaces.msg import RovCommands


class State(Enum):
    SEARCH = 0
    ALIGN_SWAY = 1
    ALIGN_HEAVE = 2
    SURGE = 3
    ADVANCE_SLIGHTLY = 4
    DONE = 5


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class BalloonStateMachine(Node):
    """
    Subscribes:  /nemo_auv/balloon_coord   (BalloonCoOrd)

    Publishes:   /nemo_auv/input_cmd       (RovCommands)
                /nemo_auv/arm             (Bool)
    """

    def __init__(self):
        super().__init__("balloon_state_machine")

        # ---- Subscribers / Publishers ----
        self.sub = self.create_subscription(
            BalloonCoOrd, "/nemo_auv/balloon_coord", self.balloon_cb, 10
        )

        # Publish custom command type instead of Twist
        self.cmd_pub = self.create_publisher(
            RovCommands, "/nemo_auv/input_cmd", 10
        )

        self.arm_pub = self.create_publisher(
            Bool, "/nemo_auv/arm", 10
        )

        # ---- Parameters (tune these) ----
        self.declare_parameter("dx_tol", 50)
        self.declare_parameter("dy_tol", 50)

        self.declare_parameter("sway_speed", 0.45)
        self.declare_parameter("heave_speed", 0.5)
        self.declare_parameter("surge_speed", 0.35)

        self.declare_parameter("area_close_enough", 200000)
        self.declare_parameter("lost_timeout_s", 0.6)

        self.declare_parameter("advance_time_s", 0.6)
        self.declare_parameter("advance_speed", 0.25)

        self.declare_parameter("align_timeout_s", 6.0)
        self.declare_parameter("surge_timeout_s", 5.0)

        # ---- Internal state ----
        self.state = State.SEARCH
        self.last_msg = None
        self.last_msg_time = 0.0
        self.state_enter_time = time.time()

        # periodic tick
        self.timer = self.create_timer(0.05, self.tick)  # 20 Hz

        # arm once at startup (optional)
        self.arm_once()

        self.get_logger().info("Balloon state machine started (SEARCH)")

    def arm_once(self):
        msg = Bool()
        msg.data = True
        self.arm_pub.publish(msg)

    def balloon_cb(self, msg: BalloonCoOrd):
        self.last_msg = msg
        self.last_msg_time = time.time()

    def set_state(self, new_state: State):
        if new_state != self.state:
            self.state = new_state
            self.state_enter_time = time.time()
            self.get_logger().info(f"STATE -> {self.state.name}")

    # -------------------- NEW: Publish RovCommands --------------------
    def publish_cmd(self, surge: float = 0.0, sway: float = 0.0, heave: float = 0.0, yaw: float = 0.0):
        cmd = RovCommands()
        cmd.surge = float(surge)
        cmd.sway = float(sway)
        cmd.yaw = float(yaw)
        cmd.heave = float(heave)
        self.cmd_pub.publish(cmd)

    def stop(self):
        self.publish_cmd(0.0, 0.0, 0.0, 0.0)

    def have_fresh_detection(self) -> bool:
        if self.last_msg is None:
            return False
        return (time.time() - self.last_msg_time) <= float(self.get_parameter("lost_timeout_s").value)

    def tick(self):
        now = time.time()

        # If detection is stale, treat as lost (and stop motion if needed)
        fresh = self.have_fresh_detection()

        dx_tol = float(self.get_parameter("dx_tol").value)
        dy_tol = float(self.get_parameter("dy_tol").value)

        sway_speed = float(self.get_parameter("sway_speed").value)
        heave_speed = float(self.get_parameter("heave_speed").value)
        surge_speed = float(self.get_parameter("surge_speed").value)

        area_close = float(self.get_parameter("area_close_enough").value)

        align_timeout = float(self.get_parameter("align_timeout_s").value)
        surge_timeout = float(self.get_parameter("surge_timeout_s").value)

        # ---------------- STATE: SEARCH ----------------
        if self.state == State.SEARCH:
            # Hold position (or add a slow yaw scan here)
            self.stop()

            if fresh:
                self.set_state(State.ALIGN_SWAY)
            return

        # For all other states, if we lose detection -> go back to SEARCH safely
        if not fresh:
            self.stop()
            self.set_state(State.SEARCH)
            return

        # from here, we have last_msg
        msg = self.last_msg
        dx = float(msg.dx)
        dy = float(msg.dy)
        area = float(msg.contour_area)

        centered_x = abs(dx) <= dx_tol
        centered_y = abs(dy) <= dy_tol

        # ---------------- STATE: ALIGN_SWAY ----------------
        if self.state == State.ALIGN_SWAY:
            if (now - self.state_enter_time) > align_timeout:
                self.stop()
                self.set_state(State.SEARCH)
                return

            if centered_x:
                self.stop()
                self.set_state(State.ALIGN_HEAVE)
                return

            sway_cmd = clamp((dx / 200.0) * sway_speed, -sway_speed, sway_speed)
            self.publish_cmd(surge=0.0, sway=sway_cmd, heave=0.0, yaw=0.0)
            return

        # ---------------- STATE: ALIGN_HEAVE ----------------
        if self.state == State.ALIGN_HEAVE:
            if (now - self.state_enter_time) > align_timeout:
                self.stop()
                self.set_state(State.SEARCH)
                return

            if centered_y:
                self.stop()
                self.set_state(State.SURGE)
                return

            heave_cmd = clamp((dy / 200.0) * heave_speed, -heave_speed, heave_speed)
            self.publish_cmd(surge=0.0, sway=0.0, heave=heave_cmd, yaw=0.0)
            return

        # ---------------- STATE: SURGE ----------------
        if self.state == State.SURGE:
            if not centered_x:
                self.stop()
                self.set_state(State.ALIGN_SWAY)
                return
            if not centered_y:
                self.stop()
                self.set_state(State.ALIGN_HEAVE)
                return

            if area >= area_close:
                self.stop()
                self.set_state(State.ADVANCE_SLIGHTLY)
                return

            if (now - self.state_enter_time) > surge_timeout:
                self.stop()
                self.set_state(State.ADVANCE_SLIGHTLY)
                return

            self.publish_cmd(surge=surge_speed, sway=0.0, heave=0.0, yaw=0.0)
            return

        # ---------------- STATE: ADVANCE_SLIGHTLY ----------------
        if self.state == State.ADVANCE_SLIGHTLY:
            adv_t = float(self.get_parameter("advance_time_s").value)
            adv_speed = float(self.get_parameter("advance_speed").value)

            if (now - self.state_enter_time) < adv_t:
                self.publish_cmd(surge=adv_speed, sway=0.0, heave=0.0, yaw=0.0)
                return

            self.stop()
            self.set_state(State.DONE)
            return

        # ---------------- STATE: DONE ----------------
        if self.state == State.DONE:
            self.stop()
            return


def main():
    rclpy.init()
    node = BalloonStateMachine()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.stop()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
