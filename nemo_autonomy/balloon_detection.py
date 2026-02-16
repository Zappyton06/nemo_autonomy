#!/usr/bin/env python3
import time
from enum import Enum

import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool
from geometry_msgs.msg import Pose

from nemo_interfaces.msg import BalloonCoOrd
from nemo_interfaces.msg import RovCommands

FT_TO_M = 0.3048


class State(Enum):
    DIVE_INIT = 0
    SEARCH = 1
    ALIGN_SWAY = 2
    ALIGN_HEAVE = 3
    SURGE = 4
    ADVANCE_SLIGHTLY = 5
    SURFACE = 6
    POPPED = 7
    DONE = 8


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class BalloonStateMachine(Node):
    def __init__(self):
        super().__init__("balloon_state_machine")

        # ---------------- I/O ----------------
        self.sub = self.create_subscription(BalloonCoOrd, "/nemo_auv/balloon_coord", self.balloon_cb, 10)
        self.pose_sub = self.create_subscription(Pose, "/nemo_auv/pose", self.pose_cb, 50)
        self.cmd_pub = self.create_publisher(RovCommands, "/nemo_auv/input_cmd", 10)
        self.arm_pub = self.create_publisher(Bool, "/nemo_auv/arm", 10)
        self.depth_lock_pub = self.create_publisher(Bool, "/nemo_auv/depth_lock", 10)

        # ---------------- Task parameters ----------------
        self.declare_parameter("dx_tol", 50.0)
        self.declare_parameter("dy_tol", 50.0)
        self.declare_parameter("sway_speed", 0.25)
        self.declare_parameter("heave_speed", 0.25)
        self.declare_parameter("surge_speed", 0.25)
        self.declare_parameter("area_close_enough", 900000.00)
        self.declare_parameter("lost_timeout_s", 0.6)
        self.declare_parameter("advance_time_s", 0.6)
        self.declare_parameter("advance_speed", 0.25)
        self.declare_parameter("align_timeout_s", 6.0)
        self.declare_parameter("surge_timeout_s", 5.0)
        # SEARCH behavior
        self.declare_parameter("search_yaw_speed", 0.15)     # yaw effort/rate (whatever your controller expects)
        self.declare_parameter("search_flip_period_s", 30.0)  # flip yaw direction every N seconds
        # Optional stability gate befor SURGE
        self.declare_parameter("require_center_stable", True)
        self.declare_parameter("center_stable_time_s", 0.30)
        # ---------------- Dive/surface parameters ----------------
        self.declare_parameter("dive_min_depth_m", 0.5)
        self.declare_parameter("dive_max_depth_m", 1.5)
        self.declare_parameter("dive_target_depth_m", 1.0)
        # magnitude only (sign auto-detected)
        self.declare_parameter("dive_heave_mag", 1.0)
        self.declare_parameter("surface_heave_mag", 1.0)
        self.declare_parameter("depth_reach_tol_m", 0.05)
        self.declare_parameter("dive_timeout_s", 15.0)
        # Make surfacing NOT take long
        self.declare_parameter("surface_depth_done_m", 0.10)   # consider “surfaced” when depth <= this
        self.declare_parameter("surface_timeout_s", 6.0)       # cap how long we try to surface
        self.declare_parameter("enable_depth_lock_after_dive", True)
        self.declare_parameter("allow_heave_alignment_when_depth_locked", False)
        # ---------------- Sign detection ----------------
        self.declare_parameter("depth_is_negative_down", True)   # True if z gets more negative when going down
        self.declare_parameter("heave_sign_test_mag", 0.4)
        self.declare_parameter("heave_sign_test_time_s", 0.6)
        self.declare_parameter("heave_sign_min_delta_m", 0.01)

        # ---------------- State ----------------
        self.state = State.DIVE_INIT
        self.state_enter_time = time.time()
        self.last_msg = None
        self.last_msg_time = 0.0
        self.z_raw = 0.0
        self.current_depth = 0.0
        self.have_pose = False
        # sign handling
        self.depth_is_negative_down = bool(self.get_parameter("depth_is_negative_down").value)
        self.heave_down_sign = None  # +1 or -1 once detected
        # sign-test bookkeeping
        self._sign_test_phase = 0
        self._sign_test_start_time = 0.0
        self._sign_test_start_depth = 0.0
        # search sweep bookkeeping
        self._search_dir = +1.0
        self._search_flip_t0 = time.time()
        # center stability bookkeeping
        self._centered_since = None
        self._was_close = False

        # 20 Hz tick
        self.timer = self.create_timer(0.05, self.tick)

        self.arm_once()
        self.publish_depth_lock(False)
        self.get_logger().info("Balloon state machine started (DIVE_INIT)")

    # ---------------- Basic helpers ----------------
    def arm_once(self):
        msg = Bool()
        msg.data = True
        self.arm_pub.publish(msg)

    def pose_cb(self, msg: Pose):
        self.z_raw = float(msg.position.z)
        # Convert z -> "depth positive down"
        self.current_depth = (-self.z_raw) if self.depth_is_negative_down else self.z_raw
        self.have_pose = True

    def balloon_cb(self, msg: BalloonCoOrd):
        self.last_msg = msg
        self.last_msg_time = time.time()

    def set_state(self, new_state: State):
        if new_state != self.state:
            self.state = new_state
            self.state_enter_time = time.time()
            self._centered_since = None
            self.get_logger().info(f"STATE -> {self.state.name}")

            if new_state == State.SEARCH:
                self._search_dir = +1.0
                self._search_flip_t0 = time.time()

    def publish_depth_lock(self, enabled: bool):
        msg = Bool()
        msg.data = bool(enabled)
        self.depth_lock_pub.publish(msg)

    def publish_cmd(self, surge=0.0, sway=0.0, heave=0.0, yaw=0.0):
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

    # ---------------- Heave sign detection ----------------
    def _update_heave_sign_detection(self, now: float) -> bool:
        """
        Determine which heave sign makes depth increase (down).
        Applies a small +heave pulse and measures delta depth.
        """
        if self.heave_down_sign is not None:
            return True

        test_mag = float(self.get_parameter("heave_sign_test_mag").value)
        test_time = float(self.get_parameter("heave_sign_test_time_s").value)
        min_delta = float(self.get_parameter("heave_sign_min_delta_m").value)

        if self._sign_test_phase == 0:
            self._sign_test_phase = 1
            self._sign_test_start_time = now
            self._sign_test_start_depth = self.current_depth
            self.get_logger().info("Heave sign test: applying +heave pulse...")
            self.publish_cmd(0.0, 0.0, heave=+test_mag, yaw=0.0)
            return False

        if self._sign_test_phase == 1:
            if (now - self._sign_test_start_time) < test_time:
                self.publish_cmd(0.0, 0.0, heave=+test_mag, yaw=0.0)
                return False

            self.stop()
            delta = self.current_depth - self._sign_test_start_depth
            if abs(delta) < min_delta:
                self.get_logger().warn(
                    f"Heave sign test: Δdepth too small (Δ={delta:.4f} m). Retrying..."
                )
                self._sign_test_phase = 0
                return False

            self.heave_down_sign = +1.0 if delta > 0.0 else -1.0
            self.get_logger().info(
                f"Heave sign test: Δdepth={delta:.3f} m => "
                f"{'+heave goes DOWN' if self.heave_down_sign > 0 else '+heave goes UP'}"
            )
            return True

        return False

    # ---------------- Center stability gate ----------------
    def _center_stable_ok(self, now: float, centered: bool) -> bool:
        if not bool(self.get_parameter("require_center_stable").value):
            return centered

        stable_t = float(self.get_parameter("center_stable_time_s").value)
        if not centered:
            self._centered_since = None
            return False

        if self._centered_since is None:
            self._centered_since = now
            return False

        return (now - self._centered_since) >= stable_t

    # ---------------- Main tick ----------------
    def tick(self):
        now = time.time()

        # ---------------- DIVE_INIT ----------------
        if self.state == State.DIVE_INIT:
            if not self.have_pose:
                self.stop()
                return

            if not self._update_heave_sign_detection(now):
                self.publish_depth_lock(False)
                return

            dive_target = float(self.get_parameter("dive_target_depth_m").value)
            dive_min = float(self.get_parameter("dive_min_depth_m").value)
            dive_max = float(self.get_parameter("dive_max_depth_m").value)
            tol = float(self.get_parameter("depth_reach_tol_m").value)

            dive_mag = float(self.get_parameter("dive_heave_mag").value)
            dive_timeout = float(self.get_parameter("dive_timeout_s").value)

            self.publish_depth_lock(False)

            lo = min(dive_min, dive_max)
            hi = max(dive_min, dive_max)
            in_band = (lo <= self.current_depth <= hi)
            close_to_target = abs(self.current_depth - dive_target) <= tol

            if in_band and close_to_target:
                if bool(self.get_parameter("enable_depth_lock_after_dive").value):
                    self.publish_depth_lock(True)
                self.stop()
                self.set_state(State.SEARCH)
                return

            if (now - self.state_enter_time) > dive_timeout:
                if bool(self.get_parameter("enable_depth_lock_after_dive").value):
                    self.publish_depth_lock(True)
                self.stop()
                self.get_logger().warn("Dive timeout reached; proceeding to SEARCH anyway.")
                self.set_state(State.SEARCH)
                return

            heave_cmd = float(self.heave_down_sign) * dive_mag
            self.publish_cmd(surge=0.0, sway=0.0, heave=heave_cmd, yaw=0.0)
            return

        fresh = self.have_fresh_detection()
        dx_tol = float(self.get_parameter("dx_tol").value)
        dy_tol = float(self.get_parameter("dy_tol").value)
        sway_speed = float(self.get_parameter("sway_speed").value)
        heave_speed = float(self.get_parameter("heave_speed").value)
        surge_speed = float(self.get_parameter("surge_speed").value)
        area_close = float(self.get_parameter("area_close_enough").value)
        align_timeout = float(self.get_parameter("align_timeout_s").value)
        surge_timeout = float(self.get_parameter("surge_timeout_s").value)
        allow_heave_align = bool(self.get_parameter("allow_heave_alignment_when_depth_locked").value)

        if self.state == State.SEARCH:
            if bool(self.get_parameter("enable_depth_lock_after_dive").value):
                self.publish_depth_lock(True)

            if fresh:
                self.stop()
                self.set_state(State.ALIGN_SWAY)
                return

            flip_period = float(self.get_parameter("search_flip_period_s").value)
            if (now - self._search_flip_t0) >= max(0.5, flip_period):
                self._search_dir *= -1.0
                self._search_flip_t0 = now

            yaw_speed = float(self.get_parameter("search_yaw_speed").value) * self._search_dir
            self.publish_cmd(surge=0.0, sway=0.0, heave=0.0, yaw=yaw_speed)
            return

        # If we lose detection during alignment/approach, go back to SEARCH
        if self.state in (State.ALIGN_SWAY, State.ALIGN_HEAVE, State.SURGE, State.ADVANCE_SLIGHTLY) and not fresh:
            self.stop()
            self.set_state(State.SEARCH)
            return

        # ---------------- SURFACE (fast exit) ----------------
        if self.state == State.SURFACE:
            if not self.have_pose or self.heave_down_sign is None:
                self.stop()
                return

            self.publish_depth_lock(False)

            surf_done = float(self.get_parameter("surface_depth_done_m").value)
            surface_timeout = float(self.get_parameter("surface_timeout_s").value)
            surf_mag = float(self.get_parameter("surface_heave_mag").value)

            # Done surfacing quickly
            if self.current_depth <= surf_done:
                self.stop()
                self.set_state(State.DONE)
                return

            # Do NOT spend long surfacing; stop and finish
            if (now - self.state_enter_time) > surface_timeout:
                self.stop()
                self.get_logger().warn("Surface timeout reached; stopping (DONE) to avoid lingering.")
                self.set_state(State.DONE)
                return

            # Up is opposite of down
            heave_cmd = (-float(self.heave_down_sign)) * surf_mag
            self.publish_cmd(surge=0.0, sway=0.0, heave=heave_cmd, yaw=0.0)
            return

        # ---------------- DONE ----------------
        if self.state == State.DONE:
            self.stop()
            self.publish_depth_lock(False)
            return

        # From here onward, we expect a message (fresh path) except safety fallback
        if self.last_msg is None:
            self.stop()
            self.set_state(State.SEARCH)
            return

        msg = self.last_msg
        dx = float(msg.dx)
        dy = float(msg.dy)
        area = float(msg.contour_area)

        centered_x = abs(dx) <= dx_tol
        centered_y = abs(dy) <= dy_tol
        centered = centered_x and (centered_y if allow_heave_align else True)

        # ---------------- ALIGN_SWAY ----------------
        if self.state == State.ALIGN_SWAY:
            if (now - self.state_enter_time) > align_timeout:
                self.stop()
                self.set_state(State.SEARCH)
                return

            if centered_x:
                self.stop()
                self.set_state(State.ALIGN_HEAVE if allow_heave_align else State.SURGE)
                return

            sway_cmd = clamp((dx / 200.0) * sway_speed, -sway_speed, sway_speed)
            self.publish_cmd(surge=0.0, sway= -1 * sway_cmd, heave=0.0, yaw=0.0)
            return

        # ---------------- ALIGN_HEAVE ----------------
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

        # ---------------- SURGE ----------------
        if self.state == State.SURGE:
            # If drifted, re-align
            if not centered_x:
                self.stop()
                self.set_state(State.ALIGN_SWAY)
                return
            if allow_heave_align and not centered_y:
                self.stop()
                self.set_state(State.ALIGN_HEAVE)
                return

            # Optional: require stable center before surging (reduces jitter-driven drift)
            if not self._center_stable_ok(now, centered_x and (centered_y if allow_heave_align else True)):
                self.stop()
                return

            if area >= area_close - 5000:
                self.stop()
                self.set_state(State.ADVANCE_SLIGHTLY)
                return

            if (now - self.state_enter_time) > 10:
                self.stop()
                self.get_logger().warn(
                    f"SURGE timeout (area={area:.0f} < target={area_close:.0f}). Returning to SEARCH."
                )
                self.set_state(State.SEARCH)
                return

            self.publish_cmd(surge=surge_speed, sway=0.0, heave=0.0, yaw=0.0)
            return

        # ---------------- ADVANCE_SLIGHTLY ----------------
        if self.state == State.ADVANCE_SLIGHTLY:
            adv_t = float(self.get_parameter("advance_time_s").value)
            adv_speed = float(self.get_parameter("advance_speed").value)

            if (now - self.state_enter_time) < adv_t:
                self.publish_cmd(surge=0.075, sway=0.0, heave=0.0, yaw=0.0)
                return
            
            if area >= area_close - 2000:
                self.stop()
                self.set_state(State.POPPED)
                return


            self.stop()
            self.set_state(State.SURFACE)
            return
        
        if self.state == State.POPPED:
            # Optional: hold position a moment so "pop" event / physics settles
            pop_hold = 2  # seconds (hardcoded; or declare_parameter)
            

            if (now - self.state_enter_time) >= pop_hold:
                self.set_state(State.SURFACE)
                return
            
            self.stop()

            self.publish_cmd(0.0,0.0,0.0,0.0)
            self.set_state(State.SURFACE)

            return
            

        # Fallback
        self.stop()
        self.set_state(State.SEARCH)


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
