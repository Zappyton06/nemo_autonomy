import rclpy
from rclpy.node import Node
from enum import Enum
from std_msgs.msg import Bool
from nemo_interfaces.msg import RovCommands


class AUVState(Enum):
    WAIT_START = 0
    DIVE = 1
    HOLD_SURGE = 2
    SWAY = 3
    YAW = 4
    EXTERNAL_SURGE = 5
    IDLE = 6


class AUVStateMachine(Node):

    def __init__(self):
        super().__init__('auv_state_machine')

        # ---------------- Publisher & Subscriber ----------------
        self.cmd_pub = self.create_publisher(
            RovCommands,
            '/nemo_auv/input_cmd',
            10
        )

        

        # ---------------- State ----------------
        self.state = AUVState.WAIT_START
        self.state_start_time = self.get_clock().now()
        self.kill_switch = False
        self.kill_latched = False

        # ---------------- Timer ----------------
        self.dt = 0.05  # 20 Hz
        self.timer = self.create_timer(self.dt, self.update)
        self.get_logger().info("AUV FSM started (float RovCommands)")

        # ---------------- Slew-rate / smoothing ----------------
        self.cmd_current = {
            'surge': 0.0,
            'sway': 0.0,
            'yaw': 0.0,
            'heave': 0.0
        }

        self.cmd_target = {
            'surge': 0.0,
            'sway': 0.0,
            'yaw': 0.0,
            'heave': 0.0
        }

        # Max rate change per second for smooth motion
        self.slew_rate = {
            'surge': 0.3,
            'sway': 0.3,
            'yaw': 0.5,
            'heave': 0.4
        }

    # ---------------- Kill switch ----------------

    # ---------------- State helpers ----------------
    def enter_state(self, new_state):
        if self.state != new_state:
            self.state = new_state
            self.state_start_time = self.get_clock().now()
            self.get_logger().info(f"State → {self.state.name}")

    def time_in_state(self):
        now = self.get_clock().now()
        return (now - self.state_start_time).nanoseconds * 1e-9

    # ---------------- Slew rate limiter ----------------
    def ramp(self, current, target, rate):
        max_step = rate * self.dt
        delta = target - current
        if abs(delta) <= max_step:
            return target
        return current + max_step * (1.0 if delta > 0 else -1.0)

    # ---------------- FSM Update ----------------
    def update(self):

        # ---------- Kill logic ----------
        if self.kill_switch and not self.kill_latched:
            self.kill_latched = True
            self.enter_state(AUVState.EXTERNAL_SURGE)

        t = self.time_in_state()

        # ---------- FSM: Set target commands ----------
        # Reset targets
        self.cmd_target['surge'] = 0.0
        self.cmd_target['sway']  = 0.0
        self.cmd_target['yaw']   = 0.0
        self.cmd_target['heave'] = 0.0

        if self.state == AUVState.WAIT_START:
            if t >= 10.0:
                self.enter_state(AUVState.DIVE)
                self.get_logger().info("Starting the Dive")

        elif self.state == AUVState.DIVE:
            self.cmd_target['heave'] = -0.5
            if t >= 3.5:
                self.enter_state(AUVState.HOLD_SURGE)
                self.get_logger().info("Starting the Surge Process")

        elif self.state == AUVState.HOLD_SURGE:
            self.cmd_target['surge'] = 0.5
            self.cmd_target['heave'] = -0.25
            if t >= 5.0:
                self.enter_state(AUVState.SWAY)
                self.get_logger().info("Starting the Sway Process")

        elif self.state == AUVState.SWAY:
            self.cmd_target['sway']  = 0.5
            self.cmd_target['heave'] = -0.2
            if t >= 3.0:
                self.enter_state(AUVState.YAW)
                self.get_logger().info("Starting the Yaw")

        elif self.state == AUVState.YAW:
            self.cmd_target['yaw']   = 0.4
            self.cmd_target['heave'] = -0.2
            if t >= 3.0:
                self.enter_state(AUVState.EXTERNAL_SURGE)
                self.get_logger().info("Starting the External Surge")

        elif self.state == AUVState.EXTERNAL_SURGE:
            self.cmd_target['surge'] = 0.3

        # ---------- Apply smoothing ----------
        for axis in self.cmd_current:
            self.cmd_current[axis] = self.ramp(
                self.cmd_current[axis],
                self.cmd_target[axis],
                self.slew_rate[axis]
            )

        # ---------- Publish ----------
        cmd = RovCommands()
        cmd.surge = self.cmd_current['surge']
        cmd.sway  = self.cmd_current['sway']
        cmd.yaw   = self.cmd_current['yaw']
        cmd.heave = self.cmd_current['heave']

        self.cmd_pub.publish(cmd)


def main():
    rclpy.init()
    node = AUVStateMachine()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
