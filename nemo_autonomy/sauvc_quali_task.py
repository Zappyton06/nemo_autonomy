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

        self.declare_parameter("depth_lock_cmd",0.00)
        self.depth_lock_cmd = self.get_parameter("depth_lock_cmd").value

        self.declare_parameter("dive_rate_cmd",0.25)
        self.dive_rate_cmd = self.get_parameter("dive_rate_cmd").value


        self.cmd_pub = self.create_publisher(
            RovCommands,
            '/nemo_auv/input_cmd',
            10
        )

        self.kill_sub = self.create_subscription(
            Bool,
            '/kill_switch',
            self.kill_callback,
            10
        )

        self.state = AUVState.WAIT_START
        self.state_start_time = self.get_clock().now()

        self.kill_switch = False        # FIXED
        self.kill_latched = False       # IMPORTANT

        self.timer = self.create_timer(0.05, self.update)  # 20 Hz
        self.get_logger().info("AUV FSM started (float RovCommands)")

    # ---------------- Kill switch ----------------
    def kill_callback(self, msg: Bool):
        self.kill_switch = msg.data

    # ---------------- State helpers ----------------
    def enter_state(self, new_state):
        if self.state != new_state:
            self.state = new_state
            self.state_start_time = self.get_clock().now()
            self.get_logger().info(f"State → {self.state.name}")

    def time_in_state(self):
        now = self.get_clock().now()
        return (now - self.state_start_time).nanoseconds * 1e-9

    # ---------------- FSM ----------------
    def update(self):
        # ---------- Kill logic ----------
        if self.kill_switch and not self.kill_latched:
            self.kill_latched = True
            self.enter_state(AUVState.EXTERNAL_SURGE)

        t = self.time_in_state()

        # ---------- Floats-safe command ----------
        cmd = RovCommands()
        cmd.surge = 0.0
        cmd.sway  = 0.0
        cmd.yaw   = 0.0
        cmd.heave = 0.0

        if self.state == AUVState.WAIT_START:
            if t >= 20.0:
                self.enter_state(AUVState.DIVE)
                print("Starting the Dive")

        elif self.state == AUVState.DIVE:
            cmd.heave = -self.dive_rate_cmd
            if t >= 3.5:
                self.enter_state(AUVState.HOLD_SURGE)
                print("Starting the Surge Process")

        elif self.state == AUVState.HOLD_SURGE:
            cmd.surge = 0.4
            cmd.heave = -self.depth_lock_cmd   # slight bias instead of zero-drop
            if t >= 5.0:
                self.enter_state(AUVState.SWAY)
                print("starting the sway process")

        elif self.state == AUVState.SWAY:
            cmd.sway = 0.25
            cmd.heave = -self.depth_lock_cmd
            if t >= 5.0:
                self.enter_state(AUVState.YAW)
                print("starting the Yaw")

        elif self.state == AUVState.YAW:
            cmd.yaw = 0.15
            cmd.heave = -self.depth_lock_cmd
            if t >= 3.0:
                self.enter_state(AUVState.EXTERNAL_SURGE)
                print("starting the external surge")
            

        elif self.state == AUVState.EXTERNAL_SURGE:
            cmd.surge = 0.3   # terminal recovery
            self.cmd_pub.publish(cmd)
            return

        self.cmd_pub.publish(cmd)


def main():
    rclpy.init()
    node = AUVStateMachine()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
