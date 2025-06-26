import omni.graph.core as og
import omni.replicator.core as rep
import omni.syntheticdata
import omni.syntheticdata._syntheticdata as sd
from geometry_msgs.msg import Pose
from omni.isaac.sensor import Camera
from omni.syntheticdata import SyntheticData
from rclpy.node import Node


def publish_camera_info(camera: Camera, freq):
    # The following code will link the camera's render product and publish the data to the specified topic name.
    render_product = camera._render_product_path
    step_size = int(60/freq)
    topic_name = camera.name+"_camera_info"
    queue_size = 1
    node_namespace = ""
    frame_id = camera.prim_path.split("/")[-1]  # This matches what the TF tree is publishing.

    stereo_offset = [0.0, 0.0]

    writer = rep.writers.get("ROS2PublishCameraInfo")
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        topicName=topic_name,
        stereoOffset=stereo_offset,
    )
    writer.attach([render_product])

    gate_path = SyntheticData._get_node_path(
        "PostProcessDispatch" + "IsaacSimulationGate", render_product
    )

    # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)
    return


def publish_rgb(camera: Camera, freq):
    # The following code will link the camera's render product and publish the data to the specified topic name.
    render_product = camera._render_product_path
    step_size = int(60/freq)
    topic_name = camera.name+"_rgb"
    queue_size = 1
    node_namespace = ""
    frame_id = camera.prim_path.split("/")[-1]  # This matches what the TF tree is publishing.

    rv = SyntheticData.convert_sensor_type_to_rendervar(sd.SensorType.Rgb.name)
    writer = rep.writers.get(rv + "ROS2PublishImage")
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        topicName=topic_name
    )
    writer.attach([render_product])

    # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
    gate_path = SyntheticData._get_node_path(
        rv + "IsaacSimulationGate", render_product
    )
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)

    return


def publish_depth(camera: Camera, freq):
    # The following code will link the camera's render product and publish the data to the specified topic name.
    render_product = camera._render_product_path
    step_size = int(60/freq)
    topic_name = camera.name+"_depth"
    queue_size = 1
    node_namespace = ""
    frame_id = camera.prim_path.split("/")[-1]  # This matches what the TF tree is publishing.

    rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
        sd.SensorType.DistanceToImagePlane.name
    )
    writer = rep.writers.get(rv + "ROS2PublishImage")
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        topicName=topic_name
    )
    writer.attach([render_product])

    # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        rv + "IsaacSimulationGate", render_product
    )
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)

    return


class goal_position(Node):
    def __init__(self, topic_name='goal_position', goal_position=[7.0, 7.0]):
        super().__init__('goal_position')
        self.publisher_ = self.create_publisher(Pose, topic_name, 10)
        self.goal_position = Pose()
        self.goal_position.position.x = float(goal_position[0])
        self.goal_position.position.y = float(goal_position[1])

        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.publish_goal_position)

    def publish_goal_position(self):
        self.publisher_.publish(self.goal_position)


class RoverPose(Node):
    def __init__(self, topic_name='robot_position'):
        super().__init__('robot_pose')
        self.publisher_ = self.create_publisher(Pose, topic_name, 10)

    def publish_robot_position(self, rover_pose):
        pose = Pose()
        pose.position.x = float(rover_pose[0])
        pose.position.y = float(rover_pose[1])
        pose.position.z = float(rover_pose[2])

        pose.orientation.w = float(rover_pose[3])
        pose.orientation.x = float(rover_pose[4])
        pose.orientation.y = float(rover_pose[5])
        pose.orientation.z = float(rover_pose[6])

        self.publisher_.publish(pose)

# def publish_position(Node):
#     def
