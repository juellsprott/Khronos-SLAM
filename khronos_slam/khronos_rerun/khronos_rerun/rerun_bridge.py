#!/usr/bin/env python3
"""ROS2 node that subscribes to Khronos/uHumans2 topics and logs them to Rerun."""

import numpy as np
import rclpy
import rerun as rr
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, Imu
from visualization_msgs.msg import Marker, MarkerArray


def _stamp_to_secs(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


class RerunBridge(Node):
    """Subscribes to Khronos-related ROS2 topics and streams data to Rerun."""

    def __init__(self):
        super().__init__("rerun_bridge")

        # --- Parameters ---
        self.declare_parameter("recording_id", "khronos")
        self.declare_parameter("spawn_viewer", True)
        self.declare_parameter("rgb_topic", "/camera/daheng/left/image_rect")
        self.declare_parameter("depth_topic", "/camera/daheng/depth_image")
        self.declare_parameter("label_topic", "/camera/daheng/left/image_seg")
        self.declare_parameter("camera_info_topic", "/camera/daheng/left/camera_info")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("imu_topic", "/imu")

        recording_id = self.get_parameter("recording_id").value
        spawn_viewer = self.get_parameter("spawn_viewer").value
        rgb_topic = self.get_parameter("rgb_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        label_topic = self.get_parameter("label_topic").value
        camera_info_topic = self.get_parameter("camera_info_topic").value
        odom_topic = self.get_parameter("odom_topic").value
        imu_topic = self.get_parameter("imu_topic").value

        # --- Rerun init ---
        rr.init(recording_id)
        if spawn_viewer:
            rr.spawn()
        else:
            rr.connect_grpc()

        colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        vectors = [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]
        origins = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        labels = ["X", "Y", "Z"]
        rr.log(
            "/",
            rr.Arrows3D(origins=origins, vectors=vectors, colors=colors, labels=labels),
        )

        self._bridge = CvBridge()

        # Use a permissive QoS to match most publishers
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # =====================================================================
        # Sensor inputs (from bag / simulator)
        # =====================================================================

        # RGB camera
        self.create_subscription(
            Image, rgb_topic, self._cb_rgb, sensor_qos
        )

        # Depth camera
        self.create_subscription(
            Image, depth_topic, self._cb_depth, sensor_qos
        )

        # Semantic segmentation (ground truth labels)
        self.create_subscription(
            Image,
            label_topic,
            self._cb_segmentation,
            sensor_qos,
        )

        # Camera info (for pinhole model)
        self.create_subscription(
            CameraInfo, camera_info_topic, self._cb_camera_info, sensor_qos
        )

        # Odometry
        self.create_subscription(Odometry, odom_topic, self._cb_odom, sensor_qos)

        # IMU
        self.create_subscription(Imu, imu_topic, self._cb_imu, sensor_qos)

        # =====================================================================
        # Khronos visualisation outputs
        # =====================================================================

        # Images from Khronos
        self.create_subscription(
            Image,
            "/khronos_node/visualization/dynamic_image",
            lambda msg: self._cb_image_generic(msg, "khronos/dynamic_image"),
            reliable_qos,
        )
        self.create_subscription(
            Image,
            "/khronos_node/visualization/semantic_image",
            lambda msg: self._cb_image_generic(msg, "khronos/semantic_image"),
            reliable_qos,
        )
        self.create_subscription(
            Image,
            "/khronos_node/visualization/object_image",
            lambda msg: self._cb_image_generic(msg, "khronos/object_image"),
            reliable_qos,
        )
        self.create_subscription(
            Image,
            "/khronos_node/visualization/tracking/image",
            lambda msg: self._cb_image_generic(msg, "khronos/tracking/image"),
            reliable_qos,
        )

        # Marker topics (point-cloud-like slices)
        self.create_subscription(
            Marker,
            "/khronos_node/visualization/dynamic_points",
            lambda msg: self._cb_marker(msg, "khronos/dynamic_points"),
            reliable_qos,
        )
        self.create_subscription(
            Marker,
            "/khronos_node/visualization/ever_free_slice",
            lambda msg: self._cb_marker(msg, "khronos/ever_free_slice"),
            reliable_qos,
        )
        self.create_subscription(
            Marker,
            "/khronos_node/visualization/tsdf_slice",
            lambda msg: self._cb_marker(msg, "khronos/tsdf_slice"),
            reliable_qos,
        )
        self.create_subscription(
            Marker,
            "/khronos_node/visualization/tracking_slice",
            lambda msg: self._cb_marker(msg, "khronos/tracking_slice"),
            reliable_qos,
        )

        # MarkerArray topics (bounding boxes, voxels, etc.)
        # self.create_subscription(
        #     MarkerArray,
        #     "/khronos_node/visualization/object_bounding_boxes",
        #     lambda msg: self._cb_marker_array(msg, "khronos/object_bounding_boxes"),
        #     reliable_qos,
        # )
        self.create_subscription(
            MarkerArray,
            "/khronos_node/visualization/tracking/bounding_box",
            lambda msg: self._cb_marker_array(msg, "khronos/tracking/bounding_box"),
            reliable_qos,
        )
        self.create_subscription(
            MarkerArray,
            "/khronos_node/visualization/tracking/pixels",
            lambda msg: self._cb_marker_array(msg, "khronos/tracking/pixels"),
            reliable_qos,
        )
        self.create_subscription(
            MarkerArray,
            "/khronos_node/visualization/tracking/voxels",
            lambda msg: self._cb_marker_array(msg, "khronos/tracking/voxels"),
            reliable_qos,
        )

        # Hydra scene graph visualisation
        self.create_subscription(
            MarkerArray,
            "/hydra_visualizer/graph",
            lambda msg: self._cb_marker_array(msg, "hydra/graph"),
            reliable_qos,
        )
        self.create_subscription(
            MarkerArray,
            "/hydra_visualizer/dynamic_objects",
            lambda msg: self._cb_marker_array(msg, "hydra/dynamic_objects"),
            reliable_qos,
        )

        self.get_logger().info("Rerun bridge started — streaming to viewer")

        # Track whether we already logged the pinhole model
        self._pinhole_logged = False

    # --------------------------------------------------------------------- #
    # Sensor callbacks
    # --------------------------------------------------------------------- #

    def _cb_rgb(self, msg: Image):
        # rr.set_time("ros_time", t)
        cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        rr.log("camera/rgb", rr.Image(cv_img))

    def _cb_depth(self, msg: Image):
        # rr.set_time_seconds("ros_time", t)
        cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth = np.asarray(cv_img, dtype=np.float32)
        # If encoded as 16UC1 (millimetres) convert to metres
        if msg.encoding == "16UC1":
            depth = depth / 1000.0
        rr.log("camera/depth", rr.DepthImage(depth, meter=1.0))

    def _cb_segmentation(self, msg: Image):
        # rr.set_time_seconds("ros_time", t)
        cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        label_img = np.asarray(cv_img)
        # If multi-channel, take first channel as class ID
        if label_img.ndim == 3:
            label_img = label_img[:, :, 0]
        rr.log("camera/segmentation", rr.SegmentationImage(label_img))

    def _cb_camera_info(self, msg: CameraInfo):
        if self._pinhole_logged:
            return
        self._pinhole_logged = True
        # K is a row-major 3x3 intrinsic matrix
        fx, fy = msg.k[0], msg.k[4]
        cx, cy = msg.k[2], msg.k[5]
        w, h = msg.width, msg.height
        rr.log(
            "camera",
            rr.Pinhole(
                focal_length=[fx, fy],
                principal_point=[cx, cy],
                resolution=[w, h],
                image_plane_distance=0.15,
                camera_xyz=rr.ViewCoordinates.FLU,
            ),
            static=True,
        )

    def _cb_odom(self, msg: Odometry):
        # rr.set_time_seconds("ros_time", t)
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation

        # Capture initial XY as origin
        if not hasattr(self, "_odom_x0"):
            self._odom_x0 = pos.x
            self._odom_y0 = pos.y
            print(f"Captured odometry origin at ({self._odom_x0}, {self._odom_y0})")

        # Log as 3D transform
        rr.log(
            "camera",
            rr.Transform3D(
                translation=[pos.x, pos.y, pos.z],
                rotation=rr.Quaternion(xyzw=[ori.x, ori.y, ori.z, ori.w]),
            ),
        )

        # Also log scalar trajectory traces
        rr.log("odometry/x", rr.Scalars(pos.x))
        rr.log("odometry/y", rr.Scalars(pos.y))
        rr.log("odometry/z", rr.Scalars(pos.z))

    def _cb_imu(self, msg: Imu):
        # rr.set_time_seconds("ros_time", t)
        a = msg.linear_acceleration
        g = msg.angular_velocity
        rr.log("imu/accel/x", rr.Scalars(a.x))
        rr.log("imu/accel/y", rr.Scalars(a.y))
        rr.log("imu/accel/z", rr.Scalars(a.z))
        rr.log("imu/gyro/x", rr.Scalars(g.x))
        rr.log("imu/gyro/y", rr.Scalars(g.y))
        rr.log("imu/gyro/z", rr.Scalars(g.z))

    # khronos callbacks

    def _cb_image_generic(self, msg: Image, entity_path: str):
        """Log any Image topic under a given Rerun entity path."""
        # rr.set_time_seconds("ros_time", t)
        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except Exception:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        rr.log(entity_path, rr.Image(cv_img))

    def _cb_marker(self, msg: Marker, entity_path: str):
        """Convert a visualization_msgs/Marker to Rerun primitives."""
        # rr.set_time_seconds("ros_time", t)

        # Log the marker's pose as a Transform3D so geometry is placed correctly
        pos = msg.pose.position
        ori = msg.pose.orientation
        rr.log(
            entity_path,
            rr.Transform3D(
                translation=[pos.x, pos.y, pos.z],
                rotation=rr.Quaternion(xyzw=[ori.x, ori.y, ori.z, ori.w]),
            ),
        )

        if msg.type in (Marker.POINTS, Marker.SPHERE_LIST, Marker.CUBE_LIST):
            self._log_marker_points(msg, entity_path)
        elif msg.type == Marker.LINE_STRIP:
            self._log_marker_line_strip(msg, entity_path)
        elif msg.type == Marker.LINE_LIST:
            self._log_marker_line_list(msg, entity_path)
        elif msg.type in (Marker.CUBE, Marker.SPHERE, Marker.CYLINDER):
            self._log_marker_single_shape(msg, entity_path)
        elif msg.type == Marker.ARROW:
            self._log_marker_arrow(msg, entity_path)
        elif msg.type == Marker.TEXT_VIEW_FACING:
            rr.log(entity_path, rr.TextLog(msg.text))
        elif msg.type == Marker.TRIANGLE_LIST:
            self._log_marker_mesh(msg, entity_path)
        elif msg.type == Marker.DELETE or msg.action == Marker.DELETE:
            rr.log(entity_path, rr.Clear(recursive=False))

    def _cb_marker_array(self, msg: MarkerArray, entity_path: str):
        """Dispatch each marker in the array individually."""
        for marker in msg.markers:
            ns = marker.ns or "default"
            child_path = f"{entity_path}/{ns}/{marker.id}"
            if marker.action == Marker.DELETEALL:
                rr.log(entity_path, rr.Clear(recursive=True))
                return
            self._cb_marker(marker, child_path)

    # Marker → Rerun helpers
    @staticmethod
    def _extract_points(msg: Marker) -> np.ndarray:
        """Extract Nx3 point array from a Marker."""
        return np.array([[p.x, p.y, p.z] for p in msg.points], dtype=np.float32)

    @staticmethod
    def _extract_colors(msg: Marker) -> np.ndarray | None:
        """Extract per-point RGBA colours if present."""
        if msg.colors:
            return np.array(
                [[c.r, c.g, c.b, c.a] for c in msg.colors], dtype=np.float32
            )
        if msg.color.a > 0:
            return np.array(
                [[msg.color.r, msg.color.g, msg.color.b, msg.color.a]],
                dtype=np.float32,
            )
        return None

    def _log_marker_points(self, msg: Marker, entity_path: str):
        pts = self._extract_points(msg)
        if pts.size == 0:
            return
        colors = self._extract_colors(msg)
        radii = None
        if msg.type == Marker.POINTS:
            radii = [msg.scale.x * 0.5]
        elif msg.type == Marker.SPHERE_LIST:
            radii = [msg.scale.x * 0.5]
        elif msg.type == Marker.CUBE_LIST:
            radii = [msg.scale.x * 0.5]
        kwargs = {"positions": pts}
        if colors is not None:
            kwargs["colors"] = colors
        if radii is not None:
            kwargs["radii"] = radii
        rr.log(entity_path, rr.Points3D(**kwargs))

    def _log_marker_line_strip(self, msg: Marker, entity_path: str):
        pts = self._extract_points(msg)
        if pts.size == 0:
            return
        colors = self._extract_colors(msg)
        kwargs = {"strips": [pts]}
        if colors is not None:
            kwargs["colors"] = colors
        rr.log(entity_path, rr.LineStrips3D(**kwargs))

    def _log_marker_line_list(self, msg: Marker, entity_path: str):
        pts = self._extract_points(msg)
        if pts.size == 0:
            return
        # LINE_LIST: pairs of points
        segments = pts.reshape(-1, 2, 3)
        colors = self._extract_colors(msg)
        kwargs = {"strips": segments.tolist()}
        if colors is not None:
            kwargs["colors"] = colors
        rr.log(entity_path, rr.LineStrips3D(**kwargs))

    def _log_marker_single_shape(self, msg: Marker, entity_path: str):
        """Log CUBE / SPHERE / CYLINDER as a box at the local origin.

        The marker's pose is already applied via Transform3D in _cb_marker,
        so we place the box at the local origin.
        """
        s = msg.scale
        colors = self._extract_colors(msg)
        kwargs: dict = {
            "centers": [[0.0, 0.0, 0.0]],
            "half_sizes": [[s.x / 2, s.y / 2, s.z / 2]],
        }
        if colors is not None:
            kwargs["colors"] = colors
        rr.log(entity_path, rr.Boxes3D(**kwargs))

    def _log_marker_arrow(self, msg: Marker, entity_path: str):
        """Log ARROW markers as Arrows3D."""
        colors = self._extract_colors(msg)
        if len(msg.points) >= 2:
            origin = np.array([msg.points[0].x, msg.points[0].y, msg.points[0].z])
            tip = np.array([msg.points[1].x, msg.points[1].y, msg.points[1].z])
            vec = tip - origin
            kwargs: dict = {
                "origins": [origin],
                "vectors": [vec],
            }
            if colors is not None:
                kwargs["colors"] = colors
            rr.log(entity_path, rr.Arrows3D(**kwargs))

    def _log_marker_mesh(self, msg: Marker, entity_path: str):
        """Log TRIANGLE_LIST as a Mesh3D."""
        pts = self._extract_points(msg)
        if pts.size == 0:
            return
        n_tri = len(pts) // 3
        indices = np.arange(n_tri * 3, dtype=np.uint32).reshape(-1, 3)
        colors = self._extract_colors(msg)
        kwargs: dict = {
            "vertex_positions": pts,
            "triangle_indices": indices,
        }
        if colors is not None:
            if len(colors) == len(pts):
                kwargs["vertex_colors"] = colors
            else:
                kwargs["vertex_colors"] = np.tile(colors[0], (len(pts), 1))
        rr.log(entity_path, rr.Mesh3D(**kwargs))


def main(args=None):
    rclpy.init(args=args)
    node = RerunBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
