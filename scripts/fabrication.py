import time

from compas.geometry import Frame
from compas.robots import Configuration
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_io import RTDEIOInterface
from rtde_receive import RTDEReceiveInterface as RTDEReceive


def get_config(ip="127.0.0.1"):
    ur_r = RTDEReceive(ip)
    robot_joints = ur_r.getActualQ()
    config = Configuration.from_revolute_values(robot_joints)
    return config


def get_tcp_offset(ip="127.0.0.1"):
    ur_c = RTDEControl(ip)
    tcp = ur_c.getTCPOffset()
    return tcp


def move_to_joints(config, speed, accel, nowait, ip="127.0.0.1"):
    # speed rad/s, accel rad/s^2, nowait bool
    ur_c = RTDEControl(ip)
    ur_c.moveJ(config.joint_values, speed, accel, nowait)


def movel_to_joints(config, speed, accel, nowait, ip="127.0.0.1"):
    # speed rad/s, accel rad/s^2, nowait bool
    ur_c = RTDEControl(ip)
    ur_c.moveL_FK(config.joint_values, speed, accel, nowait)


def get_digital_io(signal, ip="127.0.0.1"):
    ur_r = RTDEReceive(ip)
    return ur_r.getDigitalOutState(signal)


def set_digital_io(signal, value, ip="127.0.0.1"):
    io = RTDEIOInterface(ip)
    io.setStandardDigitalOut(signal, value)


def set_tool_digital_io(signal, value, ip="127.0.0.1"):
    io = RTDEIOInterface(ip)
    io.setToolDigitalOut(signal, value)


def get_tcp_frame(ip="127.0.0.1"):
    ur_r = RTDEReceive(ip)
    tcp = ur_r.getActualTCPPose()
    frame = Frame.from_axis_angle_vector(tcp[3:], point=tcp[0:3])
    return frame


def move_trajectory(configurations, speed, accel, blend, ip="127.0.0.1"):
    print(f"Move trajectory of {len(configurations)} points with speed {speed}, accel {accel} and blend {blend}")
    ur_c = RTDEControl(ip)
    path = []
    for config in configurations:
        path.append(config.joint_values + [speed, accel, blend])

    if len(path):
        ur_c.moveJ(path)


def start_teach_mode(ip="127.0.0.1"):
    ur_c = RTDEControl(ip)
    ur_c.teachMode()


def stop_teach_mode(ip="127.0.0.1"):
    ur_c = RTDEControl(ip)
    ur_c.endTeachMode()


def move_until_contact(
    xd=[0.0, 0.0, -0.03, 0.0, 0.0, 0.0], direction=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], acceleration=0.1, ip="127.0.0.1"
):
    """Move until contact force is reached.

    Args:
        xd (list[float]): tool speed [m/s] (spatial vector)
        direction (list[float]): List of six floats. The first three elements are interpreted as a 3D vector (in the robot base coordinate system)
                                 giving the direction in which contacts should be detected. If all elements of the list are zero, contacts from all
                                 directions are considered. You can also set direction=get_target_tcp_speed() in which case it will detect contacts
                                 in the direction of the TCP movement.
        acceleration (float): tool position acceleration [m/s^2].
        ip (str, optional): Robot IP address. Defaults to "127.0.0.1".
    """
    ur_c = RTDEControl(ip)
    contacted = ur_c.moveUntilContact(
        xd=xd,
        direction=direction,
        acceleration=acceleration,
    )

    if not contacted:
        raise Exception("Not found contact!")


if __name__ == "__main__":
    ip = "192.168.56.101"
    frame = get_tcp_frame(ip)
    print(frame)
    config = get_config(ip)
    print(config)
