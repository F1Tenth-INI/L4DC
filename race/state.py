"""Layout of the car state vector (alphabetical order of the variable names)."""

ANGULAR_VEL_Z_IDX = 0  # yaw rate [rad/s]
LINEAR_VEL_X_IDX = 1  # speed along the car [m/s]
LINEAR_VEL_Y_IDX = 2  # speed across the car [m/s]
POSE_THETA_IDX = 3  # heading [rad]
POSE_THETA_COS_IDX = 4
POSE_THETA_SIN_IDX = 5
POSE_X_IDX = 6  # [m]
POSE_Y_IDX = 7  # [m]
SLIP_ANGLE_IDX = 8  # [rad]
STEERING_ANGLE_IDX = 9  # [rad]
STATE_DIM = 10
