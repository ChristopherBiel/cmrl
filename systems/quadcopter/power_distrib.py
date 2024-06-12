
import pdb
import numpy as np

def compute_motor_pwm(base_thrust, motor_variation, params):
    temp = np.array([[-0.5, -0.5, -1.0],
                         [-0.5, +0.5, +1.0],
                         [+0.5, +0.5, -1.0],
                         [+0.5, -0.5, +1.0]])
    """
    temp = np.array([[+0.5, -0.5, -1.0],
                         [-0.5, -0.5, +1.0],
                         [-0.5, +0.5, -1.0],
                         [+0.5, +0.5, +1.0]])
    """
    adjustment = temp.dot(motor_variation)
    motor_pwm = base_thrust + adjustment
    motor_pwm = np.maximum(motor_pwm, params.quad.thrust_min)
    motor_pwm = np.minimum(motor_pwm, params.quad.thrust_max)

    return motor_pwm


def main():
    compute_motor_pwm(100, [100,-20,2])
    pdb.set_trace()

    pass


if __name__ == "__main__":
    main()
