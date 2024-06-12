def motor_state_update(pwm, params):
    """ computes motor squared rpm from pwm """

    # assume pwm is a numpy array
    motor_rates = params.quad.pwm2rpm_scale * pwm + params.quad.pwm2rpm_const
    squared_motor_rates = motor_rates ** 2

    return squared_motor_rates

def main():
    motor_state_update()

if __name__ == "__main__":
    main()
