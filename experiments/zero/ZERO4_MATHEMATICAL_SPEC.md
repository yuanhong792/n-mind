# ZERO-4 update rule

For group i, `theta_i <- theta_i - eta * sigmoid(g_i) * grad_i`. The controller sees only current support loss, gradient norm, step fraction, group weight norm and previous strengths. Hard protection is reported only when the applied multiplier is exactly zero; merely reducing a multiplier is soft modulation. The continual objective is `L_new + lambda * L_old`, with lambda frozen at preregistration.
