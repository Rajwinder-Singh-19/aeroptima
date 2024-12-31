import numpy as np

def lerp(start, end, t):
    lerp_point = np.array((1 - t) * start + t * end)
    return lerp_point


def quad(start, mid, end, t):
    lerp_s_m = lerp(start, mid, t)
    lerp_m_e = lerp(mid, end, t)
    quad = lerp(lerp_s_m, lerp_m_e, t)
    return quad


def cubic(start, mid_1, mid_2, end, t):
    lerp_s_m1 = lerp(start, mid_1, t)
    lerp_m1_m2 = lerp(mid_1, mid_2, t)
    lerp_m2_e = lerp(mid_2, end, t)
    cubic = quad(lerp_s_m1, lerp_m1_m2, lerp_m2_e, t)
    return cubic