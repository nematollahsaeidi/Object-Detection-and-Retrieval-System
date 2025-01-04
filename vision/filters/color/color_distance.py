import math


def distance_color(color1, color2):
    (r1, g1, b1) = color1
    (r2, g2, b2) = color2
    return math.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2)