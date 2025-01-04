import webcolors


def closest_color(request_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - request_colour[0]) ** 2
        gd = (g_c - request_colour[1]) ** 2
        bd = (b_c - request_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]


def get_color_name(request_colour):
    try:
        name_closest = name_actual = webcolors.rgb_to_name(request_colour)
    except ValueError:
        name_closest = closest_color(request_colour)
        name_actual = None
    return name_actual, name_closest


request_colour = (223, 208, 196)
actual_name, closest_name = get_color_name(request_colour)
print(actual_name, closest_name)
