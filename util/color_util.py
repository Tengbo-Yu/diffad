class ColorBoard:
    red = (255, 0, 0)
    white = (255, 255, 255)
    yellow = (255, 255, 0)
    purple = (128, 0, 128)
    light_purple = (160, 32, 240)
    teal = (0, 128, 128)
    gray = (192, 192, 192)
    blue = (0, 0, 255)
    dark_blue = (0, 0, 139)
    light_blue = (173, 216, 230)
    orange = (255, 165, 0)
    dark_red = (139, 0, 0)
    light_red = (255, 127, 127)
    green = (0, 255, 0)
    light_green = (144, 238, 144)
    dark_green = (2, 48, 32)
    light_yellow = (255, 255, 224)
    dark_yellow = (139, 128, 0)
    pink = (255, 105, 180)
    cyan = (0, 255, 255)
    light_cyan = (0, 128, 255)
    magenta = (255, 0, 255)


def interpolate_color(start_color, end_color, t):
    """
    Compute the interpolated color between start_color and end_color.
    :param start_color: The starting color in BGR format.
    :param end_color: The ending color in BGR format.
    :param t: A value between 0 and 1 representing the interpolation factor.
    :return: The interpolated color.
    """
    return tuple(int(start_color[i] + (end_color[i] - start_color[i]) * t) for i in range(3))
