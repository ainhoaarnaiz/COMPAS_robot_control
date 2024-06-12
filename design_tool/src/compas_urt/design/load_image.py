from itertools import product

from compas.colors import Color
from compas.plugins import pluggable
from compas.plugins import plugin


def load_image(image_path):
    from System.Drawing import Bitmap

    pixels = {}

    image = Bitmap(image_path)
    width = image.Width
    height = image.Height

    for x, y in product(range(width), range(height)):
        dotnet_color = image.GetPixel(x, y)
        r, g, b, a = dotnet_color.R, dotnet_color.G, dotnet_color.B, dotnet_color.A
        color = Color(r / 255.0, g / 255.0, b / 255.0, a / 255.0)
        pixels[x, y] = color

    image.Dispose()

    return pixels, width, height


@plugin(pluggable_name="load_image", category="design", requires=["PIL"])
def _load_image_with_pillow(image_path):
    from PIL import Image

    pixels = {}

    with Image.open(image_path) as image:
        width = image.height
        height = image.width

        for x, y in product(range(height), range(width)):
            r, g, b = image.getpixel((x, y))
            color = Color(r / 255.0, g / 255.0, b / 255.0)
            pixels[x, y] = color
    return pixels, width, height


class LoadedImage(object):
    def __init__(self, image_path):
        self.pixels, self.width, self.height = load_image(image_path)
        # image = load_image(image_path)
        # print(image)

    def __str__(self):
        if self.width == self.height:
            shape = "a square"
        else:
            shape = "an"
        return "This is {} image with {} pixels, width = {}, and height = {}".format(
            shape, (self.width * self.height), self.width, self.height
        )
