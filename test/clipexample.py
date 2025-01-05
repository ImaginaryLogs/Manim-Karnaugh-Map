from manim import *

class ClippingExample(Scene):
    def construct(self):
        # Create a square
        square = Square(side_length=2, color=BLUE)

        # Create a clip region (a rectangle)
        clip_region = Rectangle(width=1, height=2)
        clip_region.move_to(square.get_center() + RIGHT)  # Position to clip a corner

        # Apply the clipping path to the square
        square.set_clip_path(clip_region)

        # Add the clipped square
        self.add(square)
        self.wait()