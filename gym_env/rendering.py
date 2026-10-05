"""Pyglet rendering adapted from neuron_poker."""
from math import radians, cos, sin

import numpy as np
import pyglet

WHITE = list(np.array([255, 255, 255, 255]))
GREEN = list(np.array([0, 255, 0, 123]))
BLUE = list(np.array([0, 0, 128, 255]))
RED = list(np.array([255, 0, 0, 123]))


class PygletWindow:
    def __init__(self, width, height):
        self.active = True
        self.display_surface = pyglet.window.Window(width=width, height=height + 50)
        self.display_surface.set_caption("Neuron Poker - jeu manuel")
        self.display_surface.push_handlers(on_close=self._on_close)
        self.top = height
        self.display_surface.switch_to()
        from pyglet.gl import glEnable, glBlendFunc, GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self.reset()

    def circle(self, x_pos, y_pos, radius, color, thickness, numPoints=100):
        _ = thickness
        verts = []
        y_pos = self.top - y_pos
        from pyglet.gl import glColor4ub, GL_LINE_LOOP
        glColor4ub(*[int(c) for c in color])
        for i in range(numPoints):
            angle = radians(float(i) / numPoints * 360.0)
            verts += [radius * cos(angle) + x_pos, radius * sin(angle) + y_pos]
        circle = pyglet.graphics.vertex_list(numPoints, ("v2f", verts))
        circle.draw(GL_LINE_LOOP)

    def text(self, text, x, y, font_size=20, color=None):
        y = self.top - y
        label = pyglet.text.Label(
            text, font_size=font_size, x=x, y=y,
            anchor_x="left", anchor_y="top", color=[int(c) for c in color]
        )
        label.draw()

    def rectangle(self, x, y, dx, dy, color):
        y = self.top - y
        x, y = int(round(x)), int(round(y))
        from pyglet.gl import glColor4ub
        glColor4ub(*[int(c) for c in color])
        rect = pyglet.graphics.draw(
            4, pyglet.gl.GL_QUADS,
            ("v2f", [x, y, x + dx, y, x + dx, y + dy, x, y + dy]),
        )
        # pyglet 1.5 returns a vertex list; older code called draw() again.
        if hasattr(rect, "draw"):
            rect.draw()

    def reset(self):
        self.display_surface.switch_to()
        if not pyglet.app.event_loop.is_running:
            pyglet.clock.tick()
            self.display_surface.dispatch_events()
        from pyglet.gl import glClear
        glClear(pyglet.gl.GL_COLOR_BUFFER_BIT)

    def update(self):
        if not pyglet.app.event_loop.is_running:
            self.display_surface.flip()

    def _on_close(self):
        self.active = False

    def close(self):
        self.active = False
        self.display_surface.close()
