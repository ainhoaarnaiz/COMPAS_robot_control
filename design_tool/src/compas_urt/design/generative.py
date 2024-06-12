from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math as m
import random
import time

import Rhino.Geometry as rg
from compas.colors import Color
from compas.datastructures import Mesh
from compas.geometry import Circle
from compas.geometry import Frame
from compas.geometry import Plane
from compas.geometry import Vector
from compas.geometry import distance_point_point
from compas.utilities import flatten
from compas_rhino.conversions import RhinoCurve
from compas_rhino.conversions import frame_to_rhino

from compas_urt.design import DesignLayer
from compas_urt.design import RoundTile
from compas_urt.design.ellipse_with_frame import EllipseFrame


class GenLayer(DesignLayer):
    def generate(self):
        pass


class GridLayer(GenLayer):
    def __init__(self, rhino_brep, tile_diameter, tile_thickness, tile_joint, **kwargs):
        super(GridLayer, self).__init__(rhino_brep, **kwargs)
        # self.curve_type = curve_type
        self.tile_diameter = tile_diameter
        self.tile_joint = tile_joint
        self.tile_thickness = tile_thickness

    def generate(self):
        uniform = self.options["uniform"]
        pack_type = self.options["pack_type"]
        param_density = self.options["param_density"]
        flip_curves = self.options["flip_curves"]
        flip_frame = self.options["flip_frame"]
        curve_type = self.options.get("curve_type")
        reverse_curve_params = self.options.get("reverse_curve_params")

        unit_size = self.tile_diameter + self.tile_joint
        equilateral_height = unit_size * m.sqrt(3) / 2

        brepface = self.rhino_brep.Faces[0]

        frames = []
        uv_params = []

        # TODO: check which edge isocurve has the biggest length! otherwise you might not cover the whole surface with isocurves

        if pack_type == "hexagonal":
            unit = equilateral_height
            division_mode = "by_length"
        else:
            unit = unit_size
            division_mode = "by_count"

        min_u_edge = None
        min_v_edge = None

        for i in range(50):
            temp_u = self.compas_surface.u_isocurve(i / 50)
            temp_v = self.compas_surface.v_isocurve(i / 50)

            if temp_u.length() > 0:
                if min_u_edge is None or temp_u.length() < min_u_edge.length():
                    min_u_edge = temp_u
            if temp_v.length() > 0:
                if min_v_edge is None or temp_v.length() < min_v_edge.length():
                    min_v_edge = temp_v
        if min_u_edge is None or min_v_edge is None:
            raise Exception("The isocurve length in either u or v direction cannot be None")

        if curve_type == "isocurves":
            if flip_curves:
                # self.edge_isocurve = self.compas_surface.u_isocurve(0.5)
                self.edge_isocurve = min_u_edge
            else:
                # self.edge_isocurve = self.compas_surface.v_isocurve(0.5)
                self.edge_isocurve = min_v_edge

            edge_params = self.generate_params_on_curve(
                self.edge_isocurve,
                unit_size=unit,
                uniform=uniform,
                param_density=param_density,
                division_mode=division_mode,
            )

            self.isocurves = self.generate_isocurves(edge_params, flip_curves)

            for i, (edge_param, isocurve) in enumerate(zip(edge_params, self.isocurves)):
                if isocurve.length() < unit:
                    continue
                unit_reparam = unit_size / isocurve.length()
                parameters = self.generate_params_on_curve(isocurve, unit_size=unit_size, uniform=True, param_density=1)

                for isocurve_param in parameters:
                    if pack_type == "hexagonal":
                        if i % 2:
                            isocurve_param += unit_reparam / 2

                    if not flip_curves:
                        u = edge_param
                        v = isocurve_param
                    else:
                        v = edge_param
                        u = isocurve_param

                    pointBrepFaceRelationship = brepface.IsPointOnFace(u, v)
                    if pointBrepFaceRelationship == rg.PointFaceRelation.Exterior:
                        continue

                    point = isocurve.point_at(isocurve_param)
                    frame, uv_param = self.generate_tile_frame_on_surface(point, flip_frame)
                    frames.append(frame)
                    uv_params.append(uv_param)

        elif curve_type == "contours":
            if "input_curves" not in self.options:
                raise Exception("Please specify input curves.")

            input_curves = self.options["input_curves"]

            # for input_curve in input_curves:

            # edge_params = self.generate_params_on_curve(input_curve, unit, uniform, param_density)

            compas_contours_nested, self.contour_frames = self.generate_contours(
                input_curves, unit, uniform, param_density, reverse_curve_params
            )

            for i, contour_segments in enumerate(compas_contours_nested):
                for contour_segment in contour_segments:
                    params = self.generate_params_on_curve(
                        contour_segment, unit_size=unit, uniform=True, param_density=0.5
                    )

                    points = []
                    if params is None:
                        continue
                    for param in params:
                        point = contour_segment.point_at(param)
                        points.append(point)

                    for j, point in enumerate(points):
                        frame = None  # reset your frame, previously stored frame creates a fuckup

                        if pack_type == "hexagonal":
                            if (j % 2 == 0 and i % 2 == 0) or (j % 2 == 1 and i % 2 == 1):
                                frame, uv_param = self.generate_tile_frame_on_surface(point, flip_frame)
                        elif pack_type == "aligned":
                            if j % 2 == 0:
                                frame, uv_param = self.generate_tile_frame_on_surface(point, flip_frame)

                        if frame is not None:
                            uv_params.append(uv_param)
                            frames.append(frame)

            self.compas_contours = list(flatten(compas_contours_nested))

        for f, uv_param in zip(frames, uv_params):
            round_tile = RoundTile(
                base_frame=f, diameter=self.tile_diameter, thickness=self.tile_thickness, uv_param=uv_param
            )
            self.tiles.append(round_tile)


class BubblesFromCurveLayer(GenLayer):
    def __init__(
        self,
        rhino_brep,
        input_curve,
        division_num,
        tile_joint,
        effect_factor=0,
        xsize_domain=None,
        ysize_domain=None,
        **kwargs
    ):
        super(BubblesFromCurveLayer, self).__init__(rhino_brep, **kwargs)
        self.input_curve = input_curve
        self.division_num = division_num
        self.xsize_domain = xsize_domain
        self.ysize_domain = ysize_domain
        self.tile_joint = tile_joint
        self.effect_factor = effect_factor
        self.bubbles = []
        self.points_on_curve = []  # TODO: remove

    def generate(self):
        bubble_frames = self.generate_bubble_frames()
        for frame in bubble_frames:
            bubble_xsize = Bubble.assign_size_from_domain(self.xsize_domain)
            bubble_ysize = Bubble.assign_size_from_domain(self.ysize_domain)
            bubble = Bubble(frame, bubble_xsize, bubble_ysize)
            self.bubbles.append(bubble)

        self.relax_bubbles()

    def generate_bubble_frames(self):
        ## Generate initial ellipses
        ##------------------------------------------------------------------------------
        self.tiles = []

        bubble_frames = []

        if self.division_num == 0:
            raise Exception("Division points cannot be zero.")
        _, points = self.input_curve.divide_by_count(self.division_num, return_points=True)

        for point in points:
            _, uv_param = self.compas_surface.closest_point(point, return_parameters=True)
            frame = self.compas_surface.frame_at(uv_param[0], uv_param[1])
            bubble_frames.append(frame)

        return bubble_frames

    def relax_bubbles(self):
        TOLERANCE = 0.001
        TIMEOUT_IN_SECONDS = 60
        start_time = time.time()
        while True:
            self.push_bubbles()
            now_time = time.time()
            if self.sum_of_moves < TOLERANCE or (now_time - start_time > TIMEOUT_IN_SECONDS):
                break

        # Randomize the shape
        ##------------------------------------------------------------------------------

        # Project bubbles on compas_surface
        ##------------------------------------------------------------------------------

        for bubble in self.bubbles:
            bubble_shape = bubble.ellipse

            rhino_frame = frame_to_rhino(bubble.frame)

            rhino_bubble_shape = rg.Ellipse(rhino_frame, bubble_shape.major, bubble_shape.minor)
            rhino_nurbs_curve = rhino_bubble_shape.ToNurbsCurve()

            projected_curves = rg.Curve.ProjectToBrep(rhino_nurbs_curve, self.rhino_brep, rhino_frame.ZAxis, 0.01)

            for projected_curve in projected_curves:
                compas_nurbs_curve = RhinoCurve.from_geometry(projected_curve).to_compas()
                bubble.projected_curves.append(compas_nurbs_curve)

    def get_motion_vectors(self):
        collisions_count = []
        total_moves = []

        # initial values for total_moves and collisions_count
        ##------------------------------------------------------------------------------
        for i in range(len(self.bubbles)):
            total_moves.append(Vector(0.0, 0.0, 0.0))
            collisions_count.append(0)

        # double loop for accessing 2 tiles at the same time
        ##------------------------------------------------------------------------------
        self.circles = []
        for i in range(len(self.bubbles)):
            for j in range(i + 1, len(self.bubbles)):
                # get distance between 2 bubble centers
                ##------------------------------------------------------------------------------
                center_i = self.bubbles[i].frame.point
                center_j = self.bubbles[j].frame.point
                distance = distance_point_point(center_i, center_j)

                # get the radius of circumscribed circle, from the ellipse of each bubble
                ##------------------------------------------------------------------------------
                xsize_i = self.bubbles[i].xsize
                ysize_i = self.bubbles[i].ysize
                xsize_j = self.bubbles[j].xsize
                ysize_j = self.bubbles[j].ysize

                self.radius_i = max(xsize_i, ysize_i)
                self.radius_j = max(xsize_j, ysize_j)

                # create circumscribed circles
                ##------------------------------------------------------------------------------
                plane = Plane.from_frame(self.bubbles[j].frame)
                self.circles.append(Circle(plane, self.radius_j))

                # collision distance is the sum of the 2 radii, of the corresponding tiles
                ##------------------------------------------------------------------------------
                collision_distance = self.radius_i + self.radius_j

                # push distance is the collision_distance * effect_factor
                ##------------------------------------------------------------------------------
                push_distance = collision_distance * self.effect_factor

                # push vector pointing from j point to i point
                ##------------------------------------------------------------------------------
                if distance > push_distance:
                    continue

                push_vector_dir = center_i - center_j

                if not push_vector_dir.length == 0:
                    push_vector_dir.unitize()

                push_vector = push_vector_dir * (push_distance - distance) / 2
                # push_vector_j = push_vector_dir * (push_distance - distance) / 2

                # note: signs according to order of subtraction: push_vector_dir = center_i - center_j
                ##------------------------------------------------------------------------------
                total_moves[i] += push_vector
                collisions_count[i] += 1

                total_moves[j] -= push_vector
                collisions_count[j] += 1

        return total_moves, collisions_count

    def push_bubbles(self):
        total_moves, collisions_count = self.get_motion_vectors()
        self.sum_of_moves = sum([move.length for move in total_moves])

        for i in range(len(self.bubbles)):
            if collisions_count[i] == 0:
                continue
            # average -> sum of values / amount of values
            ##------------------------------------------------------------------------------
            average_move = total_moves[i] / collisions_count[i]
            self.bubbles[i].frame.point += average_move

            # project on the compas_surface (replace position with projected one)
            point = self.bubbles[i].frame.point
            frame, _ = self.generate_tile_frame_on_surface(point, flip_frame=False)
            self.bubbles[i].frame = frame


class Bubble(object):
    def __init__(self, frame, xsize, ysize):
        self.frame = frame
        self.xsize = xsize
        self.ysize = ysize
        if xsize == ysize:
            self.radius = xsize
        else:
            self.radius = None
        self.projected_curves = []

    @classmethod
    def assign_size_from_domain(cls, domain):
        size = random.uniform(domain[0], domain[1])
        size /= 2  # ellipse is instantiated with radius, not diameter!
        return size

    @classmethod
    def assign_size_discrete(cls, dimensions):
        diameter, thickness = random.choice(dimensions)
        radius = diameter / 2
        return radius, thickness

    @property
    def ellipse(self):
        shape = EllipseFrame(self.frame, self.ysize, self.xsize)
        return shape


class TileBrushLayer(BubblesFromCurveLayer):
    def __init__(
        self,
        rhino_brep,
        input_curve,
        division_num,
        tile_joint,
        tile_diameters,
        tile_thicknesses,
        effect_factor=0,
        **kwargs
    ):
        super(TileBrushLayer, self).__init__(
            rhino_brep, input_curve, division_num, tile_joint, effect_factor=effect_factor, **kwargs
        )
        self.tile_diameters = tile_diameters
        self.tile_thicknesses = tile_thicknesses

        self.tile_dimensions = zip(self.tile_diameters, self.tile_thicknesses)

    def generate(self):
        bubble_frames = self.generate_bubble_frames()
        tile_thicknesses = []
        self.tiles = []

        for frame in bubble_frames:
            bubble_radius, tile_thickness = Bubble.assign_size_discrete(self.tile_dimensions)
            bubble = Bubble(frame, bubble_radius, bubble_radius)
            tile_thicknesses.append(tile_thickness)
            self.bubbles.append(bubble)

        # Spread the bubbles in the area of the effect
        ##------------------------------------------------------------------------------
        self.relax_bubbles()

        brepface = self.rhino_brep.Faces[0]

        for bubble, tile_thickness in zip(self.bubbles, tile_thicknesses):
            _, uv_param = self.compas_surface.closest_point(bubble.frame.point, return_parameters=True)
            pointBrepFaceRelationship = brepface.IsPointOnFace(*uv_param)
            if pointBrepFaceRelationship == rg.PointFaceRelation.Exterior:
                continue

            round_tile = RoundTile(bubble.frame, bubble.radius * 2, thickness=tile_thickness, uv_param=uv_param)

            self.tiles.append(round_tile)


class AddTileLayer(GenLayer):
    def __init__(self, rhino_brep, points, tile_diameter, tile_thickness, **kwargs):
        super(AddTileLayer, self).__init__(rhino_brep, **kwargs)
        self.points = points
        self.tile_diameter = tile_diameter
        self.tile_thickness = tile_thickness
        self.tiles = []

    def generate(self):
        flip_frame = self.options["flip_frame"]
        for point in self.points:
            frame, uv_param = self.generate_tile_frame_on_surface(point, flip_frame)
            round_tile = RoundTile(frame, self.tile_diameter, self.tile_thickness, uv_param=uv_param)
            self.tiles.append(round_tile)
