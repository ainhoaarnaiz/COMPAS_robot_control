from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math as m
import random
from copy import copy
from copy import deepcopy

from compas.colors import Color
from compas.geometry import Frame
from compas.geometry import KDTree
from compas.geometry import Line
from compas.geometry import Plane
from compas.geometry import allclose
from compas_rhino.conversions import RhinoCurve
from compas_rhino.conversions import RhinoSurface
from compas_rhino.conversions import frame_to_rhino


class TileDesign(object):
    def __init__(self):
        self.layers = []
        self.tiles = []

    def add_layer(self, layer):
        self.tiles.extend(layer.tiles)
        self.layers.append(layer)
        self.pack()

    def combine_layers(self, layer_top, layer_bottom):
        """This function handles interactions between same layer-type and different layer-type stacking"""

        tiles_bottom_centers = [tile_bottom.base_frame.point for tile_bottom in layer_bottom.tiles]
        kdtree = KDTree(tiles_bottom_centers)

        indices_to_pop_bottom = []
        for tile_top in layer_top.tiles:
            nearest_neighbors = kdtree.nearest_neighbors(tile_top.base_frame.point, 15, True)
            for nearest_neighbor in nearest_neighbors:
                neighbor_center, neighbor_index, neighbor_dist = (
                    nearest_neighbor[0],
                    nearest_neighbor[1],
                    nearest_neighbor[2],
                )

                if neighbor_dist < (tile_top.diameter / 2):

                    # layer_bottom.tiles.pop(neighbor_index)
                    indices_to_pop_bottom.append(neighbor_index)

        indices_to_pop_bottom_sorted = sorted(indices_to_pop_bottom)

        print(list(reversed(indices_to_pop_bottom_sorted)))
        # pop things reversed, so you don't have a shifting-list-wrong-element-all-the-time-fuckup
        layer_bottom_copy = copy(layer_bottom)
        layer_bottom_copy.tiles = layer_bottom_copy.tiles[:]
        for neighbor_index in reversed(indices_to_pop_bottom_sorted):
            layer_bottom_copy.tiles.pop(neighbor_index)
        return layer_top, layer_bottom_copy


class Tile(object):
    def __init__(self, base_frame, thickness, uv_param, color):
        self.base_frame = base_frame
        self.thickness = thickness
        self.uv_param = uv_param
        self.color = color

    @property
    def pickup_frame(self):
        move_vector = self.base_frame.normal.unitized() * self.thickness
        pickup_frame = deepcopy(self.base_frame)
        pickup_frame.point += move_vector
        pickup_frame.xaxis *= -1
        return pickup_frame


class RoundTile(Tile):
    def __init__(self, base_frame, diameter, thickness, uv_param, color=Color.white(), tag=None):
        super(RoundTile, self).__init__(base_frame, thickness, uv_param, color)
        self.diameter = diameter
        self.tag = tag


class DesignLayer(object):
    def __init__(self, rhino_brep, options=None):
        self.options = options or {}
        self.rhino_brep = rhino_brep
        self.compas_surface = RhinoSurface.from_geometry(self.rhino_brep).to_compas()
        if self.compas_surface.u_domain[1] > 1 or self.compas_surface.v_domain[1] > 1:
            raise Exception("Your surface is not reparameterized.")
        self.tiles = []

    def generate_tile_frame_on_surface(self, point, flip_frame):
        _, uv_param = self.compas_surface.closest_point(point, return_parameters=True)
        frame = self.compas_surface.frame_at(uv_param[0], uv_param[1])
        if flip_frame == True:
            frame.xaxis *= -1
        return frame, uv_param

    def generate_tile_frames_on_projected_curve(self, curve, unit_size, return_params=True):
        _, adjusted_unit_size = self.max_divisions_per_curve(curve, unit_size, return_adjusted_unit_size=True)

        parameters = self.generate_params_on_curve(curve, unit_size=adjusted_unit_size, uniform=True, param_density=1)
        frames_and_uv_params = []

        for param in parameters:
            point_on_curve = curve.point_at(param)
            frame, uv_param = self.generate_tile_frame_on_surface(point_on_curve, flip_frame=False)

            if return_params:
                frames_and_uv_params.append((frame, uv_param))
            else:
                frames_and_uv_params.append(frame)

        return frames_and_uv_params

    def max_divisions_per_curve(self, curve, unit_size, return_adjusted_unit_size=False):
        if curve.length() < unit_size:
            if return_adjusted_unit_size:
                return 1, unit_size
            else:
                return 1
        max_divisions = int(curve.length() / unit_size)

        if return_adjusted_unit_size:
            adjusted_unit_size = curve.length() / max_divisions
            return max_divisions, adjusted_unit_size
        else:
            return max_divisions

    def generate_params_on_curve(
        self,
        input_curve,
        unit_size=None,
        division_mode="by_count",
        divisions_count=None,
        uniform=True,
        param_density=1,
    ):
        curve_params = []

        if not uniform:
            curve_param = 0
            domain_end = 1 + (unit_size / input_curve.length())

            while curve_param < domain_end:
                param_dist = random.uniform(unit_size, unit_size * param_density)

                # while loop condition
                # ------------------------------------------------------------------------------
                curve_param += param_dist / input_curve.length()

                # If curve param is greater than domain_end, the code should break
                # ------------------------------------------------------------------------------
                if curve_param > domain_end:
                    break
                curve_params.append(curve_param)

        else:
            if division_mode == "by_count":
                if not divisions_count:
                    divisions_count = self.max_divisions_per_curve(
                        input_curve, unit_size * param_density, return_adjusted_unit_size=False
                    )
                curve_params = input_curve.divide_by_count(divisions_count, return_points=False)
                # curve_params = input_curve.space(divisions_count)
            elif division_mode == "by_length":
                curve_params = input_curve.divide_by_length(unit_size * param_density, return_points=False)
                # print(curve_params)
            else:
                raise Exception("division_mode can only be 'by_count' or 'by_length'")

        return curve_params

    def generate_frames_on_params():
        pass

    def generate_isocurves(self, edge_params, flip_curves):
        isocurves = []
        for edge_param in edge_params:
            ## isocurve for the spacing of curves (isocurves, contours, etc)
            ##------------------------------------------------------------------------------
            if flip_curves:
                isocurve = self.compas_surface.v_isocurve(edge_param)
            else:
                isocurve = self.compas_surface.u_isocurve(edge_param)
            isocurves.append(isocurve)
        return isocurves

    def generate_contours(self, input_curves, unit_size, uniform, param_density, reverse_curve_params):
        if len(input_curves) == 1:
            return self.generate_contours_from_single_curve(input_curves[0], unit_size, uniform, param_density)

        elif len(input_curves) == 2:
            return self.generate_contours_from_two_curves(
                input_curves, unit_size, uniform, param_density, reverse_curve_params
            )

        else:
            raise Exception("generate_contours for more than 2 input curves is not implemented yet.")

    def generate_contours_from_single_curve(self, input_curve, unit_size, uniform, param_density):
        compas_contours_nested = []  # ! nested
        contour_frames = []

        divisions_count = self.max_divisions_per_curve(
            input_curve, unit_size * param_density, return_adjusted_unit_size=False
        )

        curve_params = self.generate_params_on_curve(
            input_curve,
            divisions_count=divisions_count,
            uniform=uniform,
            param_density=param_density,
        )

        for curve_param in curve_params:
            tangent = input_curve.tangent_at(curve_param)
            point_for_frame = input_curve.point_at(curve_param)
            contour_plane = Plane(point_for_frame, tangent)
            contour_frame = Frame.from_plane(contour_plane)
            contour_frames.append(contour_frame)

            rhino_plane = frame_to_rhino(contour_frame)
            contour_segments = self.rhino_brep.CreateContourCurves(self.rhino_brep, rhino_plane)

            compas_contour_segments = []
            for contour_segment in contour_segments:
                compas_contour_segment = RhinoCurve.from_geometry(contour_segment).to_compas()
                compas_contour_segments.append(compas_contour_segment)
            compas_contours_nested.append(compas_contour_segments)

        return compas_contours_nested, contour_frames

    def generate_contours_from_two_curves(self, input_curves, unit_size, uniform, param_density, reverse_curve_params):
        compas_contours_nested = []  # ! nested
        contour_frames = []
        curves_params_nested = []
        # print("accesed two curves")

        if input_curves[0].length() <= input_curves[1].length():
            input_curve, curve_number = input_curves[0], 0
        else:
            input_curve, curve_number = input_curves[1], 1
        divisions_count = self.max_divisions_per_curve(
            input_curve, unit_size * param_density, return_adjusted_unit_size=False
        )

        for curve in input_curves:
            curve_params = self.generate_params_on_curve(
                curve,
                divisions_count=divisions_count,
                uniform=uniform,
                param_density=param_density,
            )
            curves_params_nested.append(curve_params)

        if reverse_curve_params:
            curves_params_nested[0] = list(reversed(curves_params_nested[0]))  # oink

        self.contour_points = []
        self.lines = []
        for i in range(len(curves_params_nested[0])):
            point_on_curve_0 = input_curves[0].point_at(curves_params_nested[0][i])
            point_on_curve_1 = input_curves[1].point_at(curves_params_nested[1][i])
            vector = point_on_curve_1 - point_on_curve_0
            curve_param = curves_params_nested[curve_number][i]
            self.contour_points.append(point_on_curve_0)
            self.contour_points.append(point_on_curve_1)
            self.lines.append(Line(point_on_curve_0, point_on_curve_1))

            # Using compas's feature of correcting orthonoramlity of x and y axis, when creating a frame, ->
            # we will use tanget to approximate the yaxis of the contour frame
            tangent = input_curve.tangent_at(curve_param)
            point_for_frame = input_curve.point_at(curve_param)

            contour_plane = Plane(point_for_frame, vector)
            contour_frame = Frame.from_plane(contour_plane)
            rotated_frame_xaxis = contour_frame.normal
            rotated_frame_yaxis = rotated_frame_xaxis.cross(tangent)
            rotated_frame = Frame(contour_frame.point, rotated_frame_xaxis, rotated_frame_yaxis)

            contour_frames.append(rotated_frame)

            rhino_plane = frame_to_rhino(rotated_frame)
            contour_segments = self.rhino_brep.CreateContourCurves(self.rhino_brep, rhino_plane)

            compas_contour_segments = []
            for contour_segment in contour_segments:
                compas_contour_segment = RhinoCurve.from_geometry(contour_segment).to_compas()

                if allclose(compas_contour_segment.start, contour_frame.point, tol=0.01) or allclose(
                    compas_contour_segment.end, contour_frame.point
                ):
                    compas_contour_segments.append(compas_contour_segment)

            compas_contours_nested.append(compas_contour_segments)

        return compas_contours_nested, contour_frames

    def generate(self):
        raise Exception("This method is implemented in the child classes.")
