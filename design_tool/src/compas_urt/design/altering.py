from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import copy

from compas.colors import Color
from compas.datastructures import Mesh
from compas.geometry import KDTree
from compas.geometry import distance_point_point

from compas_urt.design import DesignLayer


class AltLayer(DesignLayer):
    def alter(self):
        pass


class ImageLayer(AltLayer):
    def __init__(self, rhino_brep, loaded_image, **kwargs):
        super(ImageLayer, self).__init__(rhino_brep, **kwargs)
        self.loaded_image = loaded_image
        self.meshes = []

    def alter(self, layer, nu, nv, flip_frame, available_colors, return_meshes):
        copied_layer = copy(layer)
        tiles = copied_layer.tiles

        color_quads, meshes = self.create_colored_quads(nu, nv, return_meshes)
        self.meshes = meshes
        self.assign_color(tiles, color_quads, flip_frame, available_colors)

        return copied_layer

    def assign_color(self, tiles, color_quads, flip_frame, available_colors):
        sorted_color_quads = sorted(color_quads, key=self.uv_sorting)
        sorted_uvparams = [(self.uv_sorting(quad)[0], self.uv_sorting(quad)[1], 0) for quad in sorted_color_quads]

        kdtree = KDTree(sorted_uvparams)

        if available_colors:
            available_colors_tuples = [(c.R / 255, c.G / 255, c.B / 255, c.A / 255) for c in available_colors]
            kdtree_color = KDTree(available_colors_tuples)

        for tile in tiles:
            _, uv_params = self.generate_tile_frame_on_surface(tile.base_frame.point, flip_frame)
            u, v = uv_params
            neighbor = kdtree.nearest_neighbor((u, v, 0))
            uv_param, color_quad_index, param_distance = neighbor
            color_quad = sorted_color_quads[color_quad_index]
            if "color" in color_quad:
                if available_colors:
                    existing_color = color_quad["color"]
                    closest_color, closest_color_index, closest_color_dist = kdtree_color.nearest_neighbor(
                        (existing_color.r, existing_color.g, existing_color.b, existing_color.a)
                    )
                    closest_color = Color(*closest_color)
                    tile.color = closest_color
                    tile.tag = closest_color_index
                else:
                    tile.color = color_quad["color"]

    def uv_sorting(self, quad):
        u = quad["u_domain"][0]
        v = quad["v_domain"][0]
        return u, v

    def _get_average_colors(self, target_length_x, target_length_y):
        pixel_per_unit_factor_width = int(self.loaded_image.width / target_length_x)
        pixel_per_unit_factor_height = int(self.loaded_image.height / target_length_y)

        avg_colors = []
        for ix in range(target_length_x):
            for iy in range(target_length_y):
                unit_start_ix = int(ix * pixel_per_unit_factor_width)
                unit_end_iy = int(iy * pixel_per_unit_factor_height)

                all_r, all_g, all_b, all_a = [], [], [], []

                for i in range(unit_start_ix, unit_start_ix + pixel_per_unit_factor_width):
                    for j in range(unit_end_iy, unit_end_iy + pixel_per_unit_factor_height):
                        color = self.loaded_image.pixels[i, j]
                        all_r.append(color.r)
                        all_g.append(color.g)
                        all_b.append(color.b)
                        all_a.append(color.a)

                avg_color = Color(sum(all_r) / len(all_r), sum(all_g) / len(all_g), sum(all_b) / len(all_b))
                avg_color.a = sum(all_a) / (len(all_a))
                avg_colors.append(avg_color)

        return avg_colors

    # ------------------------------------------------------------------------------------
    def _surface_to_compas_quads(self, compas_surface, nu, nv=None):
        nv = nv or nu

        # trying to get the longest isocurve..
        max_u_edge = compas_surface.u_isocurve(0)
        max_v_edge = compas_surface.v_isocurve(0)

        for i in range(50):
            temp_u = compas_surface.u_isocurve(i / 50)
            temp_v = compas_surface.v_isocurve(i / 50)
            if temp_u.length() > max_u_edge.length():
                max_u_edge = temp_u
            if temp_v.length() > max_v_edge.length():
                max_v_edge = temp_v

        u_params = list(max_v_edge.divide_by_count(nu, return_points=False))
        v_params = list(max_u_edge.divide_by_count(nv, return_points=False))

        quads = []
        for i, u in enumerate(u_params):
            for j, v in enumerate(v_params[:-1]):
                a = compas_surface.point_at(u_params[i % len(u_params)], v_params[j])
                b = compas_surface.point_at(u_params[(i + 1) % len(u_params)], v_params[j])
                c = compas_surface.point_at(u_params[(i + 1) % len(u_params)], v_params[(j + 1)])
                d = compas_surface.point_at(u_params[i % len(u_params)], v_params[(j + 1)])

                u_domain = (u_params[i % len(u_params)], u_params[(i + 1) % len(u_params)])
                v_domain = (v_params[j], v_params[(j + 1)])

                quads.append(dict(vertices=[a, b, c, d], u_domain=u_domain, v_domain=v_domain))

        return quads

    # ------------------------------------------------------------------------------------

    def create_colored_quads(self, nu, nv, return_meshes=False):
        """Creates colored quad information from the image.

        Parameters
        ----------
        nu : int
                Divisions in the U direction.
        nv : int
            Divisions in the V direction.
        return_meshes : bool, optional
            If True, return the a list of quad information dictionaries,
            and the meshes corresponding to those quads.
            If False, return only the list of quad information dictionaries.

        Returns
        -------
        list[dict] | tuple[list[dict], list[:class:`compas.datastructures.Mesh`]]
            If `return_meshes` is False, the list of quad information dictionaries.
            If `return_meshes` is True, a list of meshes in addition to the quad dictionaries.
        """
        quads = self._surface_to_compas_quads(self.compas_surface, nu, nv)
        colors = self._get_average_colors(nu, nv)

        for quad, color in zip(quads, colors):
            quad["color"] = color

        meshes = []
        if return_meshes:
            for quad in quads:
                if "color" in quad:
                    mesh = Mesh.from_vertices_and_faces(quad["vertices"], [[0, 1, 2, 3]])
                    mesh.attributes["u_domain"] = quad["u_domain"]
                    mesh.attributes["v_domain"] = quad["v_domain"]
                    mesh.attributes["color"] = quad["color"]
                    meshes.append(mesh)
        return quads, meshes


class AttractorsLayer(AltLayer):
    def __init__(self, rhino_brep, **kwargs):
        super(AttractorsLayer, self).__init__(rhino_brep, **kwargs)

    def get_affected_tiles_multiple_curves(self, layer, input_curves, effect_factor):

        affected_tiles_indices = []
        for curve in input_curves:
            affected_tiles_indices_per_curve, _ = self.get_affected_tiles_and_distances_single_curve(
                layer, curve, effect_factor
            )
            affected_tiles_indices.extend(affected_tiles_indices_per_curve)
        affected_tiles_indices = set(affected_tiles_indices)

        return affected_tiles_indices

    def get_affected_tiles_and_distances_single_curve(self, layer, input_curve, effect_factor):
        affected_tiles_indices = []
        distances = []
        for tile_index, tile in enumerate(layer.tiles):
            tile_centroid = tile.base_frame.point

            pt_on_curve = input_curve.closest_point(tile_centroid)
            dist = distance_point_point(tile_centroid, pt_on_curve)

            if dist >= effect_factor:
                continue
            else:
                distances.append(dist)
                affected_tiles_indices.append(tile_index)

        return affected_tiles_indices, distances


class TileColoringLayer(AttractorsLayer):
    def __init__(self, rhino_brep, **kwargs):
        super(TileColoringLayer, self).__init__(rhino_brep, **kwargs)

    def alter(self, color, layer, input_curves, effect_factor):
        affected_tiles_indices, distances = self.get_affected_tiles(layer, input_curves, effect_factor)
        layer_copy = copy(layer)
        layer_copy.tiles = layer_copy.tiles[:]
        for index in affected_tiles_indices:
            layer_copy.tiles[index].color = color


class VectorFieldLayer(AltLayer):
    def alter(self):
        pass
