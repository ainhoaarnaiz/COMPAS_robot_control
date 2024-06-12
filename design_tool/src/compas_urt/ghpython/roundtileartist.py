from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import Rhino.Geometry as rg
from compas_ghpython.artists import GHArtist
from compas_rhino.conversions import frame_to_rhino


class RoundTileArtist(GHArtist):
    """Artist for drawing round tiles.

    Parameters
    ----------
    RoundTile : :class:`~compas_urt.design.RoundTile`
        A COMPAS round tile.
    **kwargs : dict, optional
        Additional keyword arguments.
        See :class:`~compas_ghpython.artists.GHArtist` for more info.

    """

    def __init__(self, round_tile, **kwargs):
        super(RoundTileArtist, self).__init__(**kwargs)
        self.round_tile = round_tile

    def draw(self):
        """Draw the round tile.

        Returns
        -------
        :rhino:`Rhino.Geometry.Cylinder`

        """

        plane = frame_to_rhino(self.round_tile.base_frame)
        rhino_circle = rg.Circle(plane, (self.round_tile.diameter / 2))
        return rg.Cylinder(rhino_circle, self.round_tile.thickness).ToBrep(True, True)
