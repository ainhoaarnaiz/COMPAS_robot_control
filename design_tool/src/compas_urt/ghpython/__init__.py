from compas.artists import Artist
from compas.plugins import plugin
from compas_urt.design import RoundTile
from compas_urt.ghpython.roundtileartist import RoundTileArtist


@plugin(category="factories", requires=["Rhino"])
def register_artists():
    Artist.register(RoundTile, RoundTileArtist, context="Grasshopper")
