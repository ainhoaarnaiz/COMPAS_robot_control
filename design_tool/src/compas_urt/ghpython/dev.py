from compas_rhino import unload_modules


def reload_urt_modules():
    unload_modules("compas_urt.ghpython.roundtileartist")
    unload_modules("compas_urt.design")

    from compas.artists import Artist
    from compas_urt.ghpython.roundtileartist import RoundTileArtist
    from compas_urt.design import RoundTile

    for key in Artist.ITEM_ARTIST["Grasshopper"].keys():
        if str(key) == "<class 'compas_urt.design.RoundTile'>":
            del Artist.ITEM_ARTIST["Grasshopper"][key]

    Artist.register(RoundTile, RoundTileArtist, context="Grasshopper")
