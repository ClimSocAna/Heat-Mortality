import geojson
from shapely.geometry import Polygon, Point, MultiPolygon
import geopandas as gpd

def get_gdf():

    with open('./data/geodata/georef-germany-kreis.geojson') as f:
        gj = geojson.load(f)


    def process_coordinates(coord, polygons):
        if type(coord[0][0][0]) == float:
            p = Polygon(coord[0])
            for j in range(1, len(coord)):
                p = p.difference(Polygon(coord[j]))
            polygons.append(p)
        else:
            for c in coord:
                process_coordinates(c, polygons)

    kreise_shapes = []
    for i_gj in range(len(gj['features'])):
        coordinates = gj['features'][i_gj]['geometry']['coordinates']
        polygons = []
        process_coordinates(coordinates, polygons)
        kreise_shapes.append(polygons)

    kreise = [MultiPolygon(s) for s in kreise_shapes]
    kreise[118] = kreise[118].union(kreise[82])
    kreise.pop(82)
    d1 = {'krs_code': [gj['features'][i_gj]['properties']['krs_code'][0] for i_gj in range(len(gj['features'])) if i_gj != 82], 'geometry': kreise}
    gdf1 = gpd.GeoDataFrame(d1, crs="EPSG:3857")
    return gdf1.sort_values(by = 'krs_code').reset_index(drop=True)