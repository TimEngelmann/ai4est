"""
Script which provides functions that take RGB drone imagery of
a site as well as labeled field data and provide a masked
image, showing only the plot for which carbon is measured.
"""

import rasterio
import rasterio.mask
import geopandas as gpd
import pandas as pd
import PIL.Image as im

def get_path_to_data():
    return "/home/jan/sem1/ai4good/data/reforestree/"

def get_raster_path(site):
    return get_path_to_data() + f"wwf_ecuador/RGB Orthomosaics/{site}.tif"

def get_path_to_field_data():
    return get_path_to_data() + "field_data.csv"

def get_field_data(site:str) -> gpd.GeoDataFrame:
    field_data = pd.read_csv(get_path_to_field_data())
    geo_field_data = gpd.GeoDataFrame(field_data, geometry=gpd.points_from_xy(field_data.lon, field_data.lat))
    return geo_field_data.loc[geo_field_data.site==site].copy()

wwf_name_from_site = {
    "Carlos Vera Guevara RGB" : "Vera Gevara Carlos",
    "Carlos Vera Arteaga RGB" : "Vera Arteaga Carlos",
    "Nestor Macias RGB" : "WWF Nestor Macias",
    "Manuel Macias RGB" : "Macias Guevara Manuel",
    "Flora Pluas RGB" : "Flora Puas 2017",
    "Leonor Aspiazu RGB" : "Aspiazu Mendoza leonor"
}


def get_wwf_site_data(site):
    name = wwf_name_from_site_id[site]
    path = get_path_to_data() + "wwf_ecuador/Merged_final_plots/Merged_final_plots.shp"
    site_data = gpd.read_file(path)

    return sites[sites.Name==name]

def create_boundary(site, convex_hull=True):

    if convex_hull:
        field_data = get_field_data(site)
        boundary = field_data.unary_union.convex_hull
        boundary = gpd.GeoSeries({"geometry" : boundary})
    else:
        boundary = get_wwf_boundary(site).boundary

    return boundary


def make_image(site, rotation=0.0):
    boundary = create_boundary(site)

    with rasterio.open(get_raster_path(site)) as raster:
        masked_img, _ = rasterio.mask.mask(raster, boundary, crop=True)

    if rotation != 0.0:
        raise NotImplementedError("Rotations not implemented")

    return masked_img

def test():
    name = "Manuel Macias RGB"
    img = make_masked_image(name)
    pil_img = im.fromarray(img.T)
    pil_img.save("test.png")

if __name__ == "__main__":
    test()
