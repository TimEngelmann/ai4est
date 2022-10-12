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

def get_field_data(site:str, path_to_data:str) -> gpd.GeoDataFrame:
    """
    Loads the field data and creates a GeoDataFrame using the
    provided coordinates as latitude and longitude and filters
    for the provied site.
    """
    field_data = pd.read_csv(path_to_data + "field_data.csv")
    geo_field_data = gpd.GeoDataFrame(field_data, geometry=gpd.points_from_xy(field_data.lon, field_data.lat))
    return geo_field_data.loc[geo_field_data.site==site].copy()

#Dictionary mapping the names from the field data
#to the names from the wwf site data
wwf_name_from_site = {
    "Carlos Vera Guevara RGB" : "Vera Gevara Carlos",
    "Carlos Vera Arteaga RGB" : "Vera Arteaga Carlos",
    "Nestor Macias RGB" : "WWF Nestor Macias",
    "Manuel Macias RGB" : "Macias Guevara Manuel",
    "Flora Pluas RGB" : "Flora Puas 2017",
    "Leonor Aspiazu RGB" : "Aspiazu Mendoza leonor"
}


def get_wwf_site_data(site, path_to_data):
    """
    Loads wwf site data and restricts the data
    to the provided site
    """
    name = wwf_name_from_site[site]
    path = path_to_data + "wwf_ecuador/Merged_final_plots/Merged_final_plots.shp"
    site_data = gpd.read_file(path)

    return sites[sites.Name==name]

def create_boundary(site:str, path_to_data:str, convex_hull=True):
    """
    Creates a boundary for the plot at the specified site
    using either the collected field data or the site data
    provided by the wwf
    Inputs:
        site : The name of the site as specified in the field
            data
        path_to_data : The path to the reforstree folder
        convex_hull : Determines whether the convex hull
            of the field data or the site data is used to
            generate the boundary
    """

    if convex_hull:
        field_data = get_field_data(site, path_to_data)
        boundary = field_data.unary_union.convex_hull
        boundary = gpd.GeoSeries({"geometry" : boundary})
    else:
        boundary = get_wwf_boundary(site, path_to_data).boundary

    return boundary


def make_image(site, path_to_data, rotation=0.0):
    boundary = create_boundary(site, path_to_data)

    raster_path = path_to_data + f"/wwf_ecuador/RGB Orthomosaics/{site}.tif"
    with rasterio.open(raster_path) as raster:
        masked_img, _ = rasterio.mask.mask(raster, boundary, crop=True)

    if rotation != 0.0:
        raise NotImplementedError("Rotations not implemented")

    return masked_img

def test():
    name = "Manuel Macias RGB"
    path = "/home/jan/sem1/ai4good/data/reforestree/"
    img = make_image(name, path)
    pil_img = im.fromarray(img.T)
    pil_img.save("test.png")

if __name__ == "__main__":
    test()