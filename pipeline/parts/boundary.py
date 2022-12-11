"""
Script which provides functions that take RGB drone imagery of
a site as well as labeled field data and provide a masked
image, showing only the plot for which carbon is measured.
"""
import rasterio
import rasterio.mask
import geopandas as gpd
import pandas as pd
import alphashape

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


def get_wwf_boundary(site, path_to_data):
    """
    Loads wwf site data and restricts the data
    to the provided site
    """
    name = wwf_name_from_site[site]
    path = path_to_data + "wwf_ecuador/Merged_final_plots/Merged_final_plots.shp"
    site_data = gpd.read_file(path)

    return site_data[site_data.Name==name].geometry


def create_boundary(site:str, path_to_data:str, shape="convex_hull"):
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

    if shape == "convex_hull":
        field_data = get_field_data(site, path_to_data)
        boundary = field_data.unary_union.convex_hull
        boundary = gpd.GeoSeries({"geometry" : boundary})
    elif shape == "alpha_shape":
        field_data = get_field_data(site, path_to_data)
        alpha_shape = alphashape.alphashape(field_data[['lon', 'lat']].values, 15000)
        boundary = gpd.GeoSeries({"geometry" : alpha_shape})
    else:
        boundary = get_wwf_boundary(site, path_to_data)

    return boundary

def main(path):
    "Creates boundaries and saves them to shape file"

    fd = pd.read_csv(path + "field_data.csv")
    sites = fd.site.unique()

    ch = []
    wwf = []
    alpha = []

    for site  in sites:
        ch.append(create_boundary(site, path, shape="convex_hull").item())
        wwf.append(create_boundary(site, path, shape="wwf").item())
        alpha.append(create_boundary(site, path, shape="alpha").item())

    ch = gpd.GeoDataFrame({"site": sites}, geometry=ch)
    wwf = gpd.GeoDataFrame({"site": sites}, geometry=wwf)
    alpha = gpd.GeoDataFrame({"site": sites}, geometry=alpha)

    return ch, wwf, alpha




if __name__ == "__main__":
    path = input("Path to reforestree \n >")
    ch, wwf, alpha = main(path)

    out_path = input("Output path \n >")
    ch.to_file(out_path + "convex_boundary.shp")
    wwf.to_file(out_path + "provided_boundary.shp")
    alpha.to_file(out_path + "alphashape_boundary.shp")