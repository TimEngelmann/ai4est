import numpy as np

gps_error_rgb = {"Flora Pluas RGB": [0.25, 0.66],
                "Nestor Macias RGB": [0.6, 0.53],
                "Manuel Macias RGB": [0.69, 0.30],
                "Leonor Aspiazu RGB": [0.47, 0.45],
                "Carlos Vera Arteaga RGB": [0.26, 0.59],
                "Carlos Vera Guevara RGB": [0.27, 0.65]}
gsd_scale_rgb = {"Flora Pluas RGB": 0.97,
            "Nestor Macias RGB": 1.10,
            "Manuel Macias RGB": 1.13,
            "Leonor Aspiazu RGB": 1.06,
            "Carlos Vera Arteaga RGB": 1.04,
            "Carlos Vera Guevara RGB": 1.19}
gps_error_ocn = {"Flora Pluas RGB": [2.20, 2.98],
        "Nestor Macias RGB": [2.36, 2.74],
        "Manuel Macias RGB": [4.15, 4.07],
        "Leonor Aspiazu RGB": [1.53, 0.82],
        "Carlos Vera Arteaga RGB": [1.17, 1.77],
        "Carlos Vera Guevara RGB": [1.19, 1.84]}
gsd_scale_ocn = {"Flora Pluas RGB": 2.08,
            "Nestor Macias RGB": 2.28,
            "Manuel Macias RGB": 2.39,
            "Leonor Aspiazu RGB": 2.27,
            "Carlos Vera Arteaga RGB": 2.25,
            "Carlos Vera Guevara RGB": 2.98}

def get_gps_error():
    gps_error = {}
    for site in gps_error_rgb.keys():
        gsd = gsd_scale_rgb[site]
        rgb = np.array([gps_error_rgb[site][0] * 100 /gsd, gps_error_rgb[site][1] * 100 /gsd])
        gsd = gsd_scale_ocn[site]
        ocn = np.array([gps_error_ocn[site][0] * 100 /gsd, gps_error_ocn[site][1] * 100 /gsd])
        gps_error[site] = rgb**2 + ocn**2
    return gps_error