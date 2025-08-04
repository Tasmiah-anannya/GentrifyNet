"""
Description:
    1. Generates a combined building list CSV for a city and year by merging all tract-level building CSV files, extracting GEOID, ensuring columns, and filling missing heights.
    2. Rasterizes 2D polygonal building shapes from CSV to binary images.
    3. Encodes building types as one-hot arrays.
    
Expected input folder ('Building list'):
- Contains tract-level building CSVs, named like 'buildings_tract_*.csv'. * represents tract no.
- Each CSV corresponds to one tract and has columns: 'geometry' (WKT), 'height', 'nycdoitt:bin'.
- Files can have missing columns; these will be filled as needed in the script.

"""

import argparse
import os
import numpy as np
import pandas as pd
from shapely import wkt, affinity
import rasterio.features
from tqdm import trange
import glob

CITY = "Chicago"  #Chicago, Brooklyn, Los Angeles for our case
YEAR = "2013"   #2013, 2018 and 2020 for two configurations from our paper
INPUT_FOLDER = f"/data/folder/{CITY}/Building Footprint {YEAR}_{CITY}/"   #this folder contains csv files contain tract-level CSV files (e.g. 'buildings_tract_*.csv'), * is the tract_no, each representing building footprints and attributes for a single tract.
OUTPUT_CSV = f"/data/folder/{CITY}/Building Footprint {YEAR}_{CITY}/Building list/combined_building_list.csv"
TRACT_PATTERN = "buildings_tract_*.csv"  

# ------------- Create combined_building_list.csv -------------
def generate_combined_building_list(input_folder, output_csv, tract_pattern="buildings_tract_*.csv"):
    df_list = []
    csv_files = glob.glob(os.path.join(input_folder, tract_pattern))
    if not csv_files:
        print(f"No files found with pattern {tract_pattern} in {input_folder}")
        return

    for csv_file in csv_files:
        print(f"Reading: {csv_file}")
        base_name = os.path.basename(csv_file)
        parts = base_name.split('_')
        geoid = parts[2] if len(parts) >= 3 else np.nan

        df = pd.read_csv(csv_file)

        if 'height' in df.columns:
            df['height'] = pd.to_numeric(df['height'], errors='coerce')
        required_cols = ['geometry', 'height', 'nycdoitt:bin']
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan
        df.insert(0, "GEOID", geoid)
        df = df[['GEOID', 'geometry', 'height', 'nycdoitt:bin']]
        df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)
    avg_heights = combined_df.groupby("GEOID")["height"].mean()

    def fill_height(row):
        if pd.isnull(row["height"]):
            return avg_heights.get(row["GEOID"], np.nan)
        else:
            return row["height"]

    combined_df["height"] = combined_df.apply(fill_height, axis=1)
    combined_df.to_csv(output_csv, index=False)
    print(f"Combined CSV written to: {output_csv}")
    print(combined_df.head())
    


# ------------- Building Type Definitions according to OSM -------------
accommodation_tags = {
    "yes","apartments","barracks","bungalow","cabin","detached","annexe","dormitory","farm","ger",
    "hotel","house","houseboat","residential","semidetached_house","static_caravan","stilt_house",
    "terrace","tree_house","trullo"
}
commercial_tags = {"commercial","industrial","kiosk","office","retail","supermarket","warehouse"}
religious_tags = {"religious","cathedral","chapel","church","kingdom_hall","monastery","mosque",
                  "presbytery","shrine","synagogue","temple"}
amenity_tags = {"bakehouse","bridge","civic","college","fire_station","government","gatehouse","hospital",
                "kindergarten","museum","public","school","toilets","train_station","transportation","university"}
agriculture_tags = {"barn","conservatory","cowshed","farm_auxiliary","greenhouse","slurry_tank","stable","sty","livestock"}
sports_tags = {"grandstand","pavilion","riding_hall","sports_hall","sports_centre","stadium"}
storage_tags = {"allotment_house","boathouse","hanger","hut","shed"}
cars_tags = {"carport","garage","garages","parking"}
power_building_tags = {"digester","service","tech_cab","transformer_tower","water_tower","silo","storage_tank"}
other_building_tags = {
    "beach_hut","bunker","castle","construction","container","guardhouse","military","outbuilding",
    "pagoda","quonset_hut","roof","ruins","ship","tent","tower","triumphal_arch","windmill"
}
group_names = [
    "accommodation", "commercial", "religious", "amenity",
    "agriculture", "sports", "storage", "cars",
    "power_building", "other_building"
]
group_to_tags = {
    "accommodation": accommodation_tags,
    "commercial": commercial_tags,
    "religious": religious_tags,
    "amenity": amenity_tags,
    "agriculture": agriculture_tags,
    "sports": sports_tags,
    "storage": storage_tags,
    "cars": cars_tags,
    "power_building": power_building_tags,
    "other_building": other_building_tags
}
onehot_dim = len(group_names)  # 10

# ----------- Building Type Utilities -----------
def get_group_index(b_type):
    """Return index (0..9) for building type. Default to 'other_building'."""
    if not isinstance(b_type, str):
        b_type = ""
    b_lower = b_type.lower()
    for idx, group in enumerate(group_names):
        if b_lower in group_to_tags[group]:
            return idx
    return group_names.index("other_building")

def group_to_onehot(g_idx):
    """Convert group index to one-hot vector."""
    arr = np.zeros(onehot_dim, dtype=np.float64)
    arr[g_idx] = 1.0
    return arr

# ----------- Load CSV Data -----------
def load_building_list(csv_file):
    """
    Loads the building CSV file and parses WKT geometry.
    Returns: list of dicts (building_list)
    """
    df = pd.read_csv(csv_file)
    if 'geometry' not in df.columns:
        raise ValueError("CSV must have a 'geometry' column in WKT format.")
    df['shape'] = df['geometry'].apply(wkt.loads)
    return df.to_dict('records')

# ----------- Rasterization Function -----------
def rasterize_buildings(building_list, out_dir, rotation=True, batch_size=10000):
    """
    Rasterizes each building polygon to 224x224 image and saves results.
    - Stores image arrays and per-building features (rotation, height, type).
    - Saves as .npz files in out_dir.
    """
    image_out_path = os.path.join(out_dir, 'building_raster.npz')
    rot_type_out_path = os.path.join(out_dir, 'building_rot_type.npz')
    N = len(building_list)
    print(f"Total buildings: {N}")
    
    images = np.zeros((N, 224, 224), dtype=np.uint8)
    rot_type_dim = 3 + onehot_dim  # [cos, sin, height] + 10-dim onehot
    rot_type_arr = np.zeros((N, rot_type_dim), dtype=np.float64)

    for start in trange(0, N, batch_size, desc="Processing batches"):
        end = min(start + batch_size, N)
        for i in range(start, end):
            building = building_list[i]
            polygon = building['shape']
            b_type = building.get('building', "")
            g_idx = get_group_index(b_type)
            oh_vec = group_to_onehot(g_idx)

            # If geometry is not a polygon but a point, buffer it
            if polygon.geom_type not in ['Polygon', 'MultiPolygon']:
                if polygon.geom_type == 'Point':
                    polygon = polygon.buffer(1.0)
                else:
                    continue

            cosp, sinp = 0.0, 0.0
            if rotation:
                rectangle = polygon.minimum_rotated_rectangle
                xc = polygon.centroid.x
                yc = polygon.centroid.y
                rec_x, rec_y = zip(*rectangle.exterior.coords)
                top = np.argmax(rec_y)
                top_left = top - 1 if top > 0 else 3
                top_right = top + 1 if top < 3 else 0
                x0, y0 = rec_x[top], rec_y[top]
                x1, y1 = rec_x[top_left], rec_y[top_left]
                x2, y2 = rec_x[top_right], rec_y[top_right]
                d1 = np.linalg.norm([x0 - x1, y0 - y1])
                d2 = np.linalg.norm([x0 - x2, y0 - y2])
                if d1 > d2:
                    cosp = (x1 - x0) / d1
                    sinp = (y0 - y1) / d1
                else:
                    cosp = (x2 - x0) / d2
                    sinp = (y0 - y2) / d2
                matrix = (cosp, -sinp, 0.0,
                          sinp,  cosp, 0.0,
                          0.0,   0.0,  1.0,
                          xc - xc * cosp + yc * sinp, yc - xc * sinp - yc * cosp, 0.0)
                polygon = affinity.affine_transform(polygon, matrix)

            # Save rotation, height, one-hot
            rot_type_arr[i, 0] = cosp
            rot_type_arr[i, 1] = sinp
            rot_type_arr[i, 2] = building.get("height", np.nan)
            rot_type_arr[i, 3:3+onehot_dim] = oh_vec

            # Rasterize
            min_x, min_y, max_x, max_y = polygon.bounds
            length_x = max_x - min_x
            length_y = max_y - min_y
            if length_x > length_y:
                diff = length_x - length_y
                min_y -= diff / 2
                max_y += diff / 2
                length = length_x
            else:
                diff = length_y - length_x
                min_x -= diff / 2
                max_x += diff / 2
                length = length_y
            min_x -= length * 0.1
            min_y -= length * 0.1
            max_x += length * 0.1
            max_y += length * 0.1

            transform = rasterio.transform.from_bounds(min_x, min_y, max_x, max_y, 224, 224)
            image = rasterio.features.rasterize([polygon], out_shape=(224, 224), transform=transform)
            images[i] = image

    # Save compressed versions
    np.savez_compressed(image_out_path, images_mem=images)
    np.savez_compressed(rot_type_out_path, rot_type_mem=rot_type_arr)
    print(f"Saved: {image_out_path}, {rot_type_out_path}")

    return images, rot_type_arr

def main():
    # Combine tract CSVs into a single building list
    generate_combined_building_list(INPUT_FOLDER, OUTPUT_CSV, TRACT_PATTERN)

    # Rasterize buildings and encode types
    out_dir = INPUT_FOLDER   
    print("Loading building data...")
    building_list = load_building_list(OUTPUT_CSV)
    print(f"Loaded {len(building_list)} buildings.")
    print("Rasterizing and encoding buildings...")
    rasterize_buildings(building_list, out_dir=out_dir, rotation=True, batch_size=10000)
    print("Done.")

if __name__ == '__main__':
    main()
