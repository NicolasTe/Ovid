from functools import partial
from itertools import islice

import pandas as pd
from io import StringIO
from shapely.geometry import LineString, Point
from shapely.ops import transform

from dateutil.parser import parse

import pyproj


def parse_csv_repeating_headers(path, *args, **kwargs):
    contents = ""
    header = None
    with open(path, 'r', encoding='utf-8') as fi:
        for line in fi:
            if header is None:
                header = line
            elif header == line:
                continue

            contents += line
    file_buffer = StringIO(contents)

    return pd.read_csv(file_buffer, *args, **kwargs)


def parse_timestamp_feature(ts):
    try:
        return int(parse(ts).timestamp())
    except TypeError:
        return 0


def parse_map_feature_list(path):
    map_features = {}
    with open(path, 'r', encoding='utf-8') as fi:
        for line in islice(fi, 1, None):
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            k, v = parts

            if k not in map_features:
                map_features[k] = set()
            map_features[k].add(v)
    return map_features


def geom_dist(geo_str1, geo_str2, osm_type):
    if (type(geo_str1) == str) and (type(geo_str2) == str) and (geo_str1 != "-1") and (geo_str2 != "-1"):
        if osm_type == "node":
            return point_distance(geo_str1, geo_str2)
        elif osm_type == "way":
            return line_string_distance(geo_str1, geo_str2)
    return 0


def finish_row(current_row, edit_count, max_edits, no_features, current_mask_row):
    if edit_count >= max_edits:
        current_row = current_row[:(no_features * max_edits) + 1]
        current_mask_row = [False] * max_edits
    else:
        current_row = current_row + [0] * no_features * (max_edits - edit_count)
        current_mask_row = current_mask_row + [False] * (max_edits - edit_count)

    return current_row + current_mask_row


def geom_to_point(geom):
    lat, lon = map(float, geom.split())
    return Point(lat, lon)


def geom_to_linestring(geom):
    parts = geom.split(",")
    points = []
    for p in parts:
        lat, lon = map(float, p.split())
        points.append((lat, lon))
    if len(points) == 1:
        return Point(points[0][0], points[0][1])
    return LineString(points)


def geo_distance(geom1, geom2):
    project = partial(
        pyproj.transform,
        pyproj.Proj('EPSG:4326'),
        pyproj.Proj('EPSG:3857'))

    g_1 = transform(project, geom1)
    g_2 = transform(project, geom2)
    return g_1.distance(g_2)


def point_distance(current_geom, previous_geom):
    p_1 = geom_to_point(current_geom)
    p_2 = geom_to_point(previous_geom)

    if p_1.x == p_2.x and p_1.y == p_2.y:
        return 0

    return geo_distance(p_1, p_2)


def line_string_distance(current_geom, previous_geom):
    l_1 = geom_to_linestring(current_geom)
    l_2 = geom_to_linestring(previous_geom)
    return geo_distance(l_1, l_2)

