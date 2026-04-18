# -*- coding: utf-8 -*-
import re
from datetime import datetime
import requests
import time
import shutil
import itertools
from PIL import Image
from io import BytesIO
import os
import numpy as np
from skimage import io

def _panoids_url(lat, lon):
    url = "https://maps.googleapis.com/maps/api/js/GeoPhotoService.SingleImageSearch?pb=!1m5!1sapiv3!5sUS!11m2!1m1!1b0!2m4!1m2!3d{0:}!4d{1:}!2d50!3m10!2m2!1sen!2sGB!9m1!1e2!11m4!1m3!1e2!2b1!3e2!4m10!1e1!1e2!1e3!1e4!1e8!1e6!5m1!1e2!6m1!1e2&callback=_xdc_._v2mub5"
    return url.format(lat, lon)


def _panoids_data(lat, lon, proxies=None):
    url = _panoids_url(lat, lon)
    return requests.get(url, proxies=None)


def panoids(lat, lon, closest=False, disp=False, proxies=None):

    resp = _panoids_data(lat, lon)

    # Get all the panorama IDs and coordinates to be ordered like:
    # 2015
    # XXXX
    # XXXX
    # 2012
    # 2013
    # 2014

    pans = re.findall(r'\[[0-9]+,"(.+?)"\].+?\[\[null,null,(-?[0-9]+.[0-9]+),(-?[0-9]+.[0-9]+)', resp.text)
    pans = [{
        "panoid": p[0],
        "lat": float(p[1]),
        "lon": float(p[2])} for p in pans]  # Convert to floats

    # Remove duplicate panoramas
    pans = [p for i, p in enumerate(pans) if p not in pans[:i]]

    if disp:
        for pan in pans:
            print(pan)

    dates = re.findall(r'([0-9]?[0-9]?[0-9])?,?\[(20[0-9][0-9]),([0-9]+)\]', resp.text)
    dates = [list(d)[1:] for d in dates]  # Convert to lists and drop the index

    if len(dates) > 0:
        dates = [[int(v) for v in d] for d in dates]

        dates = [d for d in dates if d[1] <= 12 and d[1] >= 1]

        year, month = dates.pop(-1)
        pans[0].update({'year': year, "month": month})
        dates.reverse()
        for i, (year, month) in enumerate(dates):
            pans[-1-i].update({'year': year, "month": month})

    def func(x):
        if 'year'in x:
            return datetime(year=x['year'], month=x['month'], day=1)
        else:
            return datetime(year=3000, month=1, day=1)
    pans.sort(key=func)

    if closest:
        return [pans[i] for i in range(len(dates))]
    else:
        return pans


def tiles_info(panoid, zoom=5):
    image_url = "http://cbk0.google.com/cbk?output=tile&panoid={}&zoom={}&x={}&y={}"

    coord = list(itertools.product(range(26), range(13)))

    tiles = [(x, y, "%s_%dx%d.jpg" % (panoid, x, y), image_url.format(panoid, zoom, x, y)) for x, y in coord]

    return tiles


def download_tiles(tiles, directory, disp=False):

    for i, (x, y, fname, url) in enumerate(tiles):

        if disp and i % 20 == 0:
            print("Image %d / %d" % (i, len(tiles)))

        while True:
            try:
                response = requests.get(url, stream=True)
                break
            except requests.ConnectionError:
                print("Connection error. Trying again in 2 seconds.")
                time.sleep(2)

        with open(directory + '/' + fname, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        del response


def stich_tiles(panoid, tiles, directory, final_directory):
    tile_width = 512
    tile_height = 512

    panorama = Image.new('RGB', (26*tile_width, 13*tile_height))

    for x, y, fname, url in tiles:

        fname = directory + "/" + fname
        tile = Image.open(fname)

        panorama.paste(im=tile, box=(x*tile_width, y*tile_height))

        del tile

    panorama.save(final_directory + ("/%s.jpg" % panoid))
    del panorama
    


def download_panorama_v3(panoid, zoom=5, disp=False):
    tile_width = 512
    tile_height = 512
    # img_w, img_h = int(np.ceil(416*(2**zoom)/tile_width)*tile_width), int(np.ceil(416*( 2**(zoom-1) )/tile_width)*tile_width)
    img_w, img_h = 416*(2**zoom), 416*( 2**(zoom-1) )
    tiles = tiles_info( panoid, zoom=zoom)
    valid_tiles = []
    # function of download_tiles
    for i, tile in enumerate(tiles):
        x, y, fname, url = tile
        if disp and i % 20 == 0:
            print("Image %d / %d" % (i, len(tiles)))
        if x*tile_width < img_w and y*tile_height < img_h: # tile is valid
            # Try to download the image file
            while True:
                try:
                    response = requests.get(url, stream=True)
                    break
                except requests.ConnectionError:
                    print("Connection error. Trying again in 2 seconds.")
                    time.sleep(2)
            valid_tiles.append( Image.open(BytesIO(response.content)) )
            del response
            
    # function to stich
    panorama = Image.new('RGB', (img_w, img_h))
    i = 0
    for x, y, fname, url in tiles:
        if x*tile_width < img_w and y*tile_height < img_h: # tile is valid
            tile = valid_tiles[i]
            i+=1
            panorama.paste(im=tile, box=(x*tile_width, y*tile_height))
    return np.array(panorama)

def download_panorama_v1(panoid, zoom=5, disp=False, directory='temp'):
    tiles = tiles_info( panoid, zoom=zoom)
    if not os.path.exists(directory):
        os.makedirs( directory )
    # function of download_tiles
    for i, (x, y, fname, url) in enumerate(tiles):

        if disp and i % 20 == 0:
            print("Image %d / %d" % (i, len(tiles)))

        # Try to download the image file
        while True:
            try:
                response = requests.get(url, stream=True)
                break
            except requests.ConnectionError:
                print("Connection error. Trying again in 2 seconds.")
                time.sleep(2)
        with open(directory + '/' + fname, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        del response
    tile_width = 512
    tile_height = 512

    panorama = Image.new('RGB', (26*tile_width, 13*tile_height))

    for x, y, fname, url in tiles:
        fname = directory + "/" + fname
        tile = Image.open(fname)
        panorama.paste(im=tile, box=(x*tile_width, y*tile_height))
        del tile
    delete_tiles( tiles, directory )
    return np.array(panorama)

def download_panorama_v2(panoid, zoom=5, disp=False, directory='temp'):

    img_w, img_h = 416*(2**zoom), 416*( 2**(zoom-1) )
    tile_width = 512
    tile_height = 512
    
    tiles = tiles_info( panoid, zoom=zoom)
    valid_tiles = []
    if not os.path.exists(directory):
        os.makedirs( directory )
    for i, tile in enumerate(tiles):
        x, y, fname, url = tile
        if disp and i % 20 == 0:
            print("Image %d / %d" % (i, len(tiles)))
        if x*tile_width < img_w and y*tile_height < img_h: # tile is valid
            valid_tiles.append(tile)
            # Try to download the image file
            while True:
                try:
                    response = requests.get(url, stream=True)
                    break
                except requests.ConnectionError:
                    print("Connection error. Trying again in 2 seconds.")
                    time.sleep(2)
            with open(directory + '/' + fname, 'wb') as out_file:
                shutil.copyfileobj(response.raw, out_file)
            del response
            
    panorama = Image.new('RGB', (img_w, img_h))
    for x, y, fname, url in tiles:
        if x*tile_width < img_w and y*tile_height < img_h: # tile is valid
            fname = directory + "/" + fname
            tile = Image.open(fname)
            panorama.paste(im=tile, box=(x*tile_width, y*tile_height))
            del tile
    delete_tiles( valid_tiles, directory )
    return np.array(panorama)

def delete_tiles(tiles, directory):
    for x, y, fname, url in tiles:
        os.remove(directory + "/" + fname)


def api_download(panoid, heading, flat_dir, key, width=640, height=640,
                 fov=120, pitch=0, extension='jpg', year=2017, fname=None):

    if not fname:
        fname = "%s_%s_%s" % (year, panoid, str(heading))
    image_format = extension if extension != 'jpg' else 'jpeg'

    url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        # maximum permitted size for free calls
        "size": "%dx%d" % (width, height),
        "fov": fov,
        "pitch": pitch,
        "heading": heading,
        "pano": panoid,
        "key": key
    }

    response = requests.get(url, params=params, stream=True)
    try:
        img = Image.open(BytesIO(response.content))
        filename = '%s/%s.%s' % (flat_dir, fname, extension)
        img.save(filename, image_format)
    except:
        print("Image not found")
        filename = None
    del response
    return filename


def download_flats(panoid, flat_dir, key, width=400, height=300,
                   fov=120, pitch=0, extension='jpg', year=2017):
    for heading in [0, 90, 180, 270]:
        api_download(panoid, heading, flat_dir, key, width, height, fov, pitch, extension, year)

