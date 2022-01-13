import argparse
from pathlib import Path

import xarray as xr

from downscaling import downscale


def main():
    parser = argparse.ArgumentParser(description='Downscale ER5 wind fields')
    parser.add_argument('--era', help='path to folder with ERA5 data', required=True)
    parser.add_argument('--dem', help='path to DEM data file', required=True)
    parser.add_argument('--date', help='date to downscale in YYYYMMDD format', required=True)
    parser.add_argument('--lon', default=None, help='longitude range (ex: 45.6:46.2)')
    parser.add_argument('--lat', default=None, help='latitude range (ex: 45.6:46.2)')
    parser.add_argument('-o', '--output', help='output path for the downscaled map (*.nc)', default='downscaled.nc')
    args = parser.parse_args()

    longitude_r = tuple(map(float, args.lon.split(':'))) if args.lon else None
    latitude_r = tuple(map(float, args.lat.split(':'))) if args.lat else None

    era5 = xr.open_mfdataset(Path(args.era).glob(f'{args.date}*surface*.nc'))
    raster_topo = xr.open_rasterio(args.dem)
    downscaled_maps = downscale(era5, raster_topo, range_lon=longitude_r, range_lat=latitude_r, overlap_factor=0.01)

    downscaled_maps.to_netcdf(args.output)


if __name__ == '__main__':
    main()
