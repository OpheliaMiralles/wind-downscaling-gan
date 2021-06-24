from datetime import date
from pathlib import Path

import cdsapi
import pandas as pd


def _download_ERA5_data(datapath: Path, file_suffix: str, start_date, end_date, data_name, args):
    c = cdsapi.Client()
    request_args = {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'time': [
            '00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00', '21:00', '22:00', '23:00',
        ],
        'area': [
            48.2, 5.2, 45.4, 11.02,
        ]
    }
    request_args.update(args)
    for date in pd.date_range(start_date, end_date):
        filename = f'{date.year}{date.month:02}{date.day:02}_{file_suffix}'
        dest = Path(datapath).joinpath(filename).with_suffix('.nc')
        if dest.exists():
            print(f"File {filename} already exists")
        else:
            dest.parent.mkdir(exist_ok=True)
            date_request = {**request_args, 'year': date.year, 'month': date.month, 'day': date.day, }
            c.retrieve(data_name, date_request, str(dest))


def download_ERA5_surface(datapath, start_date=date(2016, 1, 10), end_date=date(2020, 12, 31)):
    _download_ERA5_data(datapath, 'era5_surface_hourly', start_date, end_date, 'reanalysis-era5-single-levels',
                        {'variable': [
                            '100m_u_component_of_wind', '100m_v_component_of_wind', '10m_u_component_of_wind',
                            '10m_v_component_of_wind', '2m_dewpoint_temperature', '2m_temperature',
                            'boundary_layer_height', 'surface_pressure', 'surface_sensible_heat_flux',
                            'total_precipitation', 'forecast_surface_roughness',
                        ]})


def download_ERA5_pressure_500(datapath, start_date=date(2016, 1, 10), end_date=date(2020, 12, 31)):
    _download_ERA5_data(datapath, 'era5_z500_hourly', start_date, end_date, 'reanalysis-era5-pressure-levels',
                        {'pressure_level': '500', 'variable': [
                            'divergence', 'geopotential',
                            'vertical_velocity', 'vorticity',
                        ]})


def download_ERA5(datapath, start_date=date(2016, 1, 10), end_date=date(2020, 12, 31)):
    download_ERA5_surface(datapath, start_date, end_date)
    download_ERA5_pressure_500(datapath, start_date, end_date)
