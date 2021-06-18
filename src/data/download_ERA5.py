# to read netcdf (for quick check)
# to download from CDS (Copernicus Climate Data Store)
# to create dates (for loop)
from datetime import date, timedelta
# set path
from pathlib import Path

import cdsapi

# plotting

datapath = f'/Users/Boubou/Documents/GitHub/WindDownscaling_EPFL_UNIBE/data/ERA5/'
start_date = date(2016, 1, 10)
end_date = date(2020, 12, 31)
delta = end_date - start_date
print(delta.days)

c = cdsapi.Client()
# loop trough each day
for i in range(delta.days + 1):
    timestep = start_date + timedelta(days=i)
    year = str(timestep.year)
    month = str(timestep.month).zfill(2)
    day = str(timestep.day).zfill(2)
    filename = year + month + day + '_era5_surface_hourly.nc'
    print(filename)

    # checking if files already exists, else download ERA5 data
    my_file = Path(datapath + filename)
    if my_file.is_file():
        print(" File already exists!")
    else:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': [
                    '100m_u_component_of_wind', '100m_v_component_of_wind', '10m_u_component_of_wind',
                    '10m_v_component_of_wind', '2m_dewpoint_temperature', '2m_temperature',
                    'boundary_layer_height', 'surface_pressure', 'surface_sensible_heat_flux',
                    'total_precipitation', 'forecast_surface_roughness',
                ],
                'year': year,
                'month': month,
                'day': day,
                'time': [
                    '00:00', '01:00', '02:00',
                    '03:00', '04:00', '05:00',
                    '06:00', '07:00', '08:00',
                    '09:00', '10:00', '11:00',
                    '12:00', '13:00', '14:00',
                    '15:00', '16:00', '17:00',
                    '18:00', '19:00', '20:00',
                    '21:00', '22:00', '23:00',
                ],
                'area': [
                    48.2, 5.2, 45.4,
                    11.02,
                ],
            },
            datapath + filename)

# variables on 500 hPa

for i in range(delta.days + 1):
    timestep = start_date + timedelta(days=i)
    year = str(timestep.year)
    month = str(timestep.month).zfill(2)
    day = str(timestep.day).zfill(2)
    filename = year + month + day + '_era5_z500_hourly.nc'
    print(filename)

    # checking if files already exists, else download ERA5 data
    my_file = Path(datapath + filename)
    if my_file.is_file():
        print(" File already exists!")
    else:
        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': [
                    'divergence', 'geopotential', 'u_component_of_wind',
                    'v_component_of_wind', 'vertical_velocity', 'vorticity',
                ],
                'pressure_level': '500',
                'year': year,
                'month': month,
                'day': day,
                'time': [
                    '00:00', '01:00', '02:00',
                    '03:00', '04:00', '05:00',
                    '06:00', '07:00', '08:00',
                    '09:00', '10:00', '11:00',
                    '12:00', '13:00', '14:00',
                    '15:00', '16:00', '17:00',
                    '18:00', '19:00', '20:00',
                    '21:00', '22:00', '23:00',
                ],
                'area': [
                    48.2, 5.2, 45.4,
                    11.02,
                ],
            },
            datapath + filename)