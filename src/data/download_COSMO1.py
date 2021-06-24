import re
from ftplib import FTP
from pathlib import Path

import pandas as pd
import xarray as xr


def date_file_regex(d):
    return r'cosmo-1_\w+_{d.year}{d.month:02}{d.day:02}\d\d\.nc'.format(d=d)


def download_COSMO1(username, password, datapath, start_date, end_date):
    datapath = Path(datapath)
    datapath.mkdir(exist_ok=True)
    with FTP('giub-torrent.unibe.ch', username, password, timeout=200) as c:
        c.cwd('COSMO-1')
        all_files = []
        c.retrlines('NLST', all_files.append)
        for d in pd.date_range(start_date, end_date):
            day_dest = datapath / f'{d.year}{d.month:02}{d.day:02}.nc'
            if day_dest.exists():
                continue
            r = date_file_regex(d)
            date_files = [f for f in all_files if re.match(r, f)]
            if not date_files:
                print(f'No file found for {d}')
                continue
            local_files = []
            for file in date_files:
                dest = datapath / file.split('_')[-1]
                print(f'Downloading {file} to {dest}')
                with open(dest, 'wb') as fp:
                    c.retrbinary(f'RETR {file}', fp.write)
                local_files.append(dest)
            print(f"Concatenating arrays for {d}")
            day_dataset = xr.open_mfdataset(local_files)
            day_dataset.to_netcdf(day_dest)
            for f in local_files:
                f.unlink()
    print("Finished downloading COSMO data")
