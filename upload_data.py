import calendar
import datetime
import subprocess
from typing import NamedTuple, Iterable, Optional

"""
total 15848384
-rwxr-xr-x 1 miralles STAT-unit 295714339 Nov 10 17:18 x_20170101.nc
-rwxr-xr-x 1 miralles STAT-unit 295714340 Nov 10 17:31 x_20170102.nc
-rwxr-xr-x 1 miralles STAT-unit 295714340 Nov 10 17:44 x_20170103.nc
-rwxr-xr-x 1 miralles STAT-unit 295714336 Nov 10 17:57 x_20170104.nc
-rwxr-xr-x 1 miralles STAT-unit 295714339 Nov 10 18:08 x_20170105.nc
-rwxr-xr-x 1 miralles STAT-unit 295714340 Nov 10 18:16 x_20170106.nc
-rwxr-xr-x 1 miralles STAT-unit 295714339 Nov 10 18:28 x_20170107.nc
-rwxr-xr-x 1 miralles STAT-unit 295714338 Nov 10 18:40 x_20170108.nc
-rwxr-xr-x 1 miralles STAT-unit 295714340 Nov 10 18:51 x_20170109.nc
-rwxr-xr-x 1 miralles STAT-unit 295714340 Nov 10 19:49 x_20170110.nc
-rwxr-xr-x 1 miralles STAT-unit 295714338 Nov 10 20:00 x_20170111.nc
-rwxr-xr-x 1 miralles STAT-unit 295714338 Nov 10 20:12 x_20170112.nc
"""


def date_range(start, end, step=datetime.timedelta(1)):
    while start <= end:
        yield start
        start += step


class File(NamedTuple):
    size: int
    date: datetime.date
    name: str


def get_uploaded_files() -> Iterable[File]:
    command = 'ssh miralles@izar.epfl.ch ls -l /home/miralles/WindDownscaling_EPFL_UNIBE/data/img_prediction_files'
    result = subprocess.run(command.split(), capture_output=True)
    output = result.stdout.splitlines()[1:]
    month_abbr = list(calendar.month_abbr)
    for line in output:
        parts = line.split()
        month = month_abbr.index(parts[5].decode().capitalize())
        name = parts[-1].decode()
        if not name.startswith('y_'):
            yield File(size=int(parts[4]), date=datetime.date(2021, month, int(parts[6])), name=name)


def filename_date(name: str) -> datetime.date:
    return datetime.datetime.strptime(name.split('_')[1].split('.')[0], "%Y%m%d").date()


def find_date_to_upload(uploaded_files: Iterable[File]) -> Optional[datetime.date]:
    uploaded_files = list(uploaded_files)
    for f in uploaded_files:
        if f.date.month < 11 or f.size < 295000000:
            return filename_date(f.name)
    for date in date_range(datetime.date(2017, 1, 1), datetime.date(2017, 12, 31)):
        if not any(filename_date(f.name) == date for f in uploaded_files):
            return date
    return None


def upload_date(date: datetime.date):
    src = f"/Volumes/Extreme SSD/data/img_prediction_files/x_{date.strftime('%Y%m%d')}.nc"
    dest = f"miralles@izar.epfl.ch:/home/miralles/WindDownscaling_EPFL_UNIBE/data/img_prediction_files/"
    command = ['scp', src, dest]
    print(f'Running {command}')
    subprocess.run(command, check=True)


def main():
    while (date := find_date_to_upload(get_uploaded_files())) is not None:
        try:
            upload_date(date)
        except subprocess.CalledProcessError as e:
            print(f"Got error while uploading {date}: {e}")
    print('DONE')


if __name__ == '__main__':
    main()
