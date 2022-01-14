# Wind downscaling over Switzerland project (EPFL/UNIBE)

## Datasets: 

* ERA5 reanalysis from ECMWF: global gridded dataset from 1979 - today with a spatial resolution of 25 x 25 km and a 1-hourly temporal resolution    
freely available at: https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset&text=era5
* ECMWF recently released ERA5-Land. It is the land component of ERA5 with a spatial resolution of 9 km high-resolution topographic descriptors (DEM) freely available: https://geovite.ethz.ch/DigitalElevationModels.html 
* COSMO-1 anaylsis from MeteoSwiss: gridded dataset from 2016 - today with ~1 km resolution and a 1-hourly temporal resolution
*coming soon* 
* MeteoSwiss Station Data SwissMetNet: https://www.meteoswiss.admin.ch/home/measurement-and-forecasting-systems/land-based-stations/automatisches-messnetz.html

## Relevant Literature:

* Höhlein et al., 2020: A Comparative Study of Convolutional Neural Network Models for Wind Field Downscaling
   * paper: https://doi.org/10.1002/met.1961
   * code:  https://github.com/khoehlein/CNNs-for-Wind-Field-Downscaling
* Leinonen et al., 2020: Stochastic Super-Resolution for Downscaling Time-Evolving Atmospheric Fields with a Generative Adversarial Network
   * paper: https://doi.org/10.1109/TGRS.2020.3032790
   * code: https://github.com/jleinonen/downscaling-rnn-gan
* Dujardin and Lehning, 2020: Multi-Resolution Convolutional Neural Network for High-Resolution Downscaling of Wind Fields from Operational Weather Prediction Models in Complex Terrain
   * AGU Presentation: https://agu.confex.com/agu/fm20/videogateway.cgi/id/8802?recordingid=8802
   * Master Thesis M. Schaer: https://infoscience.epfl.ch/record/282346 
* Winstral et al., 2017: Statistical Downscaling of Gridded Wind Speed Data Using Local Topography
   * paper: https://doi.org/10.1175/JHM-D-16-0054.1
* Daniele Nerini, 2020: Probabilistic Deep Learning for Postprocessing Wind Forecasts in Complex Terrain
   * presentation: https://vimeo.com/465719202  
* Amato et al., 2020: A novel framework for spatio-temporal prediction of environmental data using deep learning
   * paper: https://doi.org/10.1038/s41598-020-79148-7
   * code: https://github.com/federhub/ST_DeepLearning
* Robert er al., 2012: Spatial prediction of monthly wind speeds in complex terrain with adaptive general regression neural networks
   * paper:  https://doi.org/10.1002/joc.3550
  
## Usage

### Prerequisites 

* Conda environment
* Get a Copernicus API key from: https://cds.climate.copernicus.eu/api-how-to
  * create a file at \$HOME/.cdsapirc with the required UID and key
  
### Installing dependencies

```bash
conda install -y -c conda-forge gdal tensorflow xarray numpy=1.19.5 pandas pysftp cdsapi elevation rasterio dask python-dotenv
```
Install this package:
```bash
pip install -U git+https://github.com/OpheliaMiralles/wind-downscaling-gan.git
```

### Downscaling winds
* Download ERA5 low-resolution winds for a specific area and time range using the function `download_ERA5` from the `downscaling.data` package
* Download DEM data, for example using: 
```bash
eio --product SRTM3 clip -o {dem_data_dest_folder} --bounds -4.96 42.2 8.3 51.3
```
* Downscale wind fields for a specific date and area of interest:
```bash
downscale --era {ERA5_data_folder} --dem {dem_raster.tif} --date 20160401 --lon="-1:3" --lat 48:50 -o downscaled_winds.nc
```
For a more hands on approach, you can also use the following python notebook https://github.com/OpheliaMiralles/wind-downscaling-gan/blob/master/src/downscaling/wind_downscaling.ipynb where visual representation of the downscaled maps is provided.
