import numpy as np
from netCDF4 import Dataset
from Triforce.utils import GeoMap
from Triforce.pltHead import *


def plotCascadiaSlab(lon1,lat1,lon2,lat2):
    from geographiclib.geodesic import Geodesic
    with Dataset('/home/ayu/Projects/Cascadia/Models/Slab2_Cascadia/cas_slab2_dep_02.24.18.grd') as dset:
        slabDep = GeoMap(dset['x'][()],dset['y'][()],-dset['z'][()])
    with Dataset('/home/ayu/Projects/Cascadia/Models/Slab2_Cascadia/cas_slab2_thk_02.24.18.grd') as dset:
        slabThk = GeoMap(dset['x'][()],dset['y'][()],dset['z'][()])
    with Dataset('/home/ayu/Projects/Cascadia/Models/Slab2_Cascadia/cas_slab2_dip_02.24.18.grd') as dset:
        slabDip = GeoMap(dset['x'][()],dset['y'][()],dset['z'][()])
    geoDict = Geodesic.WGS84.Inverse(lat1,lon1,lat2,lon2)
    slabU = []
    xslabD,slabD = [],[]
    x = np.linspace(0,geoDict['s12'],301)/1000
    for i,d in enumerate(x*1000):
        tmp = Geodesic.WGS84.Direct(lat1,lon1,geoDict['azi1'],d)
        slabU.append(slabDep.value(tmp['lon2'],tmp['lat2']))
        theta = slabDip.value(tmp['lon2'],tmp['lat2'])/180*np.pi
        H = slabThk.value(tmp['lon2'],tmp['lat2'])
        xslabD.append(d/1000-H*np.sin(theta))
        slabD.append(slabDep.value(tmp['lon2'],tmp['lat2'])+H*np.cos(theta))
    if abs(lon1-lon2)<0.01:
        xslabD = np.interp(xslabD,x,np.linspace(lat1,lat2,301))
        x = np.linspace(lat1,lat2,301)
    elif abs(lat1-lat2)<0.01:
        xslabD = np.interp(xslabD,x,np.linspace(lon1,lon2,301))
        x = np.linspace(lon1,lon2,301)
    plt.plot(x,slabU,'w',lw=4)
    plt.plot(xslabD,slabD,'w',lw=4)
def plotCascadiaSlab4Map(m,levels=[60,75,90,120,150]):
    with Dataset('/home/ayu/Projects/Cascadia/Models/Slab2_Cascadia/cas_slab2_dep_02.24.18.grd') as dset:
        slabDep = GeoMap(dset['x'][()],dset['y'][()],-dset['z'][()])
    XX,YY = np.meshgrid(slabDep.lons-360,slabDep.lats)
    cs = m.contour(XX,YY,slabDep.z,latlon=True,levels=levels,colors='white',linewidths=2)
    plt.clabel(cs, fontsize=9, inline=True,colors='k')
