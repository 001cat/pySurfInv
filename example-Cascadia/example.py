import sys
import numpy as np
from Triforce.utils import GeoGrid,GeoMap
from pySurfInv.point import Point
from pySurfInv.model3D import Model3D

def npzConvert():
    tmp = np.load('/home/ayu/Projects/Cascadia/Tomo/ani_tomo_A0_finalSZ.npz',allow_pickle=True)
    grd,eik = tmp['grd'][()],tmp['eikStack'][()]

    lats,lons = grd.lats,grd.lons
    np.savez_compressed('ani_tomo_A0_finalSZ.npz',lats=lats,lons=lons,eikStack=eik)


def loadNpz():
    tmp = np.load('ani_tomo_A0_finalSZ.npz',allow_pickle=True)
    lats,lons,eik = tmp['lats'][()],tmp['lons'][()],tmp['eikStack'][()]
    geoGrid = GeoGrid(lons=lons,lats=lats)
    return geoGrid,eik

def loadGeoMaps():
    from netCDF4 import Dataset
    with Dataset('infos/ETOPO_Cascadia_smoothed.grd') as dset:
        topo = GeoMap(dset['lon'][()],dset['lat'][()],dset['z'][()]/1000)
    with Dataset('infos/crsthk.grd') as dset:
        crsthk = GeoMap(dset['x'][()],dset['y'][()],dset['z'][()])
    with Dataset('infos/sedthk.grd') as dset:
        sedthk = GeoMap(dset['x'][()],dset['y'][()],dset['z'][()])
    with Dataset('infos/sedthick_world_v2.grd') as dset:
        sedthkOce = GeoMap(dset['x'][()],dset['y'][()],dset['z'][()]/1000)
    with Dataset('infos/age_JdF_model_0.01.grd') as dset:
        lithoAge = GeoMap(dset['x'][()],dset['y'][()],dset['z'][()])
    return topo,sedthk,crsthk,sedthkOce,lithoAge

def getMask(geoGrid:GeoGrid):
    from matplotlib.patches import Polygon
    bounds = [(-131,40.5726),(-127.717, 40.5726),(-127.579, 40.5385),(-127.579, 40.45),(-126.869, 40.4224),(-126.159, 40.3904),\
        (-125.45, 40.3541),(-124.742, 40.3134), (-124.953, 40.6333), (-124.933, 40.8839), (-125.089, 41.1354), (-125.182, 41.3801), \
        (-125.309, 41.6522), (-125.281, 41.8176), (-125.313, 41.9856), (-125.331, 42.1419), (-125.29, 42.2845), (-125.209, 42.3588), (-125.198, 42.4917), \
        (-125.268, 42.5614), (-125.289, 42.8624), (-125.289, 43.0514), (-125.343, 43.1426), (-125.394, 43.2671), (-125.474, 43.426), (-125.48, 43.7375), \
        (-125.45, 43.9254), (-125.429, 44.0023), (-125.362, 44.0663), (-125.372, 44.3226), (-125.372, 44.5227), (-125.376, 44.6675), (-125.434, 44.7254), \
        (-125.437, 44.8813), (-125.435, 45.1036), (-125.517, 45.4738), (-125.655, 45.9461), (-125.773, 46.295), (-125.876, 46.6433), (-125.998, 46.9697), \
        (-126.085, 47.3503), (-126.37, 47.748), (-126.486, 47.9956), (-126.733, 48.291), (-127.018, 48.5646), (-127.378, 48.7502), (-127.608, 48.8768),\
        (-131,48.8768),(-131,40.5726)]
    poly_path = Polygon(bounds).get_path()

    mask = np.ones(geoGrid.XX.shape,dtype=bool)
    for lon in geoGrid.lons:
        for lat in geoGrid.lats:
            i,j = geoGrid._findInd(lon,lat)
            mask[i,j] = not poly_path.contains_point((lon-360*(lon>=180),lat))

    return mask

def runMC():
    def getDisp(geoGrid,eik,lon,lat):
        i,j = geoGrid._findInd(lon,lat)
        pers = [float(Tstr[:-1]) for Tstr in eik.keys()]
        vels = np.array([eik[Tstr]['vel_iso'][i,j] for Tstr in eik.keys()])
        sems = np.array([eik[Tstr]['vel_sem'][i,j] for Tstr in eik.keys()])
        mask = np.array([eik[Tstr]['mask'][i,j] for Tstr in eik.keys()])
        if np.any(mask):
            return None,None,None
        return pers,vels,sems

    geoGrid,eik = loadNpz()
    topo,sedthk,crsthk,sedthkOce,lithoAge = loadGeoMaps()
    invAreaMask = getMask(geoGrid)


    for lon in geoGrid.lons:
        for lat in geoGrid.lats:

            i,j = geoGrid._findInd(lon,lat)
            if invAreaMask[i,j]:
                continue

            pers,vels,sems = getDisp(geoGrid,eik,lon,lat)
            if pers is None:
                invAreaMask[i,j] = 1
                continue

            # if f'{lon:.1f}_{lat:.1f}' != '233.0_46.0':
            #     continue

            print(f'Running inversion: lon={lon:.1f} lat={lat:.1f}')

            p = Point('cascadia-ocean.yml',{
                'topo':topo.value(lon,lat),
                'sedthk':sedthkOce.value(lon,lat),
                'lithoAge':lithoAge.value(lon,lat)
                },periods=pers,vels=vels,uncers=sems*4)
            p.pid = f'{lon:.1f}_{lat:.1f}'
            p.MCinvMP(f'mcdata',runN=50000,step4uwalk=1000,nprocess=17)
            p.MCinvMP(f'mcdata_priori',runN=50000,step4uwalk=1000,nprocess=17,priori=True)
            


if __name__ == '__main__':
    runMC()

    mod = Model3D(); mod.loadInvDir('example-Cascadia/mcdata')
    mod.load('invModel.npz');mod.smoothGrid(width=80);mod.write('invModel-Smooth.npz')

    mod.plotMapView(100,'auto',vmin=4.0,vmax=4.5)
    mod.plotSection(-130.2+360,46,-125.6+360,46,maxD=200,title='Lat = 46$\degree$')

    






