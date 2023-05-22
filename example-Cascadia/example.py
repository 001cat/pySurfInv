import numpy as np
from Triforce.pltHead import *
from Triforce.utils import GeoMap
from pySurfInv.point import PointCascadia as Point

class InvPointGenerator_Cascadia():
    infoDir  = '/home/ayu/Projects/JdF-Model/Models'
    inputDir = 'Input'
    def __init__(self,npzfile) -> None:
        import shapefile
        from matplotlib.patches import Polygon
        self.grd = np.load(npzfile,allow_pickle=True)['grd'][()]
        self.eik = np.load(npzfile,allow_pickle=True)['eikStack'][()]

        plates = shapefile.Reader(f'{self.infoDir}/Plates/PB2002_plates.shp')
        self.plateNA = Polygon(plates.shapes()[6].points).get_path()
        self.platePA = Polygon(plates.shapes()[9].points).get_path()
        self.plateJF = Polygon(plates.shapes()[26].points).get_path()
        self.prismJF = Polygon(np.loadtxt(f'{self.inputDir}/prism.csv',delimiter=','))

        import netCDF4 as nc4
        with nc4.Dataset(f'{self.infoDir}/ETOPO_Cascadia_smoothed.grd') as dset:
            self.topo = GeoMap(dset['lon'][()],dset['lat'][()],dset['z'][()]/1000)
        with nc4.Dataset(f'{self.infoDir}/Crust1.0/crsthk.grd') as dset:
            self.crsthk = GeoMap(dset['x'][()],dset['y'][()],dset['z'][()])
        with nc4.Dataset(f'{self.infoDir}/Crust1.0/sedthk.grd') as dset:
            self.sedthk = GeoMap(dset['x'][()],dset['y'][()],dset['z'][()])
        with nc4.Dataset(f'{self.infoDir}/SedThick/sedthick_world_v2.grd') as dset:
            self.sedthkOce = GeoMap(dset['x'][()],dset['y'][()],dset['z'][()]/1000)
        with nc4.Dataset(f'{self.infoDir}/age_JdF_model_0.01.grd') as dset:
            self.lithoAge = GeoMap(dset['x'][()],dset['y'][()],dset['z'][()])

        # slab_Hayes: Hayes, G., 2018, Slab2 - A Comprehensive Subduction Zone Geometry Model: 
        # U.S. Geological Survey data release, https://doi.org/10.5066/F7PV6JNV.
        with nc4.Dataset(f'{self.infoDir}/Slab2_Cascadia/cas_slab2_dep_02.24.18.grd') as dset:
            self.slabDep = GeoMap(dset['x'][()]-360,dset['y'][()],-dset['z'][()])
        with nc4.Dataset(f'{self.infoDir}/Slab2_Cascadia/cas_slab2_dip_02.24.18.grd') as dset:
            self.slabDip = GeoMap(dset['x'][()]-360,dset['y'][()],dset['z'][()])

        lons = np.arange(-132,-120,0.1); lats = np.arange(39,51,0.1)
        prismThk = np.zeros((len(lats),len(lons)))*np.nan
        for i in range(prismThk.shape[0]):
            for j in range(prismThk.shape[1]):
                lon,lat = lons[j],lats[i]
                if np.isnan(self.sedthkOce.value(lon,lat)):
                    prismThk[i,j] = self.slabDep.value(lon,lat) - max(-self.topo.value(lon,lat),0)-self.sedthk.value(lon,lat)
                else:
                    prismThk[i,j] = self.slabDep.value(lon,lat) - max(-self.topo.value(lon,lat),0)-self.sedthkOce.value(lon,lat)
                if self.plateJF.contains_point((lon,lat)) or self.platePA.contains_point((lon,lat)):
                    prismThk[i,j] = 0
        self.prismthk = GeoMap(lons,lats,prismThk,mask=np.isnan(prismThk))
        
        # self.sedthkOce = GeoMap(); self.sedthkOce.load(f'{priorDir}/sedThk.npz')
        self.mantleInitParmVs = GeoMap(); self.mantleInitParmVs.load(f'{self.inputDir}/parmVs_Ritzwoller.npz')

    def getDisp(self,ptlon,ptlat):
        grd,eik = self.grd,self.eik
        try:
            if grd._lon_type == '0 to 360':
                ptlon += 360*(ptlon<0)
            i,j = grd._findInd(ptlon,ptlat)
        except:
            raise ValueError(f'Point lon={ptlon} lat={ptlat} can not be found')
        pers = [float(Tstr[:-1]) for Tstr in eik.keys()]

        vels = np.array([eik[Tstr]['vel_iso'][i,j] for Tstr in eik.keys()])
        sems = np.array([eik[Tstr]['vel_sem'][i,j] for Tstr in eik.keys()])
        mask = np.array([eik[Tstr]['mask'][i,j] for Tstr in eik.keys()])

        vels[mask] = np.nan
        sems[mask] = np.nan
        vels = np.ma.masked_array(vels,mask=mask)
        sems = np.ma.masked_array(sems,mask=mask)
        return pers,vels,sems

    def genPoint(self,ptlat,ptlon,upscale=2,minUncer=0.005,loc=None,setting=(
                'Input/cascadia-ocean.yml',
                'Input/cascadia-prism.yml',
                'Input/cascadia-continent.yml'
                )) -> Point:
        ptlon -= 360*(ptlon>180)
        pers,vels,sems = self.getDisp(ptlon,ptlat)
        sems.mask[np.isnan(vels)] = True
        vels.mask[np.isnan(vels)] = True
        uncers = upscale*sems; uncers[uncers<minUncer] = minUncer
        if (~vels.mask).sum() < 10:
            print(f'Measurements < 10, skip')
            return None,None
        if ((not self.plateNA.contains_point((ptlon,ptlat))) and loc is None) or (loc=='ocean'):
            print(f'Inside ocean plate')
            outDir = 'OceanInv'
            p = Point(setting[0],{
                        'topo':self.topo.value(ptlon,ptlat),
                        'lithoAge':self.lithoAge.value(ptlon,ptlat),
                        'sedthk':self.sedthkOce.value(ptlon,ptlat),
                        'mantleInitParmVs':self.mantleInitParmVs.value(ptlon,ptlat)
                    },
                    periods=pers,vels=vels,uncers=uncers)
        elif (self.prismJF.contains_point((ptlon,ptlat)) and loc is None) or (loc=='prism'):
            return None,None
            # print('In prism')
            # outDir = 'PrismInv'
            # p = Point(setting[1],{
            #     'topo':self.topo.value(ptlon,ptlat),
            #     'sedthk':self.sedthk.value(ptlon,ptlat) if np.isnan(self.sedthkOce.value(ptlon,ptlat)) else self.sedthkOce.value(ptlon,ptlat),
            #     'prismthk':200 if np.isnan(self.prismthk.value(ptlon,ptlat)) else self.prismthk.value(ptlon,ptlat),
            #     'lithoAge':10
            # },periods=pers,vels=vels,uncers=uncers)
        elif loc in (None,'continent'):
            return None,None
            # print(f'In continent')
            # outDir = 'LandInv'
            # p = Point(setting[2],{
            #     'topo':self.topo.value(ptlon,ptlat),
            #     'sedthk':self.sedthk.value(ptlon,ptlat),
            #     'crsthk':self.crsthk.value(ptlon,ptlat),
            #     'lithoAge':10
            # },periods=pers,vels=vels,uncers=uncers)
        elif loc == 'test':
            pass
            # outDir = 'test'
            # setting = 'Input/cascadia-prism-test.yml'
            # p = Point(setting,{
            #     'topo':self.topo.value(ptlon,ptlat),
            #     'sedthk':self.sedthk.value(ptlon,ptlat) if np.isnan(self.sedthkOce.value(ptlon,ptlat)) else self.sedthkOce.value(ptlon,ptlat),
            #     'prismthk':200 if np.isnan(self.prismthk.value(ptlon,ptlat)) else self.prismthk.value(ptlon,ptlat),
            #     'lithoAge':10
            # },periods=pers,vels=vels,uncers=uncers)
        else:
            raise ValueError(f'Wrong location specificated {loc}')

        # return None if np.nan found in p.initMod
        from pySurfInv.utils import _dictIterModifier
        def checker(v):
            try:
                len(v);isnan=False
            except:
                isnan = bool(np.isnan(v))
            if isnan:
                raise ValueError('nan value found')
            else:
                return False
        def modifier(v):
            return v
        try:
            _dictIterModifier(p.initMod.toYML(),checker,modifier)
        except ValueError as e:
            if str(e) == 'nan value found':
                return None,None
            else:
                raise e
        
        p.pid = f'{ptlon+360*(ptlon<0):.1f}_{ptlat:.1f}'
        return p,outDir

    def genPointLst(self):
        pList = []
        for lon in self.grd.lons:
            for lat in self.grd.lats:
                p,outDir = self.genPoint(lat,lon,4,0.005)
                if p is not None:
                    pList.append((p,outDir))
        return pList

# pGen = InvPointGenerator_Cascadia('Input/ani_tomo_A0_finalInv.npz')
# p,_ = pGen.genPoint(45,-130)
# p.MCinvMP(f'test',runN=24000,step4uwalk=800,nprocess=20)
# p.MCinvMP(f'test_priori',runN=24000,step4uwalk=800,nprocess=20,priori=True)
# from pySurfInv.point import PostPoint
# postp = PostPoint('test/230.0_45.0.npz',realMCMC=True)

pGen = InvPointGenerator_Cascadia('Input/ani_tomo_A0_finalInv.npz')
pList = pGen.genPointLst()
