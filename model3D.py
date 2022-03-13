import os,glob
import numpy as np
from geographiclib.geodesic import Geodesic
from pySurfInv.models import buildModel1D,Model1D,Model1D_Puregird
from pySurfInv.point import PostPoint
from Triforce.utils import GeoGrid,savetxt
from Triforce.obspyPlus import randString
from Triforce.pltHead import *
from Triforce.customPlot import cvcpt,rbcpt

def mapSmooth(lons,lats,z,tension=0.0, width=50.):
    lons = lons.round(decimals=4)
    lats = lats.round(decimals=4)
    tmpFname = f'tmp{randString(10)}'
    XX,YY = np.meshgrid(lons,lats)
    dlon,dlat = lons[1]-lons[0],lats[1]-lats[0]
    savetxt(f'{tmpFname}.xyz',XX.flatten(),YY.flatten(),z.flatten())
    with open(f'{tmpFname}.bash','w+') as f:
        REG     = f'-R{lons[0]:.2f}/{lons[-1]:.2f}/{lats[0]:.2f}/{lats[-1]:.2f}'
        f.writelines(f'gmt gmtset MAP_FRAME_TYPE fancy \n')
        f.writelines(f'gmt surface {tmpFname}.xyz -T{tension} -G{tmpFname}.grd -I{dlon:.2f}/{dlat:.2f} {REG} \n')
        f.writelines(f'gmt grdfilter {tmpFname}.grd -D4 -Fg{width} -G{tmpFname}_Smooth.grd {REG} \n')
    os.system(f'bash {tmpFname}.bash')
    from netCDF4 import Dataset
    with Dataset(f'{tmpFname}_Smooth.grd') as dset:
        zSmooth = dset['z'][()]
    os.system(f'rm {tmpFname}* gmt.conf gmt.history')
    return zSmooth

class Model3D(GeoGrid):
    ''' to avoid bugs in gmt smooth, start/end of lons/lats should be integer '''
    def __init__(self,lons=[],lats=[]) -> None:
        super().__init__()
        self.lons = np.array(lons)
        self.lats = np.array(lats)
        self.mods       = [ [None]*len(lons) for _ in range(len(lats))]
        self._mods_init = [ [None]*len(lons) for _ in range(len(lats))]
        self._mods_avg  = None
        self.misfits    = [ [None]*len(lons) for _ in range(len(lats))]
        self.disps      = [ [None]*len(lons) for _ in range(len(lats))]
    @property
    def mask(self):
        m,n = len(self.lats),len(self.lons)
        mask = np.ones((m,n),dtype=bool)
        for i in range(m):
            for j in range(n):
                mask[i,j] = (self.mods[i][j] is None)
        return mask
    
    def loadInvDir(self,invDir='mcdata'):
        ptlons,ptlats = [],[]
        if len(self.lons) == 0:
            try: # check format and initialize
                for npzfile in glob.glob(f'{invDir}/*.npz'):
                    ptlon,ptlat = npzfile.split('/')[-1][:-4].split('_')[:]
                    ptlons.append(ptlon); ptlats.append(ptlat)
                ptlons=np.array([float(a) for a in set(ptlons)]); ptlons.sort(); dlon = min(np.diff(ptlons))
                ptlats=np.array([float(a) for a in set(ptlats)]); ptlats.sort(); dlat = min(np.diff(ptlats))
                lons = np.arange(np.floor(ptlons[0]),np.ceil(ptlons[-1])+dlon/2,dlon)
                lats = np.arange(np.floor(ptlats[0]),np.ceil(ptlats[-1])+dlat/2,dlat)
                self.__init__(lons,lats)
            except:
                raise TypeError('Could not take lat/lon, please make sure the format is invDir/lon_lat.npz')
        for npzfile in glob.glob(f'{invDir}/*.npz'):
            ptlon,ptlat = npzfile.split('/')[-1][:-4].split('_')[:]
            ptlon,ptlat = float(ptlon),float(ptlat)
            self._addInvPoint(ptlon,ptlat,PostPoint(npzfile))

    def vsProfile(self,z,lat,lon):
        def foo(j,i,z):
            try:
                return self.mods[j,i].value(z)
            except AttributeError:
                return np.nan*np.ones(z.shape)
        return self._interp2D(lat,lon,foo,z=z)
    def topo(self,lat,lon):
        def foo(j,i):
            try:
                return self.mods[j,i].info['topo']
            except AttributeError:
                return np.nan
        return self._interp2D(lat,lon,foo)
    def moho(self,lat,lon):
        def foo(j,i):
            try:
                return self.mods[j,i].moho()
            except AttributeError:
                return np.nan
        return self._interp2D(lat,lon,foo)

    def smooth(self,width=50):
        m,n = len(self.lats),len(self.lons)
        self._mods_avg = [ [None]*n for _ in range(m)]
        paras = [ [None]*n for _ in range(m)]
        mask = self.mask
        Nparas = len(self.mods[np.where(~mask)[0][0]][np.where(~mask)[1][0]]._brownians())
        for i in range(m):
            for j in range(n):
                if not mask[i,j]:
                    paras[i][j] = self.mods[i][j]._brownians()
                else:
                    paras[i][j] = [np.nan]*Nparas
        paras = np.array(paras)
        print('smoothing')
        import tqdm
        for i in tqdm.tqdm(range(paras.shape[-1])):
            paras[:,:,i] = mapSmooth(self.lons,self.lats,paras[:,:,i],width=width)
        for i in range(m):
            for j in range(n):
                if not mask[i,j]:
                    self._mods_avg[i][j] = self.mods[i][j].copy()
                    self.mods[i][j]._loadMC(paras[i][j])
    def smoothGrid(self,width=50):
        ''' To combine and smooth areas with different model settings '''
        nGridsDict = {'water':2,'sediment':6,'crust':30,'mantle':200}
        nSeisProp = 6
        def mod2grid(mod:Model1D):
            inProfiles = mod.seisPropGrids()
            outProfiles= [[] for _ in range(len(inProfiles))]
            
            grp = np.array(inProfiles[-1])
            for k,v in nGridsDict.items():
                I = grp == k
                for i in range(len(inProfiles)-1):
                    n = len(inProfiles[i][I])
                    if n == 0:
                        outProfiles_seg = np.zeros(v)*np.nan
                    else:
                        outProfiles_seg = np.interp(np.linspace(0,1,v),np.linspace(0,1,n),inProfiles[i][I])
                    outProfiles[i].extend(list(outProfiles_seg))
                outProfiles[-1].extend([k]*v)
            outProfiles = [np.array(p) for p in outProfiles[:-1]] + outProfiles[-1:]
            # print(len(outProfiles[-1]))
            return Model1D_Puregird(outProfiles,info=mod.copy().info)
            

        m,n = len(self.lats),len(self.lons)
        self._mods_avg = [ [None]*n for _ in range(m)]
        mat = np.zeros((m,n,nSeisProp,sum(nGridsDict.values())))
        for i in range(m):
            for j in range(n):
                mod = self.mods[i][j]
                self._mods_avg[i][j] = mod
                self.mods[i][j] = None if mod is None else mod2grid(mod)
                if self.mods[i][j] is None:
                    mat[i,j,:,:] = np.nan
                else:
                    mat[i,j,:,:] = np.array(self.mods[i][j].seisPropGrids()[:-1])
        matSmooth = mat.copy()
        print('smoothing')
        import tqdm
        for k in tqdm.tqdm(range(mat.shape[-1])):
            # print(f'{k+1}/{nFine}')
            for i in range(mat.shape[-2]):
                matSmooth[:,:,i,k] = mapSmooth(self.lons,self.lats,mat[:,:,i,k],width=width)
        for i in range(m):
            for j in range(n):
                if not self.mask[i,j]:
                    grp = self.mods[i][j].seisPropGrids()[-1]
                    inProfiles = [p for p in matSmooth[i,j,:,:]] + [grp]
                    self.mods[i][j] = Model1D_Puregird(inProfiles,self.mods[i][j].info)

    def write(self,fname):
        np.savez_compressed(fname,lons=self.lons,lats=self.lats,
                            misfits=self.misfits,disps=self.disps,
                            mods=self.mods,modsInit=self._mods_init,modsAvg=self._mods_avg)
    def load(self,fname):
        tmp = np.load(fname,allow_pickle=True)
        self.lons = tmp['lons'][()]
        self.lats = tmp['lats'][()]
        self.misfits    = tmp['misfits'][()]
        self.disps      = tmp['disps'][()]
        self.mods       = tmp['mods'][()]
        self._mods_init = tmp['modsInit'][()]
        self._mods_avg  = tmp['modsAvg'][()]
    def copy(self):
        from copy import deepcopy
        return deepcopy(self)


    def _addInvPoint(self,lon,lat,postpoint:PostPoint):
        print(f'Add point {lon:.1f}_{lat:.1f}')
        i,j = self._findInd(lon,lat)
        self.mods[i][j]     = postpoint.avgMod.copy()
        self._mods_init[i][j] = postpoint.initMod.copy()
        self.misfits[i][j]  = postpoint.avgMod.misfit
        self.disps[i][j]    = {'T':postpoint.obs['T'],
                               'pvelo':postpoint.obs['c'],
                               'pvelp':postpoint.avgMod.forward(postpoint.obs['T']),
                               'uncer':postpoint.obs['uncer']}
    def _interp2D(self,lat,lon,foo,**kwargs):
        lon = lon + 360*(lon < 0)
        if (lon-self.lons[0]) * (lon-self.lons[-1]) > 0:
            # print('Longitude is out of range!')
            return np.nan
        if (lat-self.lats[0]) * (lat-self.lats[-1]) > 0:
            # print('Latitude is out of range!')
            return np.nan
        i = np.where(self.lons-lon>=0)[0][0]
        j = np.where(self.lats-lat>=0)[0][0]
        p0 = foo(j-1,i-1,**kwargs)
        p1 = foo(j,i-1,**kwargs)
        p2 = foo(j-1,i,**kwargs)
        p3 = foo(j,i,**kwargs)
        Dx = self.lons[i] - self.lons[i-1]
        Dy = self.lats[j] - self.lats[j-1]
        dx = lon - self.lons[i-1]
        dy = lat - self.lats[j-1]
        p = p0+(p1-p0)*dy/Dy+(p2-p0)*dx/Dx+(p0+p3-p1-p2)*dx*dy/Dx/Dy
        return p

    ''' figure plotting '''
    def mapview(self,depth):
        mask = self.mask
        vsMap = np.ma.masked_array(np.zeros(mask.shape),mask=mask)
        m,n = len(self.lats),len(self.lons)
        for i in range(m):
            for j in range(n):
                if not mask[i,j]:
                    vsMap[i,j] = self.mods[i][j].value(depth)
        return vsMap
    def section(self,lon1,lat1,lon2,lat2):
        # lon1,lat1,lon2,lat2 = -131+360,46,-125+360,43.8
        geoDict = Geodesic.WGS84.Inverse(lat1,lon1,lat2,lon2)
        # lats,lons = [],[]
        x = np.linspace(0,geoDict['s12'],301)/1000
        y = np.linspace(0,200-0.01,201)
        z = np.zeros((len(y),len(x)))
        moho = np.zeros(len(x))
        topo = np.zeros(len(x))
        for i,d in enumerate(x*1000):
            tmp = Geodesic.WGS84.Direct(lat1,lon1,geoDict['azi1'],d)
            # lats.append(tmp['lat2']);lons.append(tmp['lon2'])
            z[:,i] = self.vsProfile(y,tmp['lat2'],tmp['lon2'])
            moho[i]= self.moho(tmp['lat2'],tmp['lon2'])
            topo[i]= self.topo(tmp['lat2'],tmp['lon2'])
        z = np.ma.masked_array(z,np.isnan(z))
        if abs(lon1-lon2)<0.01:
            x = np.linspace(lat1,lat2,301)
        elif abs(lat1-lat2)<0.01:
            x = np.linspace(lon1,lon2,301)
        XX,YY = np.meshgrid(x,y)
        return XX,YY,z,moho,topo
    def plotSection_OLD(self,lon1,lat1,lon2,lat2,vmin=4.1,vmax=4.4,cmap=cvcpt,maxD=200,shading='gouraud'):
        XX,YY,Z,moho = self.section(lon1,lat1,lon2,lat2)
        plt.figure(figsize=[8,4.8])
        # f = interpolate.interp2d(XX[0,:],YY[:,0],Z,kind='cubic')
        # newX = np.linspace(XX[0,0],XX[0,-1],300)
        # newY = np.linspace(YY[0,0],YY[-1,0],300)
        # newZ = f(newX,newY)
        # XX,YY = np.meshgrid(newX,newY)
        plt.pcolormesh(XX,YY,Z,shading=shading,cmap=cmap,vmin=vmin,vmax=vmax)
        plt.plot(XX[0,:],moho,'k',lw=4)
        plt.ylim(0,maxD)
        plt.colorbar(orientation='horizontal',fraction=0.1,aspect=50,pad=0.08)
        plt.gca().invert_yaxis()

    def plotSection(self,lon1,lat1,lon2,lat2,vCrust=[3.0,4.0],vMantle=[4.0,4.5],cmap=cvcpt,
                    maxD=200,shading='gouraud',title=None,decorateFuns=[]):
        from Triforce.customPlot import addAxes,addCAxes
        XX,YY,Z,moho,topo = self.section(lon1,lat1,lon2,lat2)
        mask = ~(YY <= np.tile(moho,(YY.shape[0],1)))
        Z_crust = np.ma.masked_array(Z,mask=mask)
        fig = plt.figure(figsize=[12,5])
        ax2 = addAxes([0,0.18,1,0.65])
        bbox = ax2.get_position()
        x,y,w,h = bbox.x0,bbox.y0,bbox.width,bbox.height
        ax1 = plt.axes([x,y+h+0.05*h,w,h/6],sharex=ax2)
        ax1.axes.xaxis.set_visible(False)
        cax1 = addCAxes(ax2,location='bottom',size=0.05,pad=0.06)
        cax2 = addCAxes(ax2,location='bottom',size=0.05,pad=0.16)

        topo_plot = topo.copy()
        topo_plot[topo_plot<0] /= 2
        topo_min,topo_max = np.nanmin(topo)*1000,np.nanmax(topo)*1000
        ax1.plot(XX[0,:], topo_plot*1000., 'k', lw=3)
        ax1.fill_between(XX[0,:], -10000, topo_plot*1000., facecolor='grey')
        ax1.fill_between(XX[0,:], 0, topo_plot*1000., where=topo_plot<0, facecolor='#d4f1f9')
        ax1.set_yticks([-2000/2,0,1000])
        ax1.set_yticklabels(['-2000','0','1000'])
        ax1.set_ylim(np.nanmin(topo_plot)*1000-100, np.nanmax(topo_plot)*1000.+300.)
        # ax1.plot(XX[0,:],np.zeros(XX.shape[1]),'--k',lw=0.5)
        
        

        ax1.set_title(title)

        plt.axes(ax2)
        plt.pcolormesh(XX,YY,Z,shading=shading,cmap=cmap,vmin=vMantle[0],vmax=vMantle[1])
        plt.colorbar(cax=cax2,orientation='horizontal')
        plt.pcolormesh(XX,YY,Z_crust,shading=shading,cmap=cmap,vmin=vCrust[0],vmax=vCrust[1])
        plt.colorbar(cax=cax1,orientation='horizontal')
        plt.plot(XX[0,:],moho,'k',lw=4)
        plt.ylim(0,maxD)
        plt.gca().invert_yaxis()

        for foo in decorateFuns:
            foo(lon1,lat1,lon2,lat2)
            
        return fig,ax1,ax2

    def _plotBasemap(self,loc='Cascadia',ax=None):
        from Triforce.customPlot import plotLocalBase
        if loc=='Cascadia':
            minlon,maxlon,minlat,maxlat,dlon,dlat = -132,-121,39,50,2,3
        elif loc=='auto':
            minlon,maxlon,minlat,maxlat = self.lons[0],self.lons[-1],self.lats[0],self.lats[-1]
            dlat,dlon = (maxlat-minlat)//5,(maxlon-minlon)//3
        else:
            minlon,maxlon,minlat,maxlat,dlon,dlat = loc
        fig,m = plotLocalBase(minlon,maxlon,minlat,maxlat,dlat=dlat,dlon=dlon,resolution='l',ax=ax)
        m.readshapefile('/home/ayu/Projects/Cascadia/Models/Plates/PB2002_boundaries','PB2002_boundaries',
            linewidth=2.0,color='orange')
        return fig,m
    def plotMapView(self,mapTerm,loc='Cascadia',vmin=4.1,vmax=4.4,cmap=None):
        fig,m = self._plotBasemap(loc=loc)
        if mapTerm == 'misfit':
            cmap = plt.cm.gnuplot_r if cmap is None else cmap
            norm = mpl.colors.BoundaryNorm(np.linspace(0.5,3,6), cmap.N)
            misfits = np.array(self.misfits)
            misfits[misfits==None] = np.nan
            misfits = np.ma.masked_array(misfits.astype(float),mask=self.mask,fill_value=0)
            # misfits -= 0.10
            m.pcolormesh(self.XX-360*(self.XX[0,0]>180),self.YY,misfits,shading='gouraud',cmap=cmap,latlon=True,norm=norm)
            plt.title(f'Misfit')
            plt.colorbar(location='bottom',fraction=0.012,aspect=50)
        else:
            if cmap is None:
                cmap = cvcpt
            vsMap = self.mapview(mapTerm)
            m.pcolormesh(self.XX-360*(self.XX[0,0]>180),self.YY,vsMap,shading='gouraud',cmap=cmap,vmin=vmin,vmax=vmax,latlon=True)
            plt.title(f'Depth: {mapTerm} km')
            plt.colorbar(location='bottom',fraction=0.012,aspect=50)
        return fig,m

    def checkLayerThick(self):
        mask = self.mask
        hCrust      =  np.ma.masked_array(np.zeros(mask.shape),mask=mask)
        hCrust0     =  np.ma.masked_array(np.zeros(mask.shape),mask=mask)
        hCrustBias  =  np.ma.masked_array(np.zeros(mask.shape),mask=mask)
        hSed        =  np.ma.masked_array(np.zeros(mask.shape),mask=mask)
        hSed0       =  np.ma.masked_array(np.zeros(mask.shape),mask=mask)
        hSedBias    =  np.ma.masked_array(np.zeros(mask.shape),mask=mask)
        m,n = len(self.lats),len(self.lons)
        for i in range(m):
            for j in range(n):
                if not mask[i,j]:
                    mod,mod0 = self.mods[i][j],self._mods_init[i][j]
                    indCrust = [l.type for l in mod.layers].index('crust')
                    indSed   = [l.type for l in mod.layers].index('sediment')
                    hCrust[i,j]     = mod.layers[indCrust].H
                    hCrust0[i,j]    = mod0.layers[indCrust].H
                    hSed[i,j]       = mod.layers[indSed].H
                    hSed0[i,j]      = mod0.layers[indSed].H
                    hCrustBias[i,j] = (hCrust[i,j] - hCrust0[i,j])/(hCrust0[i,j])*100
                    if mod.info['label'][:6] == 'Hongda' and hSed0[i,j]<1.0:
                        hSedBias[i,j]   = (hSed[i,j]-hSed0[i,j])/1.0*100
                    else:
                        hSedBias[i,j]   = (hSed[i,j]-hSed0[i,j])/hSed0[i,j]*100

        fig,m = self._plotBasemap(loc='auto')
        m.pcolormesh(self.XX-360,self.YY,hCrust,shading='gouraud',cmap=cvcpt,latlon=True)
        plt.title(f'Crust Thickness')
        plt.colorbar(location='bottom',fraction=0.012,aspect=50)
        plt.savefig('CrustH.png')

        fig,m = self._plotBasemap(loc='auto')
        m.pcolormesh(self.XX-360,self.YY,hCrustBias,shading='gouraud',cmap=cvcpt,latlon=True)
        plt.title(f'Crust Thickness Difference from Initial Model (%)')
        plt.colorbar(location='bottom',fraction=0.012,aspect=50)
        plt.clim(-25,25)
        plt.savefig('CrustH-Change.png')

        fig,m = self._plotBasemap(loc='auto')
        m.pcolormesh(self.XX-360,self.YY,hSed,shading='gouraud',cmap=cvcpt,latlon=True)
        plt.title(f'Sediment Thickness')
        plt.colorbar(location='bottom',fraction=0.012,aspect=50)
        plt.savefig('SedimentH.png')

        fig,m = self._plotBasemap(loc='auto')
        m.pcolormesh(self.XX-360,self.YY,hSedBias,shading='gouraud',cmap=cvcpt,latlon=True)
        plt.title(f'Sediment Thickness Difference from Initial Model (%)')
        plt.colorbar(location='bottom',fraction=0.012,aspect=50)
        plt.clim(-100,100)
        plt.savefig('SedimentH-Change.png')
    def checkPhaseVelocity(self,pers='all',savefig=False):
        from Triforce.customPlot import addCAxes
        vminmax = {
                '010s':(3.0,4.0),
                '012s':(3.0,4.0),
                '014s':(3.0,4.0),
                '016s':(3.0,4.0),
                '018s':(3.1,4.0),
                '020s':(3.3,4.0),
                '022s':(3.3,4.0),
                '024s':(3.3,4.0),
                '026s':(3.3,4.0),
                '028s':(3.3,4.0),
                '030s':(3.4,4.0),
                '032s':(3.5,4.0),
                '036s':(3.6,4.0),
                '040s':(3.6,4.0),
                '050s':(3.6,4.0),
                '060s':(3.7,4.0),
                '070s':(3.7,4.0),
                '080s':(3.8,4.05)
                }
        if pers == 'all':
            pers = []
            for disp in self.disps.reshape(-1):
                if disp is not None:
                    pers.extend(disp['T'])
            pers = list(set(pers)); pers.sort()
        disps = {} 
        for iper,per in enumerate(pers):
            print(per)
            Tstr = f'{int(per):03d}s'
            vmin,vmax = vminmax[Tstr] if Tstr in vminmax.keys() else (None,None)
            m,n = self.XX.shape
            pvelo = np.ma.masked_array(np.zeros(self.XX.shape),mask=self.mask)
            pvelp = np.ma.masked_array(np.zeros(self.XX.shape),mask=self.mask)
            uncer = np.ma.masked_array(np.zeros(self.XX.shape),mask=self.mask)
            for i in range(m):
                for j in range(n):
                    if self.mask[i,j]:
                        continue
                    disp = self.disps[i][j]
                    ind = disp['T'].index(int(Tstr[:-1]))
                    pvelo[i,j] = disp['pvelo'][ind]
                    pvelp[i,j] = disp['pvelp'][ind]
                    uncer[i,j] = disp['uncer'][ind]
            disps[per] = {'pvelo':pvelo,'pvelp':pvelp}

            fig, axes = plt.subplots(1,3,figsize=[12,4.8])
            plt.subplots_adjust(wspace=0.25,hspace=0.3,left=0.08,right=0.92,bottom=0.15)
            _,m1 = self._plotBasemap(loc='auto',ax=axes[0])
            _,m2 = self._plotBasemap(loc='auto',ax=axes[1])
            _,m3 = self._plotBasemap(loc='auto',ax=axes[2])

            XX,YY = self.XX-360,self.YY
            im = m1.pcolormesh(XX,YY,pvelo,latlon=True,cmap=cvcpt,shading='gouraud',ax=axes[0],vmin=vmin,vmax=vmax)
            cax = addCAxes(axes[0],location='bottom',size=0.03,pad=0.20);plt.colorbar(im,cax=cax,orientation='horizontal')
            im = m2.pcolormesh(XX,YY,pvelp,latlon=True,cmap=cvcpt,shading='gouraud',ax=axes[1],vmin=vmin,vmax=vmax)
            cax = addCAxes(axes[1],location='bottom',size=0.03,pad=0.20);plt.colorbar(im,cax=cax,orientation='horizontal')
            im = m3.pcolormesh(XX,YY,(pvelp-pvelo)/uncer,latlon=True,cmap=cvcpt,shading='gouraud',ax=axes[2],vmin=-3,vmax=3)
            cax = addCAxes(axes[2],location='bottom',size=0.03,pad=0.20);plt.colorbar(im,cax=cax,orientation='horizontal')
            axes[0].set_title(f'Observation T={int(per):02d}s')
            axes[1].set_title(f'Prediction T={int(per):02d}s')
            axes[2].set_title(f'Pred-Obs (normed by uncer)')

            if savefig:
                plt.savefig(f'PhaseVel-{int(per):02d}s.png')
                plt.close()


if __name__ == '__main__':
    mod = Model3D(); mod.loadInvDir('example-Cascadia/mcdata')
    mod.write('invModel.npz')

    mod = Model3D()
    mod.load('invModel.npz')
    # print(mod.vsProfile(np.linspace(0,200,200),46.1,233.1))
    # print(mod.moho(46.1,233.1))
    # print(mod.topo(46.1,233.1))

    mod.smoothGrid(width=80)
    mod.write('invModel-Smooth.npz')

    mod = Model3D();mod.load('invModel-Smooth.npz')
    mod.plotMapView(100,'auto',vmin=4.0,vmax=4.5)
    mod.plotSection(-130.2+360,46,-125.6+360,46,maxD=200,title='Lat = 46$\degree$')
    xlim = plt.gcf().axes[0].get_xlim();plt.plot(xlim,(150,150),'--r')

