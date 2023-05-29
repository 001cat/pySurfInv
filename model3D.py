import glob
import numpy as np
from geographiclib.geodesic import Geodesic
from pySurfInv.models import Model1D,PureGird
from pySurfInv.point import PostPoint
from Triforce.utils import GeoGrid,GeoMap
from Triforce.pltHead import *
from Triforce.customPlot import cvcpt,addAxes,addCAxes
from Triforce.cartopy import plotLocalCart,crsPlate

def mapSmooth(lons,lats,z,tension=0.0, width=50.):
    zNew = GeoMap(lons,lats,z).smooth(tension=tension,width=width).z
    zNew[np.isnan(z)] = np.nan
    return zNew

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
    def _addInvPoint(self,lon,lat,postpoint:PostPoint):
        i,j = self._findInd(lon,lat)
        self.mods[i][j]     = postpoint.avgMod.copy()
        self._mods_init[i][j] = postpoint.initMod.copy()
        self.misfits[i][j]  = postpoint.avgMod.misfit
        self.disps[i][j]    = {'T':postpoint.obs['T'],
                               'pvelo':postpoint.obs['c'],
                               'pvelp':postpoint.avgMod.forward(postpoint.obs['T']),
                               'uncer':postpoint.obs['uncer']}
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
            print(f'Add point {ptlon:.1f}_{ptlat:.1f}')
            try:
                self._addInvPoint(ptlon,ptlat,PostPoint(npzfile))
            except Exception as e:
                print(f'Warning: {e}')

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
    def smoothGrid(self,width=50,nSeisProp=6,
                   nGridsDict={'water':2,'sediment':6,'prism':10,'crust':30,'mantle':200}):
        ''' To combine and smooth areas with different model settings '''
        def mod2grid(mod:Model1D):
            inProfiles = mod.seisPropGrids()
            outProfiles= [[] for _ in range(len(inProfiles))]
            
            grp = np.array(inProfiles[-1])
            for k,v in nGridsDict.items():
                I = grp == k
                for i in range(len(inProfiles)-1):
                    n = len(inProfiles[i][I])
                    if n == 0:
                        if i == 0: # depth
                            if outProfiles[0]:
                                outProfiles_seg = np.ones(v)*outProfiles[0][-1]
                            else:
                                outProfiles_seg = np.ones(v)*inProfiles[0][0]
                        else: # Vs, Vp, ...
                            outProfiles_seg = np.zeros(v)*np.nan
                    else:
                        outProfiles_seg = np.interp(np.linspace(0,1,v),np.linspace(0,1,n),inProfiles[i][I])
                    outProfiles[i].extend(list(outProfiles_seg))
                outProfiles[-1].extend([k]*v)
            outProfiles = [np.array(p) for p in outProfiles[:-1]] + outProfiles[-1:]
            # print(len(outProfiles[-1]))
            return PureGird(outProfiles,info=mod.copy().info)

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
                    mat[i,j,:,:] = np.array(self.mods[i][j].seisPropGrids(hLowerLimit=-1)[:-1])

        # remove all nan layer
        iLayerDelete = []
        for iLayer in range(mat.shape[-1]):
            for k in range(mat.shape[-2]):
                if np.all(np.isnan(mat[:,:,k,iLayer])):
                    iLayerDelete.append(iLayer)
                    break
        mat = np.delete(mat,iLayerDelete,-1)

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
                    # if np.any(np.isnan(np.sum(matSmooth[i,j,:,:],axis=0))):
                    #     from IPython import embed; embed()
                    matSmooth[i,j,0,np.isnan(np.sum(matSmooth[i,j,:,:],axis=0))] = 0
                    grp = self.mods[i][j].seisPropGrids(hLowerLimit=-1)[-1]
                    grp = np.delete(grp,iLayerDelete,-1)
                    inProfiles = [p for p in matSmooth[i,j,:,:]] + [grp]
                    self.mods[i][j] = PureGird(inProfiles,self.mods[i][j].info)

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

    ''' miscellaneous '''
    @property
    def mask(self):
        m,n = len(self.lats),len(self.lons)
        mask = np.ones((m,n),dtype=bool)
        for i in range(m):
            for j in range(n):
                mask[i,j] = (self.mods[i][j] is None)
        return mask
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

    ''' mapview plotting '''
    def _plotBasemap(self,ax=None,loc=None):
        if loc is None:
            minlon,maxlon,minlat,maxlat = self.lons[0],self.lons[-1],self.lats[0],self.lats[-1]
            minlon,maxlon = minlon-360*(minlon>180),maxlon-360*(maxlon>180)
            lat0,lon0 = minlat,minlon
            dlat,dlon = (maxlat-minlat)//5,(maxlon-minlon)//3
        else:
            if len(loc) == 4:
                minlon,maxlon,minlat,maxlat = loc; minlon,maxlon = minlon-360*(minlon>180),maxlon-360*(maxlon>180)
                lat0,lon0 = minlat,minlon
                dlat,dlon = (maxlat-minlat)//5,(maxlon-minlon)//3
            else:
                minlon,maxlon,minlat,maxlat,dlon,dlat,lon0,lat0 = loc
        ax,crsPlate = plotLocalCart(minlon,maxlon,minlat,maxlat,ax=ax,gridlines={'dlat':dlat,'dlon':dlon,'lat0':lat0,'lon0':lon0})
        return ax,crsPlate
    def _genMap(self,foo,**kwargs):
        mask = self.mask.copy()
        v = np.ma.masked_array(np.zeros(mask.shape),mask=mask)
        m,n = len(self.lats),len(self.lons)
        for i in range(m):
            for j in range(n):
                if not mask[i,j]:
                    v[i,j] = foo(self.mods[i][j],**kwargs)
        return GeoMap(lons=self.lons,lats=self.lats,z=v,mask=mask)
    def genVsMap(self,zdepth):
        def _vs_(mod:Model1D,zdepth):
            return mod.value(zdepth)
        return self._genMap(_vs_,zdepth=zdepth)
    def genVsAvgMap(self,zdeps):
        def _vs_(mod:Model1D,zdeps):
            return mod.value(zdeps).mean()
        return self._genMap(_vs_,zdeps=zdeps)
    def plotMapView(self,mapVar='misfit',cmap=None,vmin=None,vmax=None):
        ax,crsPlate = self._plotBasemap()
        if mapVar == 'misfit':
            cmap = plt.cm.YlOrBr if cmap is None else cmap
            misfits = np.array(self.misfits)
            misfits[misfits==None] = np.nan
            misfits = np.ma.masked_array(misfits.astype(float),mask=self.mask,fill_value=0)
            plt.pcolormesh(self.XX,self.YY,misfits,shading='gouraud',cmap=cmap,transform=crsPlate,
                           norm=mpl.colors.BoundaryNorm([0,1.0,1.5,2.0,2.5,3.0],cmap.N))
            plt.title(f'Misfit')
        else:
            geoMap = mapVar; zMask = np.ma.masked_array(geoMap.z,mask=geoMap.mask)
            plt.pcolormesh(geoMap.XX,geoMap.YY,geoMap.zMasked,shading='gouraud',transform=crsPlate,
                           cmap=cmap,vmin=vmin,vmax=vmax)
        cax = addCAxes(plt.gca(),location='bottom')
        plt.colorbar(cax=cax,orientation='horizontal')
        plt.sca(ax)
        return ax,cax

    ''' vertical profile plotting '''
    def section(self,lon1,lat1,lon2,lat2,y=np.linspace(0,200-0.01,201),xtype='auto'):
        # lon1,lat1,lon2,lat2 = -131+360,46,-125+360,43.8
        geoDict = Geodesic.WGS84.Inverse(lat1,lon1,lat2,lon2)
        x = np.linspace(0,geoDict['s12'],301)/1000
        z = np.zeros((len(y),len(x)))
        moho = np.zeros(len(x))
        topo = np.zeros(len(x))
        for i,d in enumerate(x*1000):
            tmp = Geodesic.WGS84.Direct(lat1,lon1,geoDict['azi1'],d)
            z[:,i] = self.vsProfile(y,tmp['lat2'],tmp['lon2'])
            moho[i]= self.moho(tmp['lat2'],tmp['lon2'])
            topo[i]= self.topo(tmp['lat2'],tmp['lon2'])
        z = np.ma.masked_array(z,np.isnan(z))
        if xtype == 'lat' or (xtype == 'auto' and abs(lon1-lon2)<0.01):
            x = np.linspace(lat1,lat2,301)
        elif xtype == 'lon' or (xtype == 'auto' and abs(lat1-lat2)<0.01):
            x = np.linspace(lon1,lon2,301)
        elif xtype in ('km','auto'):
            pass
        else:
            raise ValueError(f'Wrong xtype: {xtype}')
        XX,YY = np.meshgrid(x,y)
        return XX,YY,z,moho,topo
    def section_rel(self,lon1,lat1,lon2,lat2,y=np.linspace(0,200-0.01,201),xtype='auto'):
        XX,YY,z,moho,topo = self.section(lon1,lat1,lon2,lat2,y,xtype)
        if not hasattr(self,'_zAvg'): 
            # fail near interface, should use Vs averaged with in same group, not same depth
            self._zAvg = np.array([self.genVsMap(dep).zMasked.mean() for dep in YY[:,0]])
        zAvg2D = np.repeat(self._zAvg.reshape((len(self._zAvg),1)),z.shape[1],axis=1)
        return XX,YY,(z - zAvg2D)/zAvg2D*100,moho,topo
    def plotSection(self,lon1,lat1,lon2,lat2,ax=None,cmap=cvcpt,maxD=200,label=None,
                    rel=False,trueAspect=False,cax=True,decorateFuns=[],figsize=[12,5]):
        ySep = 15; zoom = 3; 
        if rel:
            vLimC = [-5,5];    vLimM = [-5,5]
        else:
            vLimC = [3.0,4.0]; vLimM = [4.0,4.5]
        
        y = np.zeros(300); y[:100] = np.linspace(0,ySep,100); y[100:] = np.linspace(ySep,maxD-0.01,200)
        
        if rel:
            XX,YY,Z,moho,topo = self.section_rel(lon1,lat1,lon2,lat2,y=y) # show relative
        else:
            XX,YY,Z,moho,topo = self.section(lon1,lat1,lon2,lat2,y=y)   # show absolute

        def calYZoom(yIn):
            yOut = yIn.copy()
            yOut[yIn<ySep]  *= zoom
            yOut[yIn>=ySep] += ySep*(zoom-1)
            return yOut
        isTop = y<ySep
        YY_top,YY_bot = YY*zoom,YY + ySep*(zoom-1)
        Y_moho,Y_topo = calYZoom(moho),calYZoom(-topo)
        Z_crust = np.ma.masked_array(Z,mask=(YY > np.tile(moho,(YY.shape[0],1))))

        if ax:
            plt.axes(ax)
        else:
            plt.figure(figsize=figsize); ax = addAxes([0,0.2,1,0.8])

        imM = plt.pcolormesh(XX,YY_bot,Z,shading='gouraud',cmap=cvcpt,vmin=vLimM[0],vmax=vLimM[1],rasterized=True)
        plt.pcolormesh(XX[isTop,:],YY_top[isTop,:],Z[isTop,:],shading='gouraud',cmap=cvcpt,vmin=4.0,vmax=4.5,rasterized=True)
        imC = plt.pcolormesh(XX,YY_top,Z_crust,shading='gouraud',cmap=cvcpt,vmin=vLimC[0],vmax=vLimC[1],rasterized=True)
        plt.fill_between(XX[0,:],0,Y_topo,facecolor='#d4f1f9')
        plt.plot(XX[0,:],Y_moho,'k',lw=4)
        plt.plot(XX[0,:],Y_moho,'r',lw=2)
        plt.ylim(0,maxD+(zoom-1)*ySep)
        plt.gca().invert_yaxis()

        if zoom != 1:
            yticks = np.array(list(set(list(range(0,maxD+10,50)) + [ySep] + [maxD]))); yticks.sort()
            plt.yticks(calYZoom(yticks),[str(v) for v in yticks])
            import matplotlib.patheffects as pe
            plt.plot(plt.xlim(),[ySep*zoom]*2,'--',color='w',alpha=1.0,lw=2,
                    path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()])
        
        for foo in decorateFuns:
            foo(lon1,lat1,lon2,lat2)

        if label is not None:
            x0,x1 = ax.get_xlim()
            y0,y1 = ax.get_ylim()
            plt.text(x0,y1,label[0],va='bottom',ha='center',fontweight='bold',fontsize=20,clip_on=False,zorder=100)
            plt.text(x1,y1,label[1],va='bottom',ha='center',fontweight='bold',fontsize=20,clip_on=False,zorder=100)

        if trueAspect:
            dist = Geodesic.WGS84.Inverse(lat1,lon1,lat2,lon2)['s12']/1000
            w0 = ax.get_position().width*plt.gcf().get_figwidth()
            h0 = ax.get_position().height*plt.gcf().get_figheight()
            w1 = dist/(maxD+ySep*(zoom-1))*h0/plt.gcf().get_figwidth()
            box = ax.get_position()
            box.intervalx[1] = box.intervalx[0] + w1
            ax.set_position(box)

        if cax:
            cax1 = addCAxes(ax,location='bottom',size=0.03,pad=0.13)
            cax2 = addCAxes(ax,location='bottom',size=0.03,pad=0.25)
            plt.colorbar(imC,cax=cax1,orientation='horizontal')
            plt.colorbar(imM,cax=cax2,orientation='horizontal')

        return imC,imM
    
    '''specific methods'''
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

            fig, axes = plt.subplots(1,3,figsize=[12,4.8],subplot_kw={'projection':crsPlate})
            plt.subplots_adjust(wspace=0.25,hspace=0.3,left=0.08,right=0.92,bottom=0.15)

            XX,YY = self.XX-360,self.YY
            self._plotBasemap(ax=axes[0])
            im = plt.pcolormesh(XX,YY,pvelo,cmap=cvcpt,shading='gouraud',vmin=vmin,vmax=vmax,transform=crsPlate)
            cax = addCAxes(axes[0],location='bottom',size=0.03,pad=0.20);plt.colorbar(im,cax=cax,orientation='horizontal')
            self._plotBasemap(ax=axes[1])
            im = plt.pcolormesh(XX,YY,pvelp,cmap=cvcpt,shading='gouraud',vmin=vmin,vmax=vmax,transform=crsPlate)
            cax = addCAxes(axes[1],location='bottom',size=0.03,pad=0.20);plt.colorbar(im,cax=cax,orientation='horizontal')
            self._plotBasemap(ax=axes[2])
            im = plt.pcolormesh(XX,YY,(pvelp-pvelo)/uncer,cmap=cvcpt,shading='gouraud',vmin=-3,vmax=3,transform=crsPlate)
            cax = addCAxes(axes[2],location='bottom',size=0.03,pad=0.20);plt.colorbar(im,cax=cax,orientation='horizontal')
            axes[0].set_title(f'Observation T={int(per):02d}s')
            axes[1].set_title(f'Prediction T={int(per):02d}s')
            axes[2].set_title(f'Pred-Obs (normed by uncer)')

            if savefig:
                plt.savefig(f'PhaseVel-{int(per):02d}s.png')
                plt.close()

    ''' deprecated '''
    def plotSection_old(self,lon1,lat1,lon2,lat2,vCrust=[3.0,4.0],vMantle=[4.0,4.5],cmap=cvcpt,
                    maxD=200,shading='gouraud',title=None,decorateFuns=[],figsize=[12,5],
                    profile_labels=None,rel=False,real_ratio=False):
        if rel:
            XX,YY,Z,moho,topo = self.section_rel(lon1,lat1,lon2,lat2) # show relative
        else:
            XX,YY,Z,moho,topo = self.section(lon1,lat1,lon2,lat2)   # show absolute

        mask = ~(YY <= np.tile(moho,(YY.shape[0],1)))
        Z_crust = np.ma.masked_array(Z,mask=mask)

        # create axes frame, ax1 for topo, ax2 for structure. cax1: crust in ax2. cax2: mantle in ax2
        fig = plt.figure(figsize=figsize)
        ax2 = addAxes([0,0.18,1,0.65])
        bbox = ax2.get_position()
        x,y,w,h = bbox.x0,bbox.y0,bbox.width,bbox.height
        ax1 = plt.axes([x,y+h+0.05*h,w,h/6],sharex=ax2)
        ax1.axes.xaxis.set_visible(False)
        cax1 = addCAxes(ax2,location='bottom',size=0.05,pad=0.08)
        cax2 = addCAxes(ax2,location='bottom',size=0.05,pad=0.20)

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
        
        if title is not None:
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

        if real_ratio:
            dist = Geodesic.WGS84.Inverse(lat1,lon1,lat2,lon2)['s12']/1000
            w0 = ax2.get_position().width*plt.gcf().get_figwidth()
            h0 = ax2.get_position().height*plt.gcf().get_figheight()
            w1 = dist/maxD*h0/plt.gcf().get_figwidth()
            for ax in [ax1,ax2,cax1,cax2]:
                box = ax.get_position()
                box.intervalx[1] = box.intervalx[0] + w1
                ax.set_position(box)

        if profile_labels:
            plt.axes(ax1)
            x0,x1 = ax1.get_xlim()
            y0,y1 = ax1.get_ylim()
            plt.text(x0,y1,profile_labels[0],va='bottom',ha='center',fontweight='bold',fontsize=20,clip_on=False,zorder=100)
            plt.text(x1,y1,profile_labels[1],va='bottom',ha='center',fontweight='bold',fontsize=20,clip_on=False,zorder=100)

        return fig,ax1,ax2,cax1,cax2





if __name__ == '__main__':
    # invModel = Model3D(lons=np.arange(228,243+0.1,0.2),lats=np.arange(39,50+0.1,0.2))
    # invModel.loadInvDir('example-Cascadia/mcdata');invModel.write('tmp_model3D.npz')
    # invModel.smoothGrid(80);invModel.write('tmp_model3D_smooth.npz')

    invModel = Model3D(); invModel.load('tmp_model3D_smooth.npz')

    # invModel._plotBasemap()
    # vsMap = invModel.genVsMap(40); vsMap._lon_range_change_to('-180 to 180')
    # vsMapAvg = invModel.genVsAvgMap(np.linspace(70,100,30)); vsMapAvg._lon_range_change_to('-180 to 180')
    # invModel.plotMapView('misfit')
    # invModel.plotMapView(invModel.genVsMap(40))
    # invModel.checkPhaseVelocity()

    # invModel.plotSection(229.8,47.0,233.8,47.0,trueAspect=True)
    # invModel.plotSection(229.8,47.0,233.8,47.0,rel=True,trueAspect=False)




