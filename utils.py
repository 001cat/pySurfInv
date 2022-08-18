import numpy as np
from netCDF4 import Dataset
from Triforce.utils import GeoMap
from Triforce.pltHead import *

from IPython import embed

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

def plotCascadiaTrenchCoast(lon1,lat1,lon2,lat2):
    lon1 = lon1-360*(lon1>180)
    lon2 = lon2-360*(lon2>180)
    from shapely.geometry import LineString
    from geographiclib.geodesic import Geodesic 
    a = LineString(np.loadtxt('prism.csv',delimiter=','))
    b = LineString([(-132,45),(-118,45)])
    p1,p2 = a.intersection(b).geoms
    if p1.x > p2.x:
        p1,p2 = p2,p1
    for p in [p1,p2]:
        if abs(lon1-lon2)<0.01:
            x = p.xy[1][0]
        elif abs(lat1-lat2)<0.01:
            x = p.xy[0][0]+360
        else:
            x = Geodesic.WGS84.Inverse(lat1,lon1,p.xy[1][0],p.xy[0][0])['s12']/1000
        plt.plot([x,x],[0,200],'--',c='r',lw=0.5)
    



def plotMORLocation(lon1,lat1,lon2,lat2):
    lon1 = lon1-360*(lon1>180)
    lon2 = lon2-360*(lon2>180)
    import geopandas as gpd
    from shapely.geometry import LineString
    df = gpd.read_file('/home/ayu/Projects/Cascadia/Models/Plates/GeoJSON/PB2002_boundaries.json')
    mor = df.loc[df['Name']=='PA-JF'].geometry.values[0]
    p = mor.intersection(LineString([(lon1,lat1),(lon2,lat2)]))

    # get x coordinate in ax, only lat, only lon or using distance
    from geographiclib.geodesic import Geodesic 
    if abs(lon1-lon2)<0.01:
        x = p.xy[1][0]
    elif abs(lat1-lat2)<0.01:
        x = p.xy[0][0]+360
    else:
        x = Geodesic.WGS84.Inverse(lat1,lon1,p.xy[1][0],p.xy[0][0])['s12']/1000
    plt.plot(x,0,'^',markersize=10,markerfacecolor='r',clip_on=False,zorder=100)

def plotCascadiaSlab4Map(m,levels=[60,75,90,120,150]):
    with Dataset('/home/ayu/Projects/Cascadia/Models/Slab2_Cascadia/cas_slab2_dep_02.24.18.grd') as dset:
        slabDep = GeoMap(dset['x'][()],dset['y'][()],-dset['z'][()])
    XX,YY = np.meshgrid(slabDep.lons-360,slabDep.lats)
    cs = m.contour(XX,YY,slabDep.z,latlon=True,levels=levels,colors='white',linewidths=2)
    plt.clabel(cs, fontsize=9, inline=True,colors='k')


def plotLayer(h,v,fig=None,ax=None,label=None,**kwargs):
    if ax is None:
        fig = plt.figure(figsize=[5,7]);ax = plt.gca()
    else:
        plt.sca(ax); fig = plt.gcf()
    hNew = np.insert(np.repeat(np.cumsum(h),2)[:-1],0,0)
    vNew = np.repeat(v,2)
    ax.plot(vNew,hNew,label=label,**kwargs)
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    return ax
def plotGrid(zdepth,v,fig=None,ax=None,label=None,**kwargs):
    if ax is None:
        fig = plt.figure(figsize=[5,7]);ax = plt.gca()
    else:
        plt.sca(ax); fig = plt.gcf()
    plt.plot(v,zdepth,label=label,**kwargs)
    ax = ax or fig.axes[0]
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    return ax



def _dictIterModifier(d,checker,modifier):
    if type(d) is dict:
        dOut = {}
        for k,v in d.items():
            if checker(v):
                dOut[k] = modifier(v)
            elif type(v) in (dict,list):
                dOut[k] = _dictIterModifier(v,checker,modifier)
            else:
                dOut[k] = v
    elif type(d) is list:
        dOut = []
        for v in d:
            if checker(v):
                dOut.append(modifier(v))
            elif type(v) in (dict,list):
                dOut.append(_dictIterModifier(v,checker,modifier))
            else:
                dOut.append(v)
    else:
        dOut = d
    return dOut


if __name__ == '__main__':
    import yaml
    from brownian import BrownianVar
    def checker(v):
        return type(v) is list and len(v)>=2 \
            and v[1] in ('fixed','total','abs','rel')
    def modifier(v):
        if v[1] in ('fixed','total'):
            return v[0]
        elif v[1] == 'abs':
            return BrownianVar(v[0],v[0]-v[2],v[0]+v[2],v[3])
        elif v[1] == 'rel':
            return BrownianVar(v[0],v[0]*(1-v[2]/100),v[0]*(1+v[2]/100),v[3])
        else:
            raise ValueError(f'Error: Wrong checker??? v={v}')
    with open('setting-Hongda.yml', 'r') as f:
        d = _dictIterModifier(yaml.load(f,Loader=yaml.FullLoader),checker,modifier)
