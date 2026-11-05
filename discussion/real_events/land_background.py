"""Offline Natural Earth physical land only; no political boundaries."""
import json
from pathlib import Path
from functools import lru_cache
import numpy as np
from matplotlib.patches import Polygon

SOURCE=Path(__file__).resolve().parents[2]/'plotting/map_data/ne_10m_land.geojson'


@lru_cache(maxsize=1)
def rings():
    data=json.loads(SOURCE.read_text(encoding='utf-8'))
    result=[]
    for feature in data['features']:
        g=feature['geometry']
        polys=g['coordinates'] if g['type']=='MultiPolygon' else [g['coordinates']]
        for polygon in polys:
            for i,ring in enumerate(polygon):
                a=np.asarray(ring)
                result.append((a,i>0,(a[:,0].min(),a[:,0].max(),a[:,1].min(),a[:,1].max())))
    return result


def clip_polygon(vertices,bounds):
    pts=vertices.tolist()
    for axis,edge,sign in [(0,bounds[0],1),(0,bounds[1],-1),(1,bounds[2],1),(1,bounds[3],-1)]:
        if not pts:break
        output=[];prev=pts[-1];pin=(prev[axis]-edge)*sign>=0
        for cur in pts:
            cin=(cur[axis]-edge)*sign>=0
            if cin!=pin:
                ratio=(edge-prev[axis])/(cur[axis]-prev[axis])
                output.append([prev[j]+ratio*(cur[j]-prev[j]) for j in (0,1)])
            if cin:output.append(cur)
            prev,pin=cur,cin
        pts=output
    return pts


def add_land_sea(ax):
    x=ax.get_xlim();y=ax.get_ylim();bounds=(*x,*y)
    ax.set_facecolor('#EAF3F8')
    for vertices,hole,bbox in rings():
        if bbox[1]<x[0] or bbox[0]>x[1] or bbox[3]<y[0] or bbox[2]>y[1]:continue
        clipped=clip_polygon(vertices,bounds)
        if len(clipped)>=3:
            ax.add_patch(Polygon(clipped,closed=True,facecolor='#EAF3F8' if hole else '#F2EFE5',
                                 edgecolor='#8C918C',lw=.45,zorder=-10,clip_on=True))
    ax.set_xlim(x);ax.set_ylim(y)
