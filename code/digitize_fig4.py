import numpy as np, pandas as pd
from svgpaths import load
P=load('../refs/page8.svg')
names={'rgb(0%, 0%, 0%)':1,'rgb(0%, 0%, 100%)':5,'rgb(100%, 0%, 0%)':10,'rgb(0%, 100%, 0%)':15}
frames={'filter1_720-870':(182.77,309.02),'filter2_800-1000':(336.91,463.16)}
rows=[]
for p in P:
    if not p['stroke'] or len(p['pts'])<300: continue
    q=p['pts']; x0=q[:,0].min()
    if x0<170: continue
    key=[k for k,(a,b) in frames.items() if a-2<x0<b][0]; a,b=frames[key]
    t=(q[:,0]-a)/(b-a)*30; T=(129.82-q[:,1])/(129.82-3.58)*25
    o=np.argsort(t); tg=np.arange(0,30.01,0.25)
    for u,v in zip(tg,np.interp(tg,t[o],T[o])): rows.append((key,names[p['stroke']],u,v))
df=pd.DataFrame(rows,columns=['filter','depth_mm','t_min','dT_C']); df.to_csv('../data/pratap_fig4_plaingel_dT.csv',index=False)
print(df[df.t_min==15].pivot(index='filter',columns='depth_mm',values='dT_C').round(2))
print(df[df.t_min==30].pivot(index='filter',columns='depth_mm',values='dT_C').round(2))
