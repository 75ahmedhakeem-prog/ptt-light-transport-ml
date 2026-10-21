import numpy as np, pandas as pd
from svgpaths import load
P=load('../refs/page9.svg')
big=[p for p in P if p['stroke'] and len(p['pts'])>500]
names={'rgb(0%, 0%, 0%)':1,'rgb(0%, 0%, 100%)':5,'rgb(100%, 0%, 0%)':10,'rgb(0%, 100%, 0%)':15}
# group into panels by order of appearance: each panel has 4 big curves
panels=[]; cur=[]
for p in big:
    cur.append(p)
    if len(cur)==4: panels.append(cur); cur=[]
tab={'NR1':[19.67,17.74,12.18,7.40],'NR1+4%BSA':[23.57,19.66,15.24,8.05],'NR2':[21.67,17.90,13.92,8.46],'NR2+4%BSA':[20.76,15.68,12.57,8.23]}
rows=[]
for name,pan in zip(tab,panels):
    x0=min(p['pts'][:,0].min() for p in pan); y0=np.median([p['pts'][0,1] for p in pan])
    # x of peak (end of heating) per curve
    xs=[]; sc=[]
    for p in pan:
        q=p['pts']; i=q[:,1].argmin(); z=names[p['stroke']]
        xs.append(q[i,0]); sc.append(tab[name][[1,5,10,15].index(z)]/(y0-q[i,1]))
    # t scale: heating stops at 15 min -> use black/blue peak location (sharp kink)
    xk=np.median(xs); kt=15/(xk-x0); a=np.mean(sc)
    print(name,'y-scale per curve',np.round(sc,4),'spread %',round(100*np.std(sc)/a,2),'xkink',np.round(xs,1))
    for p in pan:
        q=p['pts']; z=names[p['stroke']]
        t=(q[:,0]-x0)*kt; T=(y0-q[:,1])*a
        o=np.argsort(t); t,T=t[o],T[o]
        tg=np.arange(0,30.01,0.25); Tg=np.interp(tg,t,T,right=np.nan)
        for a_,b_ in zip(tg,Tg): rows.append((name,z,a_,b_))
df=pd.DataFrame(rows,columns=['sample','depth_mm','t_min','dT_C']); df.to_csv('../data/pratap_fig56_dT.csv',index=False)
print(df.groupby(['sample','depth_mm']).apply(lambda g: g.loc[(g.t_min-15).abs().idxmin(),'dT_C']).unstack().round(2))
