import numpy as np, pandas as pd
from svgpaths import load
P=load('../refs/page6.svg')
fr={'a':(27.03,171.45,148.86,4.44,0.5),'b':(220.16,364.58,149.34,4.91,1.0)}
lab={('a','rgb(0%, 0%, 0%)'):'NR1',('a','rgb(100%, 0%, 0%)'):'NR1+4%BSA',('b','rgb(0%, 0%, 0%)'):'NR2',('b','rgb(100%, 0%, 0%)'):'NR2+4%BSA'}
rows=[]
for p in P:
    q=p['pts']
    if not p['stroke'] or len(q)<300: continue
    k='a' if q[:,0].min()<100 else 'b'; x0,x1,y0,y1,ym=fr[k]
    lam=400+(q[:,0]-x0)/(x1-x0)*700; A=(y0-q[:,1])/(y0-y1)*ym
    o=np.argsort(lam); g=np.arange(400,1100.1,2)
    for u,v in zip(g,np.interp(g,lam[o],A[o])): rows.append((lab[(k,p['stroke'])],u,v))
d=pd.DataFrame(rows,columns=['sample','lam_nm','absorbance']); d.to_csv('../data/pratap_fig2_absorbance.csv',index=False)
print(d.loc[d.groupby('sample').absorbance.idxmax() if False else d[d.lam_nm>650].groupby('sample').absorbance.idxmax()])
