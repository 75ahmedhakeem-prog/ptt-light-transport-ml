from PIL import Image; import numpy as np, pandas as pd
cfg={"a":("fig8-000.png",(131.5,664.5,700,1000),(457.5,35.5,0.20)),"b":("fig8-001.png",(130.5,664.5,800,1000),(457.5,41.5,0.75))}
rows=[]
for key,(f,(xa,xb,la,lb),(y0,y1,v1)) in cfg.items():
    a=np.array(Image.open("../refs/"+f).convert("RGB")).astype(int)
    red=(a[:,:,0]>190)&(a[:,:,1]<90)&(a[:,:,2]<90); blk=a.sum(2)<200
    for m in (red,blk): m[20:175,135:405]=False; m[:22,:]=False; m[:, :int(xa)+3]=False; m[int(y0)-2:,:]=False
    for color,m in (("red",red),("black",blk)):
        prev=None
        for x in range(int(xa)+3,int(xb)-1):
            ys=np.where(m[:,x])[0]
            if len(ys)==0: continue
            grp=np.split(ys,np.where(np.diff(ys)>3)[0]+1); cent=sorted([g.mean() for g in grp])
            lam=la+(x-xa)/(xb-xa)*(lb-la); val=[(y0-c)/(y0-y1)*v1 for c in cent]
            rows.append((key,color,lam,val))
d=pd.DataFrame(rows,columns=["panel","color","lam","vals"])
# assignment: two curves per colour -> scattering = larger value where 2 clusters; where 1 cluster ambiguous
out=[]
for (p,c),g in d.groupby(["panel","color"]):
    for _,r in g.iterrows():
        v=r.vals
        if len(v)>=2: out.append((p,c,r.lam,min(v),max(v)))
o=pd.DataFrame(out,columns=["panel","color","lam","low","high"])
o.to_csv("../data/pratap_fig8_raw.csv",index=False)
def band(p,c,a,b):
    s=o[(o.panel==p)&(o.color==c)&(o.lam>=a)&(o.lam<=b)]
    return s.low.median(), s.high.median(), len(s)
for p,c,name,a,b in [("a","black","NR1 (720-870)",720,870),("a","red","NR1+4%BSA (800-1000)",800,1000),("a","red","NR1+4%BSA (800-950)",800,950),
                     ("b","black","NR2 (800-1000)",800,1000),("b","red","NR2+4%BSA (800-1000)",800,1000),("b","black","NR2 (800-950)",800,950)]:
    lo,hi,n=band(p,c,a,b); print(f"{name:24s} lower-curve median {lo:.3f}  upper-curve median {hi:.3f} mm^-1  (n={n})")
