import re, numpy as np
def parse_transform(t):
    if not t: return np.eye(3)
    m=re.search(r'matrix\(([^)]+)\)',t)
    a,b,c,d,e,f=[float(v) for v in re.split(r'[ ,]+',m.group(1).strip())]
    return np.array([[a,c,e],[b,d,f],[0,0,1]])
def path_points(d):
    toks=re.findall(r'[MLCZmlcz]|-?[\d.]+(?:e-?\d+)?',d)
    pts=[];i=0;cmd=None
    while i<len(toks):
        t=toks[i]
        if t in 'MLCZmlcz': cmd=t; i+=1; continue
        if cmd in 'ML': pts.append((float(toks[i]),float(toks[i+1]))); i+=2
        elif cmd=='C': pts.append((float(toks[i+4]),float(toks[i+5]))); i+=6
        else: i+=1
    return np.array(pts)
def load(svg):
    s=open(svg).read(); out=[]
    # handle nested <g transform> is rare in pdftocairo; paths carry transform
    for p in re.findall(r'<path[^>]*/?>',s):
        d=re.search(r' d="([^"]+)"',p)
        if not d: continue
        st=re.search(r'stroke="([^"]+)"',p); fi=re.search(r'fill="([^"]+)"',p)
        T=parse_transform((re.search(r'transform="([^"]+)"',p) or [None,None])[1] if re.search(r'transform="([^"]+)"',p) else None)
        P=path_points(d.group(1))
        if len(P)==0: continue
        Q=(T@np.c_[P,np.ones(len(P))].T).T[:,:2]
        out.append(dict(stroke=st.group(1) if st else None, fill=fi.group(1) if fi else None, pts=Q))
    return out
