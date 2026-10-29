# Shared look for all charts (same palette/fonts as the LFM3 audit kit thread charts). 1200x675 px.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
BG='#0f1419'; FG='#e7e9ea'; MUT='#8b98a5'; GRID='#2a3139'
ORIG='#4cc9f0'; ABZ='#f72585'; CLEAN='#80ed99'; NOISE='#6c757d'; WARN='#ffd166'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':18,'text.color':FG,'axes.labelcolor':FG,
  'xtick.color':FG,'ytick.color':FG,'axes.facecolor':BG,'figure.facecolor':BG,'axes.edgecolor':GRID,
  'axes.titlesize':26,'axes.titleweight':'bold'})
def fig(): return plt.figure(figsize=(12,6.75),dpi=100)
def title(f,t,sub=None):
    f.text(0.04,0.93,t,fontsize=28,weight='bold',color=FG,va='top')
    if sub: f.text(0.04,0.855,sub,fontsize=17,color=MUT,va='top')
def foot(f,t='Source: expected_outputs/ in the repro kit',fs=12): f.text(0.04,0.018,t,fontsize=fs,color=MUT,va='bottom',linespacing=1.35)
def save(f,path): f.savefig(path,dpi=100,facecolor=BG); plt.close(f)
def ax_style(a):
    for s in ['top','right']: a.spines[s].set_visible(False)
    a.grid(axis='y',color=GRID,lw=1); a.set_axisbelow(True)
def overlay(f):
    a=f.add_axes([0,0,1,1]); a.axis('off'); a.set_xlim(0,1); a.set_ylim(0,1); return a
def box(f,x,y,w,h,txt,col,fs=17,ha='center',a=None):
    a=a or overlay(f)
    a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.01,rounding_size=0.02',fc=col+'22',ec=col,lw=2.5))
    if txt: a.text(x+w/2 if ha=='center' else x+0.015,y+h/2,txt,ha=ha,va='center',fontsize=fs,color=FG,linespacing=1.4)
    return a
def arrow(a,x1,y1,x2,y2,c=MUT):
    a.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='-|>',color=c,lw=2.5,mutation_scale=22))
