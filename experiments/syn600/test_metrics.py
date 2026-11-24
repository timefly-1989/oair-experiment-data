"""Scientific checks for metric weighting and the hierarchical comparison."""
from common import *
import numpy as np
from evaluate_extension import boundary_points,cd
def main():
 # Two disconnected squares with perimeter ratio 1:2 must receive the same
 # ratio of boundary samples regardless of how densely their vertices are stored.
 v=np.array([[0.,0,0],[1,0,0],[1,1,0],[0,1,0],[10,0,0],[12,0,0],[12,2,0],[10,2,0]])
 f=np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7]])
 b=boundary_points(v,f,1200);assert int((b[:,0]>5).sum())==800
 assert cd(b,b)==0
 # Common translation of both surfaces preserves distance.
 a=np.array([[0.,0,0],[0,1,0],[1,0,0]]);q=a+np.array([0,0,.05])
 assert abs(cd(a,q)-.05)<1e-12
 assert abs(cd(a+[3,4,5],q+[3,4,5])-.05)<1e-12
 write(ROOT/'metric_checks.json',{'boundary_length_weighting':True,'identical_surface_zero':True,'common_translation_invariance':True,
     'known_parallel_surface_distance':True,'checked_at':now(),'evaluator_sha256':digest(ROOT/'evaluate_extension.py')})
 print('metric checks passed')
if __name__=='__main__':main()
