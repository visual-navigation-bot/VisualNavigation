# VisualNavigation  
##Simple Social Force  
###Introduction  
Implement [Helbing 1995](http://vision.cse.psu.edu/courses/Tracking/vlpr12/HelbingSocialForceModel95.pdf) 
social force model. You will need virtual environment to run this program.  
After activate the social environment in '''sf/''', you can run '''python fancy_test.py''' to test the 
test case in the paper.  
###Issue  
  1.There are some local energy minimum in the map, so pedestrians can sometime stuck in the middle of the screen for
a while. However, this can be solved by changing obstacle energy to continuous function (different from what is written in the paper).  
  2.I didn't implement attraction force since I am sure it won't work. I will implement it in next version.  
  3.Future work will focus on more complicated social force models. Also, implement the evacuation test.  

