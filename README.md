# VisualNavigation  
##Simple Social Force  
###Introduction  
Implement [Helbing 1995](http://vision.cse.psu.edu/courses/Tracking/vlpr12/HelbingSocialForceModel95.pdf) 
social force model. You will need virtual environment to run this program.  
After activate the virtual environment in `sf/`, you can run `python fancy_test.py` to test the 
test case in the paper.  
###Issue  
  1.There are some local energy minimum in the map, so pedestrians can sometime stuck in the middle of the screen for
a while. However, this can be solved by changing obstacle energy to continuous function (different from what is written in the paper).  
  2.I didn't implement attraction force since I am sure it won't work. I will implement it in next version.  
  3.Future work will focus on more complicated social force models. Also, implement the evacuation test.  
##LTA  
###Introduction  
Implement [Pellegrini 2009](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.491.1964&rep=rep1&type=pdf) 
LTA social energy minimization model. You will use the same virtual enviornment as Simple Social Force model 
to run this program.  
Afer activate the virtual environment in `sf/`, you can run `python fancy_test.py` to test the test case 
without obstacles.  
###Issue  
  1.It is very descritize when viewing the result. This is simply because the parameters in the model is 2.5fps. 
To change the frame rate, I will have to update other well-tuned parameters. Therefore, I left this for future work.  
  2.In very rare situation, the program might crash due to divide in zero issue. It will happen when two random float 
number are exactly the same. Therefore, it is too rare that I didn't deal with it.  
  3.The program uses RMSprop to minimize the energy function. ADAM and other gradient descent methods should be 
tested in the future. Also, the parameters in RMSprop are not the optimal converging one. Future work should focus on 
update these parameters and test their speed while increasing the number of pedestrians (I use numpy, so should not change 
much).  
  4.I didn't compare the run time of brute force calculating energy and gradient(which is implemented in `unit_test.py`) 
with vectorized version. Future work should also test their speed difference while changing the number of pedestrians.  

