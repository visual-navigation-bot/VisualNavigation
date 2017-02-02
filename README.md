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
##IGP  
###Introduction  
Implement [Trautman and Krause 2010](https://las.inf.ethz.ch/files/trautman10unfreezing.pdf) 
Interactive Gaussian Process model. I used some of the code in [This GitHub Link](https://github.com/ntraft/crowd-nav). 
You will not use virtual environment in this project since matplotlib library is not 
supported in virtual environment (though I only use matplotlib for testing).  
You will need the following python libraries: numpy, matplotlib, pygame.  
Afer installing the libraries mentioned above, you can run `python fancy_test.py` to test the test case 
without obstacles.  
###Issue  
  1.It is very descritize when viewing the result. This is simply because the parameters in the model is 2.5fps. 
To change the frame rate, I will have to update other well-tuned parameters. Therefore, I left this for future work.  
  2.I assume all pedestrian knows where and exactly when all other pedestrial will arrive its destination though they
 don't have other pedestrians' past path information before they walk in the field. The assumption is not true in reality, 
since no one know when and where other people are going (not even themselves know the exact time). Future work should fix this 
problem by adding noise.  
  3.It runs too slow for 40 pedestrians. This can probably be solved by running in Titan X.  
  4.It can't model obstacles and group affect. This can be improved by changing interaction weight.  
  5.It can't model the best path since it weighted average all sampled paths. If there are two optimal path, it will get there average 
, which is often much worse than optimal path. This can be solved by not using weighted average.  

##Social-LSTM
####Introduction
This is the implementation of [Alahi et al., 2016](http://vision.stanford.edu/pdf/CVPR16_N_LSTM.pdf) and is modified from [Anirudh Vemula work](https://github.com/vvanirudh/social-lstm-tf). A simulation tool is built based on the prediction ability over pedstrians of social lstm.
####Issue
1. since social-lstm is trained with non-consecutive data(data may not come from exactly continuous frames), interpolation may be carried out across two consecutive prediction. In my implementation, linear interpolation is used.
2. since social-lstm is a short-period prediction model, it is constructed as a simulation tool by making prediction based on its own previous prediction. This may cause error propagate and accumulate throughout entire simulated trajectory, and this is why sometimes pedestrians keep accerlerating, particularly obvious in the late lifetime of existing pedestrians.
  
##Navigation map
###Version 1
####Introduction
This code can create predefined navigation map that indicates inherent properties of environment, for now including,
1. places that are not allowed to be steped on, simulating, e.g. furniture, buildings
2. places that pedestrian always walk along certain directions(8 different directions, including north, east, south, west, north-east, north-west, south-east, south-west)
Besides, those map is further discretized into patches of size defined by hand. This is done to avoid heavy load of computation and redundant information using per-pixel navigation map. You can use mouse click, drag and entering text to create a navigation map.
####Issue
1. population is not modelled since, from my current knowledge, population and other kinds of statistics are data-driven. However, even if we sample data from a dataset to create those statistics, this may drastically constrain the creation of a navigation map.
2. In indoor scene, there are often some places where people stay for a while, e.g. stove in the kitchen, and this can be modelled without data and defined simply by hand according to deployment of objects in a scene.
