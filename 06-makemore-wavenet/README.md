# Makemore

## Lesson #6: Building a WaveNet

* Now, we will finally enhance the architecture of our neural net beyond our simple 2-layer MLP that only considers the previous 3 characters.
* By the end of the lesson we will have something along the lines of Google's WaveNet architecture (there's a 2016 paper on it), which, from the looks of it, increases how much information the neural net can ingest by adopting a tree-like structure to gradually squash input information at each layer until we arrive at the exact same output: a prediction probability distribution

### Starter Code

* From here on most work will be done in Jupyter notebooks to better follow Andrej Karpathy's skeleton code
* And also because it's convenient :|
