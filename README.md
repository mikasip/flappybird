# Flappy Bird with genetic algorithm

The Flappy Bird with genetic algorithm is built in top of [Flappy bird clone](https://github.com/sourabhv/FlapPyBird). 
The birds decide continuously weather to jump or not based on a neural network with two input variables, horizontal and vertical distances to bottom of the up coming upper pipe.
The neural network has one hidden layer with four units, and the output gives value between zero and one. If the output is greater than 0.5, the bird jumps, otherwise it does not.
The parameters of the neural network are fitted by using genetic algorithm, where new population is generated by mating the two best performing birds of the previous generation.
Random mutations are added to birds, by probability p which can be selected by the user. Also the population size can be modified by the user.


## How to run?

1. Make sure you have python downloaded.
2. Install dependencies:
```pip install numpy sympy pygame```
3. Clone the repository:
```git clone https://github.com/mikasip/flappybird```
4. Navigate to cloned repo by command line.
5. Run ```python flappybird.py```
