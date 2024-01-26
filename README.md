# Markov Chain applied to baseball

## Game Simulation

It runs 9 innings using one or multiple batters transition matrices. It will run the transitions in the sequence of the list of batters that is passed (no limits applied).
The code runs 9 loops (one per inning) of a while loop, generating random states from 0 --> 25-28.

## Expected Runs


## Batting Line Optimization

If we denote the numbers of batters as n, then all possible combinations for a batting line are n! (or the permutations of n). For a 9 batters, we have a total possible combination of 362,880 for the batting line. This number encompasses all true possibilities with the assumption that position matters, not only the sequence.

With this, we can know the expected run given a sequence of batters, for the first inning.

## Explanation

There are 28 states in baseball innings:

|Runners|None|1st|2nd|3rd|1&2|1&3|2&3|1,2,3|
|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|Outs|||||||||
|0|#1|#2|#3|#4|#5|#6|#7|#8|
|1|#9|#10|#11|#12|#13|#14|#15|#16|
|2|#17|#18|#19|#20|#21|#22|#23|#24|

- State 25: 3 out, 0 runs
- State 26: 3 out, 1 run
- State 27: 3 out, 2 runs
- State 28: 3 out, 3 runs

```python
'''
24 states (state-space S):

Runners:|None|1st|2nd|3rd|1&2|1&3|2&3|1,2,3|
Outs
0|#1|#2|#3|#4|#5|#6|#7|#8|
1|#9|#10|#11|#12|#13|#14|#15|#16|
2|#17|#18|#19|#20|#21|#22|#23|#24|


3outs,0runs=#25
3outs,1run=#26
3outs,2runs=#27
3outs,3runs=#28

pij is the probability of moving from state i to state j. Therefore the
transition matrix (stochastic matrix) of pij is:

__
|p1,1|p1,2|p1,3p1,28|
|p2,1|p2,2|p2,3p2,28|
T=|p3,1,|p3,2|p3,3p3,28|
|()|()|()()|
|p28,1|p28,2|p28,3p28,28|

Tshape=(28,28)

The matrix can also be read as a combination of from-to(pre-transition and
post-transition) situations, where row-wise it has to add to 1 as it represents
the same origin state and all end states.

The matrix above can be represented as a block matrix:
    __
    |A0 B0 C0 D0|
P = |0  A1 B1 E1|
    |0  0  A2 F2|
    |0  0  0  1 |

Where A(8X8) is situation with no out, B(8X8) with one out, C(8X8)from zero to
two, D(8X4), E(8X4) and F(8X4) to end the inning,0 matrices for impossible
scenarios, and 1(4X4). The latter, specifically, is of the form:

  |1000|
1=|1000|
  |1000|
  |1000|

The reason is so all outs end in state 25(absorbing state).


Every transition brings one possible number of runs, so we can have a run matrix
R(28X1) of all expected runs given original state i:

    |R(1)=p1,1|
    |R(2)=p2,2|
R=  |()|
    |R(4)=2*p4,1+p4,4+p4,7+p4,10+p4,2|
    |()|
    |R(28)=0|

Rshape=(28,1)

There as on why R(1)=p1,1 is because from no out and no runner(#1),a batter
can only go back to the same situation to score a run. Anything else is not a
run. Similarly for R(2) and R(3) as in those scenarios there is no one on base

We can then keep track of the runs/state in the inning by using a matrix U of
20-25rows(max of runs in the inning) X 28 columns(current state).
'''
```

Sources
  - [Markov Chain Models: Theoretical Background](http://www.pankin.com/markov/theory.htm)
  - https://wwwjstororg/stable/171922?seq=1#page_scan_tab_contents
  - https://enwikipediaorg/wiki/Stochastic_matrix#Definition_and_properties
  - http://statshackercom/the-markov-chain-model-of-baseball#prettyPhoto
  - https://enwikipediaorg/wiki/State_space
  - https://enwikipediaorg/wiki/Block_matrix
