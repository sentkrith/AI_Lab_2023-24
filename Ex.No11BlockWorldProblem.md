# Ex.No: 11  Planning –  Block World Problem 
### DATE: 26/09/2025                                                                           
### REGISTER NUMBER :212223060254 
### AIM: 
To find the sequence of plan for Block word problem using PDDL  
###  Algorithm:
Step 1 :  Start the program <br>
Step 2 : Create a domain for Block world Problem <br>
Step 3 :  Create a domain by specifying predicates clear, on table, on, arm-empty, holding. <br>
Step 4 : Specify the actions pickup, putdown, stack and un-stack in Block world problem <br>
Step 5 :  In pickup action, Robot arm pick the block on table. Precondition is Block is on table and no other block on specified block and arm-hand empty.<br>
Step 6:  In putdown action, Robot arm place the block on table. Precondition is robot-arm holding the block.<br>
Step 7 : In un-stack action, Robot arm pick the block on some block. Precondition is Block is on another block and no other block on specified block and arm-hand empty.<br>
Step 8 : In stack action, Robot arm place the block on under block. Precondition is Block holded by robot arm and no other block on under block.<br>
Step 9 : Define a problem for block world problem.<br> 
Step 10 : Obtain the plan for given problem.<br> 
     
### Program:
```
(define (domain blocksworld)
(:requirements :strips :equality)
(:predicates (clear ?x)
(on-table ?x)
(arm-empty)
(holding ?x)
(on ?x ?y))
(:action pickup
:parameters (?ob)
:precondition (and (clear ?ob) (on-table ?ob) (arm-empty))
:effect (and (holding ?ob) (not (clear ?ob)) (not (on-table ?ob))
(not (arm-empty))))
(:action putdown
:parameters (?ob)
:precondition (and (holding ?ob))
:effect (and (clear ?ob) (arm-empty) (on-table ?ob)
(not (holding ?ob))))
(:action stack
:parameters (?ob ?underob)
:precondition (and (clear ?underob) (holding ?ob))
:effect (and (arm-empty) (clear ?ob) (on ?ob ?underob)
(not (clear ?underob)) (not (holding ?ob))))
(:action unstack
:parameters (?ob ?underob)
:precondition (and (on ?ob ?underob) (clear ?ob) (arm-empty))
:effect (and (holding ?ob) (clear ?underob)
(not (on ?ob ?underob)) (not (clear ?ob)) (not (arm-empty)))))
```

### Input 
**Problem 1**

```
(define (problem pb1)
(:domain blocksworld)
(:objects a b)
(:init (on-table a) (on-table b) (clear a) (clear b) (arm-empty))
(:goal (and (on a b))))
```
**Problem 2**

```
(define(problem pb3)
(:domain blocksworld)
(:objects a b c)
(:init (on-table a) (on-table b) (on-table c)
(clear a) (clear b) (clear c) (arm-empty))
(:goal (and (on a b) (on b c))))
```

### Output/Plan:
**Problem 1**

<img width="1904" height="1079" alt="image" src="https://github.com/user-attachments/assets/553519f2-0114-48d2-b0ba-b651b0891c01" />
<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/9d71adbe-286f-430d-b0da-626681bb5e8c" />

**Problem 2**

<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/feab0a95-2ecd-421a-8dcc-a220a1a36fde" />
<img width="1919" height="1077" alt="image" src="https://github.com/user-attachments/assets/5ffa418b-8bf2-476b-a635-5daf6649ef0d" />
<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/f0503de8-4b70-4118-8f25-0ea7757f7ec5" />
<img width="1913" height="1079" alt="image" src="https://github.com/user-attachments/assets/402804da-fd99-4294-aac7-b922b02d8195" />

### Result:
Thus the plan was found for the initial and goal state of block world problem.
