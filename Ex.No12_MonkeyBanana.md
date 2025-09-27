# Ex.No: 11  Planning –  Monkey Banana Problem
### DATE: 27/09/2025                                                                            
### REGISTER NUMBER :212223060254 
### AIM: 
To find the sequence of plan for Monkey Banana problem using PDDL Editor.
###  Algorithm:
Step 1:  Start the program <br> 
Step 2 : Create a domain for Monkey Banana Problem. <br> 
Step 3:  Create a domain by specifying predicates. <br> 
Step 4: Specify the actions GOTO, CLIMB, PUSH-BOX, GET-KNIFE, GRAB-BANANAS in Monkey Banana problem.<br>  
Step 5:   Define a problem for Monkey Banana problem.<br> 
Step 6:  Obtain the plan for given problem.<br> 
Step 7: Stop the program.<br> 
### Program:
```
(define (domain monkey)
 (:requirements :strips)
 (:constants monkey box knife bananas glass waterfountain)
 (:predicates (location ?x)
 (on-floor)
 (at ?m ?x)
 (hasknife)
 (onbox ?x)
 (hasbananas)
 (hasglass)
 (haswater))
 ;; movement and climbing
 (:action GO-TO
 :parameters (?x ?y)
 :precondition (and (location ?x) (location ?y) (on-floor) (at monkey ?y))
 :effect (and (at monkey ?x) (not (at monkey ?y))))
 (:action CLIMB
 :parameters (?x)
 :precondition (and (location ?x) (at box ?x) (at monkey ?x))
 :effect (and (onbox ?x) (not (on-floor))))
 (:action PUSH-BOX
 :parameters (?x ?y)
 :precondition (and (location ?x) (location ?y) (at box ?y) (at monkey ?y)
 (on-floor))
 :effect (and (at monkey ?x) (not (at monkey ?y))
 (at box ?x) (not (at box ?y))))
 ;; getting bananas
 (:action GET-KNIFE
 :parameters (?y)
 :precondition (and (location ?y) (at knife ?y) (at monkey ?y))
 :effect (and (hasknife) (not (at knife ?y))))
 (:action GRAB-BANANAS
 :parameters (?y)
 :precondition (and (location ?y) (hasknife)
 (at bananas ?y) (onbox ?y))
:effect (hasbananas))
 ;; getting water
 (:action PICKGLASS
 :parameters (?y)
 :precondition (and (location ?y) (at glass ?y) (at monkey ?y))
 :effect (and (hasglass) (not (at glass ?y))))
 (:action GETWATER
 :parameters (?y)
 :precondition (and (location ?y) (hasglass)
 (at waterfountain ?y)
 (at monkey ?y)
 (onbox ?y))
 :effect (haswater)))
```

### Input 
```
(define (problem pb1)
 (:domain monkey)
 (:objects p1 p2 p3 p4 bananas monkey box knife)
 (:init (location p1)
 (location p2)
 (location p3)
 (location p4)
 (at monkey p1)
 (on-floor)
 (at box p2)
 (at bananas p3)
 (at knife p4)
 )
 (:goal (and (hasbananas)))
 )
```
### Output/Plan:
<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/b027093f-dffe-4a10-87c0-03b1a394f815" />

<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/8677ee0f-02f4-4263-b11c-eb79b795c6fe" />

<img width="1919" height="1076" alt="image" src="https://github.com/user-attachments/assets/bd579b7a-696e-4f73-ae05-5ea3d590f0ea" />

<img width="1919" height="1067" alt="image" src="https://github.com/user-attachments/assets/86c2a5ac-2e4b-4678-ba71-24117c4fa7a0" />

<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/1004b362-263c-4f42-a3e3-288af393cb3f" />

<img width="1919" height="1078" alt="image" src="https://github.com/user-attachments/assets/b054e204-d153-44a7-85b7-2a61f049c81d" />


### Result:
Thus the plan was found for the initial and goal state of given problem.
