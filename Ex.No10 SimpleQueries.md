# Ex.No: 10  Logic Programming –  Simple queries from facts and rules
### DATE: 27/09/2025                                                                           
### REGISTER NUMBER : 212223060254
### AIM: 
To write a prolog program to find the answer of query. 
###  Algorithm:
 Step 1: Start the program <br> 
 Step 2: Convert the sentence into First order Logic  <br> 
 Step 3:  Convert the sentence into Horn clause form  <br> 
 Step 4: Add rules and predicates in a program   <br> 
 Step 5:  Pass the query to program. <br> 
 Step 6: Prolog interpreter shows the output and return answer. <br> 
 Step 8:  Stop the program.
### Program:
### Task 1:
Construct the FOL representation for the following sentences <br> 
1.	John likes all kinds of food.  <br> 
2.	Apples are food.  <br> 
3.	Chicken is a food.  <br> 
4.	Sue eats everything Bill eats. <br> 
5.	 Bill eats peanuts  <br> 
   Convert into clause form and Prove that John like Apple by using Prolog. <br> 
### Program:
```
likes(john,X):-
food(X).
eats(bill,X):-
eats(sue,X).
eats(Y,X):-
food(X).
eats(bill,peanuts).
food(apple).
food(chicken).
food(peanuts).
```

### Output:
<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/89719f07-f0e4-4aa8-9734-f64ceb2a5e67" />


### Task 2:
Consider the following facts and represent them in predicate form: <br>              
1.	Steve likes easy courses. <br> 
2.	Science courses are hard. <br> 
3. All the courses in Have fun department are easy <br> 
4. BK301 is Have fun department course.<br> 
Convert the facts in predicate form to clauses and then prove by resolution: “Steve likes BK301 course”<br> 

### Program:
```
like(steve,X):-easycourse(X).
hard(sciencecourse).
easycourse(X):-course(X,fundept).
course(bk301,fundept).
```

### Output:
<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/0293ab35-ffd1-460b-9198-4c64878cf1cf" />

### Task 3:
Consider the statement <br> 
“This is a crime for an American to sell weapons to hostile nations. The Nano , enemy of America has some missiles and its missiles were sold it by Colonal West who is an American” <br> 
Convert to Clause form and prove west is criminal by using Prolog.<br> 
### Program:
```
crime(X):-american(X),weapon(Y),sells(X,Y,Z),hostile(X,Z).
hostile(X,Z):-enemy(Z,X).
enemy(nano,west).
american(west).
weapon(Y):-missile(Y).
missile(m1).
owns(nano,m1).
sells(west,Y,nano):-owns(nano,Y),missile(Y).
```

### Output:
<img width="1919" height="1060" alt="image" src="https://github.com/user-attachments/assets/9fd7158c-8bba-4302-9cca-f50089446ef9" />

### Result:
Thus the prolog programs were executed successfully and the answer of query was found.
