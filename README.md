# KaggleTitanic
https://www.kaggle.com/c/titanic/

Data Dictionary
* Variable	Definition	Key
  * pclass	Ticket class (A proxy for socio-economic status (SES))
    * 1 = 1st
    * 2 = 2nd
    * 3 = 3rd
  * survival
    * 0 = No
    * 1 = Yes
  * sex	Sex	
  * Age	Age in years
    * Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
  * sibsp	# of siblings / spouses aboard the Titanic
    * The dataset defines family relations in this way...
      * Sibling = brother, sister, stepbrother, stepsister
      * Spouse = husband, wife (mistresses and fiancés were ignored)
  * parch	# of parents / children aboard the Titanic
    * Parent = mother, father
    * Child = daughter, son, stepdaughter, stepson
    * Some children travelled only with a nanny, therefore parch=0 for them.	
  * ticket	Ticket number	
  * fare	Passenger fare	
  * cabin	Cabin number	
  * embarked
  * Port of Embarkation
    * C = Cherbourg
    * Q = Queenstown
    * S = Southampton