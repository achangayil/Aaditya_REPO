# -*- coding: utf-8 -*-
"""Python_game.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16ADKWgkxuss-H_gNGb-bYEuwUVFG3np2
"""

import numpy as np


class Like_battleship:

  # this is the init constructor for the class Like_battleship
  def __init__(self, rw, clmn): 
    # assigning the variables to self variables
    self.row = rw
    self.column = clmn
    # numpy.random.randn returns random samples from a specific distribution
    # rw and clmn are passed as parameters
    self.arrayboard = np.random.randn(rw,clmn)
    # 
    self.boardl = [['_'] * clmn for i in range(rw)]

  # __str__ method easy to read and outputs all the members of the class
  def __str__(self): 
  # This method called when print() or string() ae invoked on an object  
  # This method returns only the string object
    return str(self.arrayboard)

  def play(self):
# the input is typecasted to integer. When the user inputs a number, 
#it is placed one number less than it, because on rows and columns,
# it starts at 0, so we need to subtract the input by 1 to match with that.
    r = int(input("Which row?: \n")) - 1
    c = int(input("Which column?: \n")) - 1
    # if the value generated by the selected choice in the 
    # randomly selected numbers is less than 0, a bomb has been found.
    if self.arrayboard[r][c] < 0:
      print("We found a BOMB!!!")
      print(self)
    else: 
    # if the value is not less than zero , a bomb hasn't been found.
    # It is asked if you would like to continue. If you answer Y or y,
    # you'll continue playing. 
      go_on = input("""There is no bomb here. 
      \nWould you like to continue? Y/N """)
      # .lower() converts string letters to lowercase
      if go_on.lower() == 'y':
        self.play()
     # If you enter anything else, the game will be done.
      else:
        print("Ok, you're done. See ya'!")
        print(self)

# this is the main method, where the main operations are controlled from
def main():
  # the integer variables are typecasted to integer forms
  rw = int(input("How many rows?\n"))
  clmn = int(input("How many columns?\n"))
  # an object is created for the Like_battleship class. rw and clms are passed as parameters. 
  game = Like_battleship(rw, clmn)
  # The play() method is called with game
  game.play()

# __name__ is a special variable that defines the name of the class from where it's called.
# __main__ represents __name__. __name__ will look for the main code to execute. 
if __name__ == "__main__":
  main() # if __name__ == "__main__", the main method will ony directly execute.