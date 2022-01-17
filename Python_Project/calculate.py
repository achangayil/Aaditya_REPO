class calculate_add_and_sub:

  def __init__(self, num1, num2):
    self.n1 = num1
    self.n2 = num2

  def add_sub(self):
    sum = self.n1 + self.n2
    sub = self.n1 - self.n2
    return sum, sub

  def show(self):
    show_sum = self.add_sub()[0]
    print("show sum", show_sum)
    show_sub = self.add_sub()[1]
    print("show sub", show_sub)

def main():
  x = 8
  y = 5
  obj = calculate_add_and_sub(x,y)
  obj.show()

if __name__ == '__main__':
  main()
