import matplotlib.pyplot as plt
import eval
x = range(1, 11)
y = []

test_scores=[]

for _ in range(10):
    # exec(open('eval.py').read())
    is_balanced = True
    y.append(eval.eval(is_balanced))

plt.plot(x, y,linewidth=2, marker='o',label='cross')
plt.xticks(range(1, 11))
print(y)
plt.xlabel('frequency')
plt.ylabel('cross')
plt.title('mul_test')
plt.legend()
plt.show()