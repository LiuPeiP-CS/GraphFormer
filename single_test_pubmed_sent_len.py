import shutil
import matplotlib.pyplot as plt
import eval

x = range(1, 6)
y = []

test_scores=[]

for i in range(5):
    # 原始文件夹路径
        source_file = f"dataset/bin_mul/5/test{i+1}.json"
        # 目标文件夹路径
        print("!!!!!!!!!!!!!!!!!!!!!!!")
        print(source_file)
        target_file = "dataset/semeval/test.json"

        # 将源文件复制到目标文件
        shutil.copy2(source_file, target_file)

        is_balanced= False
        y.append(eval.eval(is_balanced))

plt.plot(x, y,linewidth=2, marker='o',label='cross')
plt.xticks(range(1, 6))
print(y)
plt.xlabel('frequency')
plt.ylabel('cross')
plt.title('single_test')
plt.legend()
plt.show()

