import random
import json

def read_file(filename):
    print(filename)
    with open(filename,encoding="utf-8") as infile:
        data = json.load(infile)
    return data


# 从包含不同关系的数据中分别随机选择相同数量并确保不重复
def balanced(num_samples,data):
    if num_samples > 100:
        numbers = [random.randrange(num_samples-100, num_samples) for _ in range(11)]
    else:
        numbers = [random.randrange(num_samples, num_samples+1) for _ in range(11)]
        # 打印生成的随机数
    print(numbers)

    aka_data = [item for item in data if item.get("relation") == "aka"]
    attack_data = [item for item in data if item.get("relation") == "attack"]
    belong_to_data = [item for item in data if item.get("relation") == "belong_to"]
    end_time_data = [item for item in data if item.get("relation") == "end_time"]
    find_data = [item for item in data if item.get("relation") == "find"]
    goal_data = [item for item in data if item.get("relation") == "goal"]
    launch_data = [item for item in data if item.get("relation") == "launch"]
    located_data = [item for item in data if item.get("relation") == "located"]
    occur_time_data = [item for item in data if item.get("relation") == "occur_time"]
    Release_time_data = [item for item in data if item.get("relation") == "Release_time"]
    use_data = [item for item in data if item.get("relation") == "use"]


    selected_aka_data = random.sample(aka_data, min(numbers[0],len(aka_data)))
    selected_attack_data = random.sample(attack_data, min(numbers[1],len(attack_data)))
    selected_belong_to_data = random.sample(belong_to_data, min(numbers[2],len(belong_to_data)))
    selected_end_time_data = random.sample(end_time_data, min(numbers[3],len(end_time_data)))
    selected_find_data = random.sample(find_data, min(numbers[4],len(find_data)))
    selected_goal_data = random.sample(goal_data, min(numbers[5],len(goal_data)))
    selected_launch_data = random.sample(launch_data, min(numbers[6],len(launch_data)))
    selected_located_data = random.sample(located_data, min(numbers[7],len(located_data)))
    selected_occur_time_data = random.sample(occur_time_data,min(numbers[8],len(occur_time_data)))
    selected_Release_time_data = random.sample(Release_time_data, min(numbers[9],len(Release_time_data)))
    selected_use_data = random.sample(use_data, min(numbers[10],len(use_data)))

    # 合并所选数据
    selected_data = (
        selected_aka_data +
        selected_attack_data +
        selected_belong_to_data +
        selected_end_time_data +
        selected_find_data +
        selected_goal_data +
        selected_launch_data +
        selected_located_data +
        selected_occur_time_data +
        selected_Release_time_data +
        selected_use_data
    )
    return selected_data

def balanced_train_file():
    random.seed()
    data = read_file('../dataset/bin_mul/6/train.json')
    selected_data = balanced(600,data)
    # 将所选数据保存到 JSON 文件中
    random.shuffle(selected_data)
    output_file_path = "../dataset/semeval/train.json"
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        json.dump(selected_data, output_file, ensure_ascii=False, indent=2)

    print(f"Selected data saved to {output_file_path}")


def balanced_test_file():
    random.seed()
    data = read_file("../dataset/semeval/train.json")
    selected_data = balanced(150,data)
    # 将所选数据保存到 JSON 文件中
    output_file_path = "../dataset/semeval/test.json"
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        json.dump(selected_data, output_file, ensure_ascii=False)

    print(f"Selected data saved to {output_file_path}")


if __name__ == '__main__':
    balanced_train_file()
    # balanced_test_file()