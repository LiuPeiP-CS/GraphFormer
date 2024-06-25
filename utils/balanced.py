import random
import json

def read_file(filename):
    print(filename)
    with open(filename,encoding="utf-8") as infile:
        data = json.load(infile)
    return data


# 从包含不同关系的数据中分别随机选择相同数量并确保不重复
def balanced(num_samples,data):
    numbers = [random.randrange(num_samples-100, num_samples) for _ in range(5)]

    # 打印生成的随机数
    print(numbers)
    none_data = [item for item in data if item.get("relation") == "None"]
    resistance_non_response_data = [item for item in data if item.get("relation") == "resistance or non-response"]
    sensitivity_data = [item for item in data if item.get("relation") == "sensitivity"]
    response_data = [item for item in data if item.get("relation") == "response"]
    resistance_data = [item for item in data if item.get("relation") == "resistance"]

    selected_none_data = random.sample(none_data, numbers[0])
    selected_resistance_non_response_data = random.sample(resistance_non_response_data, numbers[1])
    selected_sensitivity_data = random.sample(sensitivity_data, numbers[2])
    selected_response_data = random.sample(response_data, numbers[3])
    selected_resistance_data = random.sample(resistance_data, numbers[4])

    # 合并所选数据
    selected_data = (
        selected_none_data +
        selected_resistance_non_response_data +
        selected_sensitivity_data +
        selected_response_data +
        selected_resistance_data
    )
    return selected_data

def balanced_train_file():
    random.seed()
    data = read_file('dataset/bin_mul/5/train.json')
    selected_data = balanced(1000, data)
    # 将所选数据保存到 JSON 文件中
    random.shuffle(selected_data)
    output_file_path = "dataset/semeval/train.json"
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        json.dump(selected_data, output_file, ensure_ascii=False, indent=2)

    print(f"Selected data saved to {output_file_path}")


def balanced_test_file():
    random.seed()
    data = read_file("dataset/bin_mul/5/test.json")
    selected_data = balanced(250,data)
    random.shuffle(selected_data)
    # 将所选数据保存到 JSON 文件中
    output_file_path = "dataset/pubmed/test.json"
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        json.dump(selected_data, output_file, ensure_ascii=False, indent=2)

    print(f"Selected data saved to {output_file_path}")

    # 打印所选数据
    # i=0
    # for data_item in selected_data:
    #     i += 1
    #     print(i)
    #     print(data_item)
