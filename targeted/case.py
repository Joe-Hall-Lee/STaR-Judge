import json


def calculate_percentage(file1_path, file2_path):
    try:
        # Read the first JSON file
        with open(file1_path, 'r') as file1:
            data1 = json.load(file1)

        # Read the second JSON file
        with open(file2_path, 'r') as file2:
            data2 = json.load(file2)

        # Iterate through all top-level keys in the first JSON file
        for key in data1.keys():
            # Check if the key exists in the second file and both values are lists
            if key in data2 and isinstance(data1[key], list) and isinstance(data2[key], list):
                list1 = data1[key]
                list2 = data2[key]

                # Check if the lengths of the lists are the same
                if len(list1) != len(list2):
                    print(f"Error: Key '{key}' has different number of items in the two files.")
                    continue

                # Initialize a counter for matching conditions
                count = 0
                total_items = len(list1)

                # Iterate through the lists and compare the data
                for i in range(total_items):
                    orig1 = list1[i]["result"]["orig"]
                    orig2 = list2[i]["result"]["orig"]

                    # Check the condition
                    if orig1["is_correct"] == True and orig2["is_correct"] == False:
                        count += 1
                    swap1 = list1[i]["result"]["swap"]
                    swap2 = list2[i]["result"]["swap"]

                    # Check the condition
                    if swap1["is_correct"] == True and swap2["is_correct"] == False:
                        count += 1 

                total_items = 2 * total_items
                # Calculate the percentage
                percentage = (count / total_items) * 100 if total_items > 0 else 0

                # Print the results for the current key
                print(f"Key: '{key}' -> {count}/{total_items} items match the condition. Percentage: {percentage:.2f}%")

            else:
                # If the key is missing in the second file or is not a list, print a warning
                print(f"Warning: Key '{key}' is missing in the second file or is not a list.")

    except Exception as e:
        # Print any errors encountered during execution
        print(f"An error occurred: {e}")

instruct_name = "Qwen2.5-3B-Instruct"
finetune_name = "Qwen2.5-3B-Instruct-helpsteer_et_lr_2e-5_epoch_3"

file1_path = f"../POJudge/result/{instruct_name}/llmbar.json"
file2_path = f"../POJudge/result/{finetune_name}/llmbar.json"

calculate_percentage(file1_path, file2_path)

file1_path = f"../POJudge/result/{instruct_name}/mtbench.json"
file2_path = f"../POJudge/result/{finetune_name}/mtbench.json"

calculate_percentage(file1_path, file2_path)

file1_path = f"../POJudge/result/{instruct_name}/judgelm.json"
file2_path = f"../POJudge/result/{finetune_name}/judgelm.json"

calculate_percentage(file1_path, file2_path)

file1_path = f"../POJudge/result/{instruct_name}/hhh.json"
file2_path = f"../POJudge/result/{finetune_name}/hhh.json"

calculate_percentage(file1_path, file2_path)

file1_path = f"../POJudge/result/{instruct_name}/biasbench.json"
file2_path = f"../POJudge/result/{finetune_name}/biasbench.json"

calculate_percentage(file1_path, file2_path)
