import json
import sys

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def compare_json(json1, json2):
    dict1 = {item["layer"]: item for item in json1}
    dict2 = {item["layer"]: item for item in json2}

    diff_found = False
    for layer in dict1:
        if layer in dict2:
            in1, out1 = dict1[layer]["input_norm"], dict1[layer]["output_norm"]
            in2, out2 = dict2[layer]["input_norm"], dict2[layer]["output_norm"]

            if in1 != in2 or out1 != out2:
                diff_found = True
                print(f"Layer: {layer}")
                print(f"  input_norm: {in1} vs {in2}")
                print(f"  output_norm: {out1} vs {out2}")
                print("-" * 50)
                exit(0)

    if not diff_found:
        print("✅ 两个文件的 input_norm/output_norm 完全一致")

if __name__ == "__main__":
    # if len(sys.argv) != 3:
    #     print(f"用法: python {sys.argv[0]} file1.json file2.json")
    #     sys.exit(1)

    file1= "14-diffusion-rollout-0-result-compile-true-grad-false-rank-0.json"
    file2= "14-diffusion-rollout-0-result-compile-true-grad-true-rank-0.json"
    json1 = load_json(file1)
    json2 = load_json(file2)
    compare_json(json1, json2)
