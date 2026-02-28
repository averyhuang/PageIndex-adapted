from pageindex.utils import create_node_mapping
import json


json_doc_path = "./results/veterinary_internal_medicine_structure.json"
with open(json_doc_path, "r") as f:
    json_doc = json.load(f)
nodes = create_node_mapping(json_doc)
summary_len = []
print(nodes)
for i in range(len(nodes)):
    summary_len.append(len(nodes[f"{str(i).zfill(4)}"]["summary"]))


import matplotlib.pyplot as plt
plt.bar(range(len(summary_len)), summary_len)
plt.show()