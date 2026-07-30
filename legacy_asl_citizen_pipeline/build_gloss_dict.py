import csv

gloss_list = []
with open('data_csv/train.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader, None)  # skip header row
    for row in reader:
        g = row[2].strip()
        if g not in gloss_list:
            gloss_list.append(g)

gloss_list.sort()

gloss_dict = {}
for i, g in enumerate(gloss_list):
    gloss_dict[g] = i

print(f"Total unique glosses: {len(gloss_dict)}")
print("First 5:", list(gloss_dict.items())[:5])
print("Last 5:", list(gloss_dict.items())[-5:])

# Save it so we don't have to rebuild this every time
import json
with open('gloss_dict.json', 'w') as f:
    json.dump(gloss_dict, f)
print("Saved to gloss_dict.json")
