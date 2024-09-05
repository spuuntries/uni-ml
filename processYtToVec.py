from sentence_transformers import SentenceTransformer
import json

model = SentenceTransformer("BAAI/bge-base-en")
data = json.loads(
    open(
        "../uni-damin/tugas1_youtube/ytResults_10_14b1abe2-a848-4d7d-b6c5-8544a64d571d.json"
    ).read()
)

for e in data.values():
    e["embedded"] = model.encode(e["desc"]).tolist()

print(list(data.values())[0])

with open("res.json", "w") as f:
    f.write(json.dumps(data))
