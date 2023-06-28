from pathlib import Path
import yaml
import json


def parse_yaml_file(yaml_file):
    with open(yaml_file, "r") as file:
        yaml_content = yaml.safe_load(file)

    try:
        collection_name = yaml_content["Collections"][0]["Name"]
    except:
        try:
            collection_name = yaml_content["Models"][0]["In Collection"]
        except:
            collection_name = yaml_content[0]["In Collection"]

    if isinstance(yaml_content, list):
        models = yaml_content
    elif yaml_content.get("Models"):
        models = yaml_content["Models"]
    else:
        print(f"skip: {yaml_file}")
        return collection_name, [], []

    tasks_list = []
    weights = []
    for model in models:
        tasks = tuple([result["Task"] for result in model["Results"]])
        tasks_list.append(tasks)
        if model.get("Weights"):
            weights.append(model["Weights"])

    return collection_name, tasks_list, weights


def json_dump(obj, file):
    with open(file, "w") as f:
        json.dump(obj, f, indent=2)


def json_load(file):
    with open(file, "r") as f:
        return json.load(f)


metafiles = list(map(str, Path("configs").rglob("*.yml")))
meta = json_load("models/detection_meta.json")
meta += json_load("models/instance_segmentation_meta.json")
meta = {x["model_name"]: dict(paper_from=x["paper_from"], year=x["year"]) for x in meta}


det_models = []
segm_models = []
for metafile in metafiles:
    collection_name, tasks_list, weights = parse_yaml_file(metafile)

    if len(weights) == 0:
        continue
    if len(tasks_list) != len(weights):
        print(f"warning (not all weights): {metafile}")

    d = {
        "model_name": collection_name,
        "yml_file": metafile[len("configs/") :],
        "paper_from": "",
        "year": "",
        "mono_task": True,
        "n_models": len(tasks_list),
        "n_weights": len(weights),
    }
    if meta.get(collection_name):
        d.update(meta[collection_name])
    if set(tasks_list) == {("Object Detection", "Instance Segmentation")}:
        segm_models.append(d)
        det_models.append(d)
    elif set(tasks_list) == {("Object Detection",)}:
        det_models.append(d)
    else:
        d["mono_task"] = False
        det_models.append(d)
        tasks = set([x for t in tasks_list for x in t])
        if "Instance Segmentation" in tasks:
            segm_models.append(d)

json_dump(det_models, "models/det_models.json")
json_dump(segm_models, "models/segm_models.json")
