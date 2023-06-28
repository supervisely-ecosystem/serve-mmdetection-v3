from collections import defaultdict
from pathlib import Path
import yaml
import json


def parse_yaml_file(yaml_file):
    with open(yaml_file, "r") as file:
        yaml_content = yaml.safe_load(file)

    collections = {}
    yaml_models = []
    if isinstance(yaml_content, dict):
        if yaml_content.get("Collections"):
            if isinstance(yaml_content["Collections"], list):
                for c in yaml_content["Collections"]:
                    collections[c["Name"]] = c
            else:
                raise
        else:
            print(f"Has not collections: {yaml_file}.")
        if yaml_content.get("Models"):
            yaml_models = yaml_content["Models"]
    elif isinstance(yaml_content, list):
        yaml_models = yaml_content
        print(f"Only list: {yaml_file}.")
    else:
        raise

    models = defaultdict(list)
    for m in yaml_models:
        if not m.get("Weights"):
            print(f"skip {m['Name']} in {yaml_file}.")
            continue
        collection = m["In Collection"]
        m = {
            "tasks": [r["Task"] for r in m["Results"]],
            "config": m["Config"],
            "weights": m["Weights"],
            "metrics": [r["Metrics"] for r in m["Results"]],
        }
        models[collection].append(m)

    return collections, models


def json_dump(obj, file):
    with open(file, "w") as f:
        json.dump(obj, f, indent=2)


def json_load(file):
    with open(file, "r") as f:
        return json.load(f)


metafiles = list(map(str, Path("configs").rglob("*.yml")))
# meta = json_load("models/detection_meta.json")
# meta += json_load("models/instance_segmentation_meta.json")
# meta = {x["model_name"]: dict(paper_from=x["paper_from"], year=x["year"]) for x in meta}


collections = {}
models = defaultdict(list)
for metafile in metafiles:
    new_collections, new_models = parse_yaml_file(metafile)
    collections.update(new_collections)
    for c, m_list in new_models.items():
        assert isinstance(m_list, list)
        models[c] += m_list

json_dump(collections, "models/collections.json")
json_dump(models, "models/models.json")
