import enum
import os
import json

# =============================== MM-Safety-Bench ===============================
import random

categories = [
    "01-Illegal_Activity",
    "02-HateSpeech",
    "03-Malware_Generation",
    "04-Physical_Harm",
    "05-EconomicHarm",
    "06-Fraud",
    "07-Sex",
    "08-Political_Lobbying",
    "09-Privacy_Violence",
    "10-Legal_Opinion",
    "11-Financial_Advice",
    "12-Health_Consultation",
    "13-Gov_Decision",
]


class MMImageTypes(enum.StrEnum):
    IMAGE = "SD"
    IMAGE_WITH_TEXT = "SD_TYPO"
    TEXT = "TYPO"


def load_mm_safetybench(
    dataset_path: str,
    first_n: int = -1,
    image_type: MMImageTypes = MMImageTypes.IMAGE_WITH_TEXT,
) -> list[dict[str, str]]:
    image_base_dir = os.path.join(dataset_path, "imgs")
    prompt_base_dir = os.path.join(dataset_path, "processed_questions")
    dataset = []

    for category in categories:
        with open(os.path.join(prompt_base_dir, category + ".json"), "r", encoding="utf-8") as my_file:
            prompts = json.load(my_file)
            cases = [
                {
                    "image_path": os.path.normpath(
                        os.path.join(image_base_dir, category, image_type.value, f"{k}.jpg")
                    ),
                    "question": v["Rephrased Question"],
                    "unsafe_category": category,
                }
                for k, v in prompts.items()
            ]

            dataset += cases[0:first_n]

    return dataset


def load_mm_safetybench_sample(
    dataset_path: str,
    image_type: MMImageTypes = MMImageTypes.IMAGE_WITH_TEXT,
    sample_size: int | None = None,
    seed: int | None = None,
) -> list[dict[str, str]]:
    if seed is not None:
        random.seed(seed)

    image_base_dir = os.path.join(dataset_path, "imgs")
    prompt_base_dir = os.path.join(dataset_path, "processed_questions")

    category_data = {}
    for category in categories:
        with open(os.path.join(prompt_base_dir, category + ".json"), "r", encoding="utf-8") as my_file:
            prompts = json.load(my_file)
            cases = [
                {
                    "image_path": os.path.normpath(
                        os.path.join(image_base_dir, category, image_type.value, f"{k}.jpg")
                    ),
                    "question": v["Rephrased Question"],
                    "unsafe_category": category,
                }
                for k, v in prompts.items()
            ]
            category_data[category] = cases

    total_count = sum(len(cases) for cases in category_data.values())
    dataset = []

    if sample_size is None or sample_size >= total_count:
        for category in categories:
            dataset += category_data[category]
    else:
        for category in categories:
            cases = category_data[category]
            if len(cases) == 0:
                continue

            proportion = len(cases) / total_count
            n_samples = max(1, round(sample_size * proportion))
            n_samples = min(n_samples, len(cases))

            if n_samples == len(cases):
                sampled = cases
            else:
                sampled = random.sample(cases, n_samples)

            dataset += sampled

    return dataset


# =============================== SPA-VL-harm ===============================
def load_spa_data(
    dataset_path: str,  # ./spa-vl-harm/
    first_n: int = -1,
) -> list[dict[str, str]]:
    image_base_dir = os.path.join(dataset_path, "images")
    spa_harm_file = os.path.join(dataset_path, "spa_vl_harm.json")
    dataset = []

    with open(spa_harm_file, "r", encoding="utf-8") as my_file:
        spa_harm_data = json.load(my_file)
        cases = [
            {
                "image_path": os.path.normpath(os.path.join(image_base_dir, f"{k}.png")),
                "question": v["question"],
            }
            for k, v in spa_harm_data.items()
        ]

        if first_n == -1:
            dataset = cases
        else:
            dataset += cases[0:first_n]

    return dataset


# =============================== SIUO ===============================
def load_siuo_data(
    dataset_path: str,  # ./SIUO/
    first_n: int = -1,
) -> list[dict[str, str]]:
    image_base_dir = os.path.join(dataset_path, "images")
    siuo_file = os.path.join(dataset_path, "siuo_gen.json")
    dataset = []

    with open(siuo_file, "r", encoding="utf-8") as my_file:
        siuo_data = json.load(my_file)

        cases = []
        for item in siuo_data:
            cases.append(
                {
                    "image_path": os.path.normpath(os.path.join(image_base_dir, item["image"])),
                    "question": item["question"],
                }
            )

        if first_n == -1:
            dataset = cases
        else:
            dataset += cases[0:first_n]

    return dataset


# =============================== MSSBench ===============================
def load_mssbench_data(
    dataset_path: str,  # ./MSSBench/
    first_n: int = -1,
) -> list[dict[str, str]]:
    image_base_dir = os.path.join(dataset_path, "chat")
    mssbench_file = os.path.join(dataset_path, "combined.json")
    dataset = []

    with open(mssbench_file, "r", encoding="utf-8") as my_file:
        mssbench_data = json.load(my_file)
        cases = []

        for item in mssbench_data["chat"]:
            cases.append(
                {
                    "image_path": os.path.normpath(os.path.join(image_base_dir, item["unsafe_image_path"])),
                    "question": item["queries"][0],
                    "hint": f"intent: {item['intent']}, unsafe reason: {item['unsafe_image']}",
                    "type": item["Type"],
                }
            )

        if first_n == -1:
            dataset = cases
        else:
            dataset += cases[0:first_n]

    return dataset


# =============================== FigStep ===============================
def load_figStep(
    dataset_path: str,
    first_n: int = -1,
) -> list[dict[str, str]]:
    figStep_file = os.path.join(dataset_path, "safebench.json")
    dataset = []

    with open(os.path.join(figStep_file), "r", encoding="utf-8") as my_file:
        figstep_data = json.load(my_file)
        cases = [
            {
                "image_path": os.path.normpath(os.path.join(dataset_path, v["image_path"])),
            }
            for k, v in figstep_data.items()
        ]

        if first_n == -1:
            dataset = cases
        else:
            dataset += cases[0:first_n]

    return dataset
