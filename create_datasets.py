import numpy as np
import os
import json

def read_files(file_name: str, max_items: int = 10) -> list:
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()[:max_items]

    pairs = [line[line.find('.')+2:] for line in lines]  # remove numbering
    tuples_list = [(p.split('/')[0].strip(), p.split('/')[1].strip()) for p in pairs]

    return tuples_list


def create_sva_dataset(
    subject_file: str,
    object_file: str,
    rel_verb_file: str,
    main_verb_pair: tuple,
    relativizers: tuple,
    language: str,
    max_items: int = 10
):
    """Generates a subject-verb agreement dataset with matched/mismatched objects."""
    subjects = read_files(subject_file, max_items)
    objects = read_files(object_file, max_items)
    rel_verbs = read_files(rel_verb_file, max_items)
    main_v_s, main_v_p = main_verb_pair
    rel_sg, rel_pl = relativizers

    items = []
    item_id_counter = 0

    for subj_s, subj_p in subjects:
        for obj_s, obj_p in objects:
            for rel_v_s, rel_v_p in rel_verbs:

                # two contexts per (subject, object, verb) combo
                contexts = [
                    ("matched", 
                     (subj_s, obj_s, rel_v_s, rel_sg, "singular"),
                     (subj_p, obj_p, rel_v_p, rel_pl, "plural")),
                    ("mismatched",
                     (subj_s, obj_p, rel_v_s, rel_sg, "singular"),
                     (subj_p, obj_s, rel_v_p, rel_pl, "plural"))
                ]

                for ctx_name, (s_s, o_s, rv_s, r_s, _), (s_p, o_p, rv_p, r_p, _) in contexts:

                    # singular subject item
                    if language == "italian":
                        prefix_sg = f"{s_s} {r_s} {rv_s} {o_s}"
                    else:
                        prefix_sg = f"{s_s} {r_s} {rv_s} {o_s}"

                    current_id = item_id_counter

                    items.append({
                        "item_id": current_id,
                        "language": language,
                        "context_type": ctx_name,
                        "subject_number": "singular",
                        "object_number": "plural" if ctx_name == "mismatched" else "singular",
                        "prefix": prefix_sg,
                        "correct_verb": main_v_s,
                        "incorrect_verb": main_v_p,
                    })

                    # plural subject item
                    if language == "italian":
                        prefix_pl = f"{s_p} {r_p} {rv_p} {o_p}"
                    else:
                        prefix_pl = f"{s_p} {r_p} {rv_p} {o_p}"

                    items.append({
                        "item_id": current_id,
                        "language": language,
                        "context_type": ctx_name,
                        "subject_number": "plural",
                        "object_number": "singular" if ctx_name == "mismatched" else "plural",
                        "prefix": prefix_pl,
                        "correct_verb": main_v_p,
                        "incorrect_verb": main_v_s,
                    })

                    item_id_counter += 1

    return items


def save_datasets(italian_items, finnish_items, output_dir="./datasets/final_datasets"):
    """Shuffles by item_id block and split into train/val/test."""
    assert len(italian_items) == len(finnish_items), "Mismatched item counts"
    n_blocks = max(item["item_id"] for item in italian_items) + 1
    block_indices = list(range(n_blocks))
    np.random.seed(42)
    np.random.shuffle(block_indices)

    it_blocks = {}
    fi_blocks = {}
    for item in italian_items:
        it_blocks.setdefault(item["item_id"], []).append(item)
    for item in finnish_items:
        fi_blocks.setdefault(item["item_id"], []).append(item)

    italian_ordered = []
    finnish_ordered = []
    for bid in block_indices:
        italian_ordered.extend(it_blocks[bid])
        finnish_ordered.extend(fi_blocks[bid])

    n = len(italian_ordered)
    train_end = n * 70 // 100
    val_end = n * 85 // 100

    splits = {
        "train": (0, train_end),
        "validation": (train_end, val_end),
        "test": (val_end, n),
    }

    os.makedirs(output_dir, exist_ok=True)
    for split, (start, end) in splits.items():
        it_split = {i: italian_ordered[start + i] for i in range(end - start)}
        fi_split = {i: finnish_ordered[start + i] for i in range(end - start)}

        with open(f"{output_dir}/italian_{split}_dataset.json", "w", encoding="utf-8") as f:
            json.dump(it_split, f, indent=4, ensure_ascii=False)
        with open(f"{output_dir}/finnish_{split}_dataset.json", "w", encoding="utf-8") as f:
            json.dump(fi_split, f, indent=4, ensure_ascii=False)

        print(f"Saved {split}: {len(it_split)} Italian, {len(fi_split)} Finnish items")


def create_sva_datasets():
    """Generates aligned Italian-Finnish SVA datasets."""
    italian_items = create_sva_dataset(
        subject_file="datasets/word_lists/italian_singular_plural_nouns.txt",
        object_file="datasets/word_lists/italian_singular_plural_nouns.txt",
        rel_verb_file="datasets/word_lists/italian_singular_plural_past_verbs.txt",
        main_verb_pair=('è', 'sono'),
        relativizers=('che', 'che'),
        language="italian"
    )

    finnish_items = create_sva_dataset(
        subject_file="datasets/word_lists/finnish_singular_plural_nouns_nom.txt",
        object_file="datasets/word_lists/finnish_singular_plural_nouns_part.txt",
        rel_verb_file="datasets/word_lists/finnish_singular_plural_past_verbs.txt",
        main_verb_pair=('on', 'ovat'),
        relativizers=('joka', 'jotka'),
        language="finnish"
    )

    save_datasets(italian_items, finnish_items)


if __name__ == "__main__":
    print("Creating datasets...")
    create_sva_datasets()
    print("Datasets created successfully.")
