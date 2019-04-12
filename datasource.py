import os
import re

DATA_ROOT = "JPEGImages"
SPLITS_ROOT = "./ImageSplits"
ACTIONS_FILE = "./actions.txt"

RE_ACTIONS_FILE = r"^\s*(?P<action_name>\w+)\s+(?P<number_of_images>\d+)\s*$"
RE_SPLIT_FILENAME = r"^(?P<action_name>\w+)_(?P<set_type>test|train).txt$"


class ActionData:
    def __init__(self, name, training_set, testing_set):
        self.name = name
        self.training_set = training_set
        self.testing_set = testing_set


def index_split_data(data_root=None, splits_root=None):
    if data_root is None:
        data_root = os.path.join(os.getcwd(), DATA_ROOT)

    if splits_root is None:
        splits_root = os.path.join(os.getcwd(), SPLITS_ROOT)

    actions = dict()

    # Find all action types as defined in actions.txt
    with open(os.path.join(splits_root, ACTIONS_FILE)) as action_file:
        data = action_file.read()

        # Extract action name from file using regular expression, and create new ActionData object to hold it
        for action_match in re.finditer(RE_ACTIONS_FILE, data, re.MULTILINE):
            action_name = action_match.group("action_name")
            actions[action_name] = ActionData(action_name, [], [])

    # Iterate over all files in the splits directory
    for split_filename in os.listdir(splits_root):
        split_match = re.match(RE_SPLIT_FILENAME, split_filename)

        # The regular expression does not match actions.txt, or any unexpected files
        if split_match is None:
            continue

        action_name = split_match.group("action_name")
        action_data = actions[action_name]

        split_path = os.path.join(splits_root, split_filename)

        # Read specific split file
        with open(split_path) as split_file:
            split_data = split_file.read().splitlines()

            split_type = split_match.group("set_type")

            # Assign lines to specific set
            if split_type == "train":
                action_data.training_set.extend(split_data)
            elif split_type == "test":
                action_data.testing_set.extend(split_data)

    for action in actions.values():
        action.training_set = verify_data_index(action.training_set, data_root)
        action.testing_set = verify_data_index(action.testing_set, data_root)

    filtered = list(filter(
        lambda a:
            a.testing_set is not None and len(a.testing_set) > 0 and
            a.training_set is not None and len(a.training_set) > 0, actions.values()))

    if len(filtered) == 0:
        raise Exception("No images found!")

    return list(filtered)


def verify_data_index(files, data_root):

    verified = []

    for file in files:
        # Make absolute path
        path = os.path.join(data_root, file)

        # Verify that file exists
        if os.path.exists(path):
            verified.append(path)

    return verified
