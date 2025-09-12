SOCIAL_GROUPS = {
    "gender": ["female", "male"],
    "race": ["black", "white"], 
    "religion": ["christian", "muslim", "jewish"],
} ## how about more than 2 groups?

## the list of all groups
ALL_GROUPS = [group for groups in SOCIAL_GROUPS.values() for group in groups]

PREFIX = "/nethome/yifwang/fairness_x_explainability/new_fairness_explainability/utils/vocabulary"

def load_vocabulary(path):
    with open(path, "r", encoding="utf-8") as f:
        vocabulary = [line.strip().lower() for line in f.readlines()]
    return vocabulary

VOCABULARY_PATH = {group: f"{PREFIX}/{group}.txt" for group in ALL_GROUPS} ## vocabularies are in pairs for each group, there could be repeated ones
VOCABULARY = {group: load_vocabulary(path) for group, path in VOCABULARY_PATH.items()}
EXPANDED_VOCABULARY_PATH = {group: f"{PREFIX}/{group}_expanded.txt" for group in ALL_GROUPS} ## expanded vocabulary, e.g., with names, etc.
EXPANDED_VOCABULARY = {group: load_vocabulary(path) for group, path in EXPANDED_VOCABULARY_PATH.items()}

PERTURBATION_LIST = {bias_type: {orig_group:{perturbed_group:[[orig_word, perturbed_word] for orig_word, perturbed_word in zip(VOCABULARY[orig_group], VOCABULARY[perturbed_group])] for perturbed_group in SOCIAL_GROUPS[bias_type] if perturbed_group!=orig_group} for orig_group in SOCIAL_GROUPS[bias_type]} for bias_type in SOCIAL_GROUPS.keys()}
EXPANDED_PERTURBATION_LIST = {bias_type: {orig_group:{perturbed_group:[[orig_word, perturbed_word] for orig_word, perturbed_word in zip(EXPANDED_VOCABULARY[orig_group], EXPANDED_VOCABULARY[perturbed_group])] for perturbed_group in SOCIAL_GROUPS[bias_type] if perturbed_group!=orig_group} for orig_group in SOCIAL_GROUPS[bias_type]} for bias_type in SOCIAL_GROUPS.keys()}

FORBIDDEN_VOCABULARY_PATH = {group: f"{PREFIX}/{group}_forbidden.txt" for group in ALL_GROUPS}
FORBIDDEN_VOCABULARY = {group: load_vocabulary(path) for group, path in FORBIDDEN_VOCABULARY_PATH.items()}

SHOULD_CONTAIN = {bias_type: {group: list(set(VOCABULARY[group])) for group in SOCIAL_GROUPS[bias_type]} for bias_type in SOCIAL_GROUPS.keys()}
SHOULD_NOT_CONTAIN = {bias_type: {group: list(set(FORBIDDEN_VOCABULARY[group])) for group in SOCIAL_GROUPS[bias_type]} for bias_type in SOCIAL_GROUPS.keys()}
EXPANDED_SHOULD_CONTAIN = {bias_type: {group: list(set(EXPANDED_VOCABULARY[group])) for group in SOCIAL_GROUPS[bias_type]} for bias_type in SOCIAL_GROUPS.keys()}

SENSITIVE_TOKENS = {bias_type: {group: list(set(VOCABULARY[group])) for group in SOCIAL_GROUPS[bias_type]} for bias_type in SOCIAL_GROUPS.keys()}
EXPANDED_SENSITIVE_TOKENS = {bias_type: {group: list(set(EXPANDED_VOCABULARY[group])) for group in SOCIAL_GROUPS[bias_type]} for bias_type in SOCIAL_GROUPS.keys()}