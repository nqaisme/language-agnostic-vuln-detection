from datasets import Dataset, DatasetDict

train_functions = [
    "int main() { return 0; }",
    "char buf[8]; gets(buf);",
    "void foo() { printf(\"Hello\"); }",
]

train_labels = [
    0,
    1,
    0,
]

test_functions = [
    "char *p = malloc(10);",
    "strcpy(buf, input);",
]

test_labels = [
    0,
    1,
]

# ==========================================================
# Tạo Dataset
# ==========================================================

train_dataset = Dataset.from_dict({
    "function": train_functions,
    "label": train_labels,
})

test_dataset = Dataset.from_dict({
    "function": test_functions,
    "label": test_labels,
})

tmp_dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset,
})