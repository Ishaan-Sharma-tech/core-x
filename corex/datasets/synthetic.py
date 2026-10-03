import random
from dataclasses import dataclass


@dataclass
class SyntheticExample:
    inputs: list[int]
    target: object
    metadata: dict


def generate_identity_example(
    rng,
    length=8,
    vocab_size=16,
):
    inputs = [
        rng.randrange(vocab_size)
        for _ in range(length)
    ]

    return SyntheticExample(
        inputs=inputs,
        target=inputs.copy(),
        metadata={
            "task": "identity",
            "length": length,
            "vocab_size": vocab_size,
        },
    )


def generate_sequence_example(
    rng,
    length=8,
    vocab_size=16,
):
    inputs = [
        rng.randrange(vocab_size)
        for _ in range(length)
    ]

    return SyntheticExample(
        inputs=inputs,
        target=list(reversed(inputs)),
        metadata={
            "task": "sequence_reverse",
            "length": length,
            "vocab_size": vocab_size,
        },
    )


def generate_classification_example(
    rng,
    length=8,
    vocab_size=16,
):
    inputs = [
        rng.randrange(vocab_size)
        for _ in range(length)
    ]

    target = sum(inputs) % 2

    return SyntheticExample(
        inputs=inputs,
        target=target,
        metadata={
            "task": "parity_classification",
            "length": length,
            "vocab_size": vocab_size,
        },
    )


def generate_relation_example(
    rng,
    vocab_size=16,
):
    a, b, c = rng.sample(range(vocab_size), 3)

    relations = [
        (a, b),
        (b, c),
    ]

    query = (a, c)

    return SyntheticExample(
        inputs=relations,
        target=True,
        metadata={
            "task": "transitive_relation",
            "query": query,
            "vocab_size": vocab_size,
        },
    )


def generate_dataset(
    task: str,
    num_examples: int,
    seed: int = 42,
):
    rng = random.Random(seed)

    generators = {
        "identity": generate_identity_example,
        "sequence_reverse": generate_sequence_example,
        "parity_classification": generate_classification_example,
        "transitive_relation": generate_relation_example,
    }

    if task not in generators:
        raise ValueError(
            f"Unknown task: {task}. "
            f"Available: {list(generators)}"
        )

    generator = generators[task]

    return [
        generator(rng=rng)
        for _ in range(num_examples)
    ]


def split_dataset(
    examples,
    train_ratio=0.8,
    val_ratio=0.1,
):
    total = len(examples)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train = examples[:train_end]
    val = examples[train_end:val_end]
    test = examples[val_end:]

    return train, val, test


def generate_relation_chain_example(
    rng,
    chain_length=3,
    distractor_count=0,
    vocab_size=32,
):
    entities = list(range(vocab_size))
    rng.shuffle(entities)

    required_entities = chain_length + 1

    if required_entities > vocab_size:
        raise ValueError(
            "chain_length is too large for vocab_size."
        )

    chain_entities = entities[:required_entities]

    relations = [
        (
            chain_entities[i],
            chain_entities[i + 1],
        )
        for i in range(chain_length)
    ]

    distractors = []

    remaining = entities[required_entities:]

    for i in range(
        min(distractor_count, len(remaining) // 2)
    ):
        a = remaining[2 * i]
        b = remaining[2 * i + 1]
        distractors.append((a, b))

    all_relations = relations + distractors
    rng.shuffle(all_relations)

    query = (
        chain_entities[0],
        chain_entities[-1],
    )

    return SyntheticExample(
        inputs=all_relations,
        target=True,
        metadata={
            "task": "relation_chain",
            "chain_length": chain_length,
            "distractor_count": len(distractors),
            "query": query,
            "vocab_size": vocab_size,
        },
    )


def generate_compositional_relation_example(
    rng,
    chain_length=5,
    distractor_count=3,
    vocab_size=32,
    positive=True,
):
    entities = list(range(vocab_size))
    rng.shuffle(entities)

    required = chain_length + 1

    if required > vocab_size:
        raise ValueError(
            "chain_length is too large for vocab_size."
        )

    chain_entities = entities[:required]

    relations = [
        (
            chain_entities[i],
            chain_entities[i + 1],
        )
        for i in range(chain_length)
    ]

    remaining = entities[required:]

    distractors = []

    for i in range(
        min(distractor_count, len(remaining) // 2)
    ):
        distractors.append(
            (
                remaining[2 * i],
                remaining[2 * i + 1],
            )
        )

    all_relations = relations + distractors
    rng.shuffle(all_relations)

    if positive:
        query = (
            chain_entities[0],
            chain_entities[-1],
        )
        target = True
    else:
        used = set(chain_entities)

        candidates = [
            entity
            for entity in range(vocab_size)
            if entity not in used
        ]

        if not candidates:
            raise ValueError(
                "Need unused entities for negative query."
            )

        query = (
            chain_entities[0],
            candidates[0],
        )

        target = False

    return SyntheticExample(
        inputs=all_relations,
        target=target,
        metadata={
            "task": "compositional_relation",
            "chain_length": chain_length,
            "distractor_count": len(distractors),
            "query": query,
            "positive": positive,
            "vocab_size": vocab_size,
        },
    )
