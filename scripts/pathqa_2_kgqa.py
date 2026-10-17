import argparse
import hashlib
import json
import logging
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union

import pandas as pd

LOGGER = logging.getLogger("pathqa_2_kgqa")

def setup_logging(question_path: str, dataset: str) -> str:
    """Log conversion output to both the console and the QA output folder."""
    log_path = os.path.join(
        question_path,
        f"{dataset.lower()}_conversion.log",
    )

    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    LOGGER.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        log_path,
        mode="w",
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    LOGGER.addHandler(stream_handler)
    LOGGER.addHandler(file_handler)

    LOGGER.info("Logging conversion output to: %s", log_path)
    return log_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert PathQuestion / PathQuestion-Large to the "
            "navigation-ready MultiHopKGQA format."
        )
    )

    # Input
    parser.add_argument(
        "--source-path",
        type=str,
        default="./data/source/PathQuestion/",
        help="Folder containing the PathQuestion dataset.",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Training proportion. Remaining data are divided evenly into dev/test.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used when constructing deterministic splits.",
    )
    parser.add_argument(
        "-l",
        "--use-large",
        action="store_true",
        help="Use PathQuestion-Large (PQL) instead of PathQuestion (PQ).",
    )
    parser.add_argument(
        "--encode-ids",
        action="store_true",
        help="Encode entity/relation identifiers in navigation-facing output fields.",
    )
    parser.add_argument(
        "--graph-regime",
        choices=["filtered", "source", "full", "both"],
        default="both",
        help=(
            "Graph condition(s) to materialize. 'filtered' uses the released "
            "hop-specific benchmark KGs; 'source'/'full' uses Freebase13.txt "
            "for PQ or FB2M_TRI_NEW.txt for PQL; 'both' writes both graph "
            "conditions in one run and uses one shared entity/relation encoding."
        ),
    )
    parser.add_argument(
        "--source-kg-file",
        type=str,
        default=None,
        help=(
            "Optional explicit source/full KG file. By default this is resolved "
            "as Freebase13.txt (PQ) or FB2M_TRI_NEW.txt (PQL) under --source-path."
        ),
    )
    parser.add_argument(
        "--allow-source-coverage-gaps",
        action="store_true",
        help=(
            "Allow released filtered triples or annotated path triples that are "
            "absent from the normalized source graph. Gaps are still logged and "
            "family answer coverage is still validated in every active graph."
        ),
    )

    # Output
    parser.add_argument(
        "--triplet-path",
        type=str,
        default="./data/link_prediction/",
        help="Root output directory for the navigation KG.",
    )
    parser.add_argument(
        "--question-path",
        type=str,
        default="./data/qa/",
        help="Root output directory for converted QA files.",
    )

    return parser.parse_args()

def save_triplets(df: pd.DataFrame, file_path: str) -> None:
    """
    Saves a pandas DataFrame containing triplets (head, relation, tail) to a tab-separated text file.
    """
    df.to_csv(file_path, sep='\t', header=False, index=False)

def load_pandas(file_path: Union[str, List[str]], sep: str = ',', header = 'infer', names = None) -> pd.DataFrame:
    """
    Loads a CSV file into a pandas DataFrame.
    """
    if type(file_path) is str:
        return pd.read_csv(file_path, sep=sep, header=header, names=names).fillna('')
    elif type(file_path) is list:
        df_list = [pd.read_csv(fp, sep=sep, header=header, names=names).fillna('') for fp in file_path]
        return pd.concat(df_list, ignore_index=True)
    else:
        assert False, 'Error! The file_path must either be a string or a list of strings'

def load_triplets(file_path: Union[str, List[str]], sep: str = '\t') -> pd.DataFrame:
    """
    Loads a triplet dataset from one or more file paths into a pandas DataFrame.
    """
    if type(file_path) is str:
        # Load the triplets into a DataFrame
        return pd.read_csv(file_path, sep=sep, header=None, names=['head', 'relation', 'tail'])
    elif type(file_path) is list:
        # Load and merge the DataFrames from the list of file paths
        df_list = [pd.read_csv(fp, sep=sep, header=None, names=['head', 'relation', 'tail']) for fp in file_path]
        return pd.concat(df_list, ignore_index=True)
    else:
        assert False, 'Error! The file_path must either be a string or a list of strings'

def split_path_tokens(tokens: List[str]) -> List[List[str]]:
    """
    Convert

        [e0, r0, e1, r1, e2, ...]

    into

        [[e0, r0, e1], [e1, r1, e2], ...].
    """
    if not tokens:
        raise ValueError("Path is empty.")

    if len(tokens) < 3 or len(tokens) % 2 == 0:
        raise ValueError(
            f"Malformed path with {len(tokens)} tokens: {tokens}"
        )

    hops = (len(tokens) - 1) // 2

    return [
        [
            tokens[2 * i],
            tokens[2 * i + 1],
            tokens[2 * i + 2],
        ]
        for i in range(hops)
    ]


def parse_annotated_path(
    path_string: str,
    expected_hops: int,
) -> Tuple[List[List[str]], str]:
    """
    Parse one released PQ/PQL entity-level reference path.

    The IRN-format files may append

        #<end>#ANSWER

    after the actual KG path.  <end> is a model/control symbol and is not
    treated as a KG relation.

    Returns
    -------
    triplets : list[list[str]]
        The actual KG transitions.
    end_entity : str
        The endpoint encoded by the annotated row.
    """
    if pd.isna(path_string):
        raise ValueError("Annotated path is empty.")

    tokens = str(path_string).strip().split("#")

    if "<end>" in tokens:
        end_idx = tokens.index("<end>")
        path_tokens = tokens[:end_idx]

        if end_idx + 1 >= len(tokens):
            raise ValueError(
                f"<end> has no following entity: {path_string}"
            )

        end_entity = tokens[end_idx + 1]

        if end_idx + 2 != len(tokens):
            raise ValueError(
                f"Unexpected tokens after <end> entity: {path_string}"
            )
    else:
        path_tokens = tokens
        end_entity = path_tokens[-1]

    triplets = split_path_tokens(path_tokens)

    if len(triplets) != expected_hops:
        raise ValueError(
            f"Expected {expected_hops} hops, but parsed {len(triplets)} "
            f"from: {path_string}"
        )

    if triplets[-1][2] != end_entity:
        raise ValueError(
            f"Path endpoint '{triplets[-1][2]}' does not match "
            f"end entity '{end_entity}': {path_string}"
        )

    return triplets, end_entity



def _find_answer_set_wrapper_start(value: str) -> int:
    """Return the opening ``(`` of the final balanced parenthesized suffix.

    PQ/PQL entity strings may themselves contain parentheses, e.g.
    ``PG_(USA)``.  Therefore the first ``(`` in the field cannot be used as
    the boundary between the designated answer and the released answer set.
    Instead, scan backward from the final ``)`` and find its matching opening
    parenthesis while respecting nested parentheses.
    """
    if not value.endswith(")"):
        return -1

    depth = 0
    for idx in range(len(value) - 1, -1, -1):
        char = value[idx]
        if char == ")":
            depth += 1
        elif char == "(":
            depth -= 1
            if depth == 0:
                return idx
            if depth < 0:
                break

    return -1


def parse_released_answer_field(answer_raw: str) -> Tuple[str, List[str]]:
    """
    Parse the released PQ/PQL answer field.

    The public IRN release stores rows as

        designated_answer(answer1/answer2/.../)

    where the prefix is the designated endpoint of the annotated path and
    the final parenthesized suffix is the complete released valid-answer set.

    Entity strings may themselves contain parentheses.  For example,

        PG_(USA)(PG_(USA)/)

    must be parsed as designated answer ``PG_(USA)`` and answer set
    ``["PG_(USA)"]``.

    Returns
    -------
    designated_answer : str
        The answer paired with the annotated reference path in this row.
    answer_set : list[str]
        The complete released valid-answer set, with duplicate entries
        removed while preserving release order.
    """
    if pd.isna(answer_raw):
        raise ValueError("Released answer field is empty.")

    value = str(answer_raw).strip()
    if not value:
        raise ValueError("Released answer field is empty.")

    wrapper_start = _find_answer_set_wrapper_start(value)

    if wrapper_start == -1:
        # Defensive fallback for malformed/legacy rows that contain no
        # explicit answer-set wrapper.
        designated = value
        answers = [designated]
    else:
        designated = value[:wrapper_start].strip()
        answer_payload = value[wrapper_start + 1:-1]

        # The released answer-set encoding uses '/' as the item separator
        # and normally includes a trailing '/'.  Parentheses inside entity
        # strings are preserved because only the outer wrapper was removed.
        answers = [
            answer.strip()
            for answer in answer_payload.split("/")
            if answer.strip()
        ]

    if not designated:
        raise ValueError(f"Missing designated answer in: {answer_raw}")

    # Preserve release order while removing accidental duplicates.
    answers = list(dict.fromkeys(answers))

    if not answers:
        raise ValueError(f"Released answer set is empty in: {answer_raw}")

    if designated not in answers:
        raise ValueError(
            f"Designated answer '{designated}' is not contained in the "
            f"released answer set {answers}: {answer_raw}"
        )

    return designated, answers


def make_path_key(path: List[List[str]]) -> str:
    """
    Convert an entity-level path into its ordered relation-chain key.

    Example
    -------
    [[A, r1, B], [B, r2, C]] -> "r1->r2"
    """
    if not path:
        raise ValueError("Cannot construct Path-Key from an empty path.")

    return "->".join(triplet[1] for triplet in path)


def build_relation_adjacency(
    kg_graph: pd.DataFrame,
) -> Dict[Tuple[str, str], Set[str]]:
    """Build (head, relation) -> tails adjacency for validation."""
    adjacency = defaultdict(set)

    for head, relation, tail in kg_graph[
        ["head", "relation", "tail"]
    ].itertuples(index=False, name=None):
        adjacency[(head, relation)].add(tail)

    return adjacency


def follow_relation_chain(
    source: str,
    path_key: str,
    adjacency: Dict[Tuple[str, str], Set[str]],
) -> Set[str]:
    """Return all entities reachable by following exactly one relation chain."""
    current = {source}

    for relation in path_key.split("->"):
        next_entities = set()

        for entity in current:
            next_entities.update(
                adjacency.get((entity, relation), set())
            )

        current = next_entities

        if not current:
            break

    return current


def enumerate_relation_chain_paths(
    source: str,
    path_key: str,
    adjacency: Dict[Tuple[str, str], Set[str]],
) -> List[List[List[str]]]:
    """Enumerate every unrestricted entity-level realization of one relation chain.

    The returned value is a deterministic ``List[Path]`` where each ``Path`` is
    ``List[[head, relation, tail]]``. Entity revisits/backtracking are intentionally
    allowed so Graph-Multi-Paths follows the same released walk semantics used for
    Graph-Answer. Exact duplicate paths are removed.
    """
    states: List[Tuple[str, List[List[str]]]] = [(source, [])]

    for relation in path_key.split("->"):
        next_states: List[Tuple[str, List[List[str]]]] = []
        for entity, path in states:
            for tail in sorted(adjacency.get((entity, relation), set())):
                next_states.append(
                    (tail, path + [[entity, relation, tail]])
                )
        states = next_states
        if not states:
            break

    unique_paths: Dict[Tuple[Tuple[str, str, str], ...], List[List[str]]] = {}
    for _, path in states:
        if len(path) != len(path_key.split("->")):
            continue
        signature = tuple(tuple(triple) for triple in path)
        unique_paths.setdefault(signature, path)

    return [unique_paths[key] for key in sorted(unique_paths)]



def follow_relation_chain_simple(
    source: str,
    path_key: str,
    adjacency: Dict[Tuple[str, str], Set[str]],
) -> Set[str]:
    """Return endpoints reachable by an entity-simple realization of Path-key.

    Unlike :func:`follow_relation_chain`, this traversal keeps path history and
    rejects any transition whose tail already appears earlier in that path.
    Thus A -> B -> A is excluded from this entity-simple diagnostic.
    PQ/PQL paths are only 2--3 hops, so exact path-state tracking is practical.
    This helper measures whether an answer has a realization without repeated
    entities and does not alter the unrestricted-walk semantics used for
    reconstruction and evaluation.
    """
    states: Set[Tuple[str, Tuple[str, ...]]] = {(source, (source,))}

    for relation in path_key.split("->"):
        next_states: Set[Tuple[str, Tuple[str, ...]]] = set()

        for entity, visited_path in states:
            for tail in adjacency.get((entity, relation), set()):
                if tail in visited_path:
                    continue
                next_states.add((tail, visited_path + (tail,)))

        states = next_states
        if not states:
            break

    return {entity for entity, _ in states}


def annotated_path_entity_sequence(path: List[List[str]]) -> List[str]:
    """Return the ordered entity sequence of an annotated KG path."""
    if not path:
        return []
    return [path[0][0]] + [triplet[2] for triplet in path]


def annotated_path_is_simple(path: List[List[str]]) -> bool:
    """Whether the released entity-level path visits every entity at most once."""
    entities = annotated_path_entity_sequence(path)
    return len(entities) == len(set(entities))


def annotated_path_has_backtrack(path: List[List[str]]) -> bool:
    """Whether the path immediately returns to the entity from two states ago."""
    entities = annotated_path_entity_sequence(path)
    return any(
        entities[idx] == entities[idx - 2]
        for idx in range(2, len(entities))
    )


def normalize_pql_source_entity(entity: str) -> str:
    """Map a PQL source-KG entity label to the released filtered convention."""
    return str(entity).strip().replace(" ", "_").replace("/", "_").replace("#", "_")


def normalize_pql_source_relation(relation: str) -> str:
    """Map a PQL source-KG Freebase relation path to filtered-PQL form."""
    return str(relation).strip().replace("/", "__")


def _collect_normalization_collisions(
    values: Set[str],
    normalizer,
) -> Dict[str, List[str]]:
    """Return canonical labels produced by more than one distinct raw value."""
    reverse: Dict[str, List[str]] = defaultdict(list)

    for raw_value in values:
        reverse[normalizer(raw_value)].append(raw_value)

    return {
        canonical: sorted(set(raw_values))
        for canonical, raw_values in reverse.items()
        if len(set(raw_values)) > 1
    }


def _check_normalization_collisions(
    values: Set[str],
    normalizer,
    value_type: str,
) -> Dict[str, List[str]]:
    """Return unresolved normalization collisions and log them as warnings."""
    collisions = _collect_normalization_collisions(values, normalizer)

    if collisions:
        examples = [
            f"{canonical!r} <- {raw_values}"
            for canonical, raw_values
            in list(collisions.items())[:10]
        ]

        LOGGER.warning(
            "PQL source %s normalization is not one-to-one after "
            "disambiguation. Found %s unresolved collisions. Examples: %s",
            value_type,
            len(collisions),
            examples,
        )

    return collisions


def build_pql_source_entity_normalization_map(
    raw_entities: Set[str],
) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, int]]:
    """Build a context-aware PQL source-entity normalization map.

    The ordinary released-style normalization maps spaces, '/', and '#' to '_'.
    That is retained for every non-colliding entity.  If two or more *distinct*
    source labels collapse to the same canonical value, only slash-containing
    members of that collision group are retried with '/' -> '__'.  For example::

        "Bitter Sweet"  -> "Bitter_Sweet"
        "Bitter/Sweet"  -> "Bitter__Sweet"

    Thus '__' is introduced for '/' only when the ordinary mapping would merge
    distinct source entities.  A final global collision check is still performed
    because a disambiguated label could theoretically collide with some other
    pre-existing source identifier.  Any such unresolved collision is returned
    for the later QA-traversal safety audit.
    """
    base_mapping = {
        raw: normalize_pql_source_entity(raw)
        for raw in raw_entities
    }
    base_collisions = _collect_normalization_collisions(
        raw_entities,
        lambda raw: base_mapping[raw],
    )

    final_mapping = dict(base_mapping)
    changed_examples = []

    for base_canonical, raw_values in sorted(base_collisions.items()):
        for raw_value in raw_values:
            if "/" not in raw_value:
                continue

            slash_disambiguated = (
                str(raw_value)
                .strip()
                .replace(" ", "_")
                .replace("/", "__")
                .replace("#", "_")
            )

            if slash_disambiguated != base_canonical:
                final_mapping[raw_value] = slash_disambiguated
                if len(changed_examples) < 10:
                    changed_examples.append(
                        f"{raw_value!r}: {base_canonical!r} -> "
                        f"{slash_disambiguated!r}"
                    )

    final_collisions = _collect_normalization_collisions(
        raw_entities,
        lambda raw: final_mapping[raw],
    )

    resolved_groups = 0
    for base_canonical, raw_values in base_collisions.items():
        final_values = [final_mapping[raw] for raw in raw_values]
        if (
            len(set(final_values)) == len(raw_values)
            and not any(value in final_collisions for value in final_values)
        ):
            resolved_groups += 1

    if base_collisions:
        LOGGER.warning(
            "PQL source entity base normalization produced %s collision groups. "
            "Resolved %s by using '__' for '/' only on colliding slash-containing "
            "labels; %s remain unresolved.",
            len(base_collisions),
            resolved_groups,
            len(final_collisions),
        )
        if changed_examples:
            LOGGER.info(
                "PQL source entity collision-disambiguation examples: %s",
                changed_examples,
            )

    if final_collisions:
        examples = [
            f"{canonical!r} <- {raw_values}"
            for canonical, raw_values
            in list(final_collisions.items())[:10]
        ]
        LOGGER.warning(
            "PQL source entity normalization still has %s unresolved collision "
            "groups after slash disambiguation. Examples: %s",
            len(final_collisions),
            examples,
        )

    normalization_stats = {
        "base_entity_collision_groups": len(base_collisions),
        "resolved_entity_collision_groups": resolved_groups,
        "unresolved_entity_collision_groups": len(final_collisions),
    }

    return final_mapping, final_collisions, normalization_stats


def normalize_source_kg(
    kg_graph: pd.DataFrame,
    use_large: bool,
) -> Tuple[
    pd.DataFrame,
    Dict[str, List[str]],
    Dict[str, List[str]],
    Dict[str, int],
]:
    """Canonicalize the source/full KG to the identifiers used by released QA.

    PQ already uses compatible lowercase/underscore identifiers. PQL source
    entities may contain spaces, '/', or '#', while source relations use Freebase
    '/' separators. Source PQL is normalized to the released convention before
    any coverage test or shared ID encoding.

    PQL entity normalization is deliberately two-pass. The standard mapping uses
    '_' for spaces, '/', and '#'. Only when that mapping would collapse distinct
    raw entities do slash-containing labels use '__' for '/', preserving the two
    source nodes without changing ordinary labels. Any collision that remains
    after this targeted disambiguation is reported and audited later against the
    actual benchmark traversals.
    """
    normalized = kg_graph[["head", "relation", "tail"]].copy()

    if not use_large:
        return (
            normalized.drop_duplicates().reset_index(drop=True),
            {},
            {},
            {
                "base_entity_collision_groups": 0,
                "resolved_entity_collision_groups": 0,
                "unresolved_entity_collision_groups": 0,
                "unresolved_relation_collision_groups": 0,
            },
        )

    raw_entities = set(normalized["head"]).union(set(normalized["tail"]))
    raw_relations = set(normalized["relation"])

    entity_mapping, entity_collisions, normalization_stats = (
        build_pql_source_entity_normalization_map(raw_entities)
    )

    relation_collisions = _check_normalization_collisions(
        raw_relations,
        normalize_pql_source_relation,
        "relation",
    )

    normalization_stats["unresolved_relation_collision_groups"] = len(
        relation_collisions
    )

    normalized["head"] = normalized["head"].map(entity_mapping)
    normalized["tail"] = normalized["tail"].map(entity_mapping)
    normalized["relation"] = normalized["relation"].map(
        normalize_pql_source_relation
    )

    return (
        normalized.drop_duplicates().reset_index(drop=True),
        entity_collisions,
        relation_collisions,
        normalization_stats,
    )


def trace_relation_chain(
    source: str,
    path_key: str,
    adjacency: Dict[Tuple[str, str], Set[str]],
) -> Tuple[Set[str], Set[str]]:
    """Return all entities and relations touched while executing one Path-key.

    The entity set includes the source, every intermediate entity reached after
    each hop, and all terminal entities. Relations are recorded when they are
    attempted from a reachable frontier. This is intentionally stricter than
    checking terminals only: an entity-normalization collision at an intermediate
    hop can splice together edges from distinct raw source-KG entities and create
    a synthetic downstream path.
    """
    current = {source}
    touched_entities = {source}
    touched_relations: Set[str] = set()

    for relation in path_key.split("->"):
        if not current:
            break

        touched_relations.add(relation)
        next_entities: Set[str] = set()
        for entity in current:
            next_entities.update(
                adjacency.get((entity, relation), set())
            )

        touched_entities.update(next_entities)
        current = next_entities

    return touched_entities, touched_relations


def build_normalization_collision_audit(
    qa_df: pd.DataFrame,
    adjacency: Dict[Tuple[str, str], Set[str]],
    entity_collisions: Dict[str, List[str]],
    relation_collisions: Dict[str, List[str]],
) -> pd.DataFrame:
    """Audit source-normalization collisions against benchmark traversals.

    Global source-KG collisions are allowed because most may be irrelevant to the
    benchmark. A collision is marked ``Used-In-QA-Traversal=True`` when its
    canonical entity is reached at any hop (including source/intermediate/terminal)
    or its canonical relation is actually attempted by a retained Path-key.
    """
    entity_usage: Dict[str, Set[str]] = defaultdict(set)
    relation_usage: Dict[str, Set[str]] = defaultdict(set)
    family_examples: Dict[str, Tuple[str, str]] = {}

    collided_entities = set(entity_collisions)
    collided_relations = set(relation_collisions)

    for _, row in qa_df.iterrows():
        family_id = str(row["Question-Family-ID"])
        family_examples[family_id] = (
            str(row["Question"]),
            str(row["Source-Entity"]),
        )

        for path_key in path_keys_to_list(row["Path-keys"]):
            touched_entities, touched_relations = trace_relation_chain(
                str(row["Source-Entity"]),
                path_key,
                adjacency,
            )

            for canonical in touched_entities & collided_entities:
                entity_usage[canonical].add(family_id)

            for canonical in touched_relations & collided_relations:
                relation_usage[canonical].add(family_id)

    rows = []
    for collision_type, collisions, usage in (
        ("entity", entity_collisions, entity_usage),
        ("relation", relation_collisions, relation_usage),
    ):
        for canonical, raw_values in sorted(collisions.items()):
            affected_families = sorted(usage.get(canonical, set()))
            example_questions = []
            example_sources = []
            for family_id in affected_families[:10]:
                question, source = family_examples[family_id]
                if question not in example_questions:
                    example_questions.append(question)
                if source not in example_sources:
                    example_sources.append(source)

            rows.append(
                {
                    "Type": collision_type,
                    "Canonical": canonical,
                    "Raw-Values": json.dumps(raw_values, ensure_ascii=False),
                    "Used-In-QA-Traversal": bool(affected_families),
                    "Num-Affected-Families": len(affected_families),
                    "Affected-Family-IDs": json.dumps(
                        affected_families, ensure_ascii=False
                    ),
                    "Example-Questions": json.dumps(
                        example_questions, ensure_ascii=False
                    ),
                    "Example-Sources": json.dumps(
                        example_sources, ensure_ascii=False
                    ),
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "Type",
            "Canonical",
            "Raw-Values",
            "Used-In-QA-Traversal",
            "Num-Affected-Families",
            "Affected-Family-IDs",
            "Example-Questions",
            "Example-Sources",
        ],
    )


def enforce_safe_normalization_collisions(
    collision_audit: pd.DataFrame,
    audit_path: str,
) -> None:
    """Continue on irrelevant collisions; fail if benchmark traversal touches one."""
    collision_audit.to_csv(audit_path, index=False)

    if len(collision_audit) == 0:
        LOGGER.info(
            "No unresolved source normalization collisions remain after "
            "disambiguation. Audit saved to: %s",
            audit_path,
        )
        return

    entity_count = int((collision_audit["Type"] == "entity").sum())
    relation_count = int((collision_audit["Type"] == "relation").sum())
    affected = collision_audit[
        collision_audit["Used-In-QA-Traversal"] == True  # noqa: E712
    ]

    LOGGER.warning(
        "Source normalization collisions: %s entity / %s relation; %s/%s are "
        "touched by retained benchmark traversals. Full audit: %s",
        entity_count,
        relation_count,
        len(affected),
        len(collision_audit),
        audit_path,
    )

    if len(affected) == 0:
        LOGGER.info(
            "✅ No source normalization ambiguity affects any retained QA traversal."
        )
        return

    LOGGER.error(
        "Normalization ambiguity affects benchmark traversal. Examples:\n%s",
        affected.head(10).to_string(index=False),
    )
    raise ValueError(
        f"{len(affected)} source normalization collision(s) are touched by "
        "retained benchmark Path-key traversals. Continuing could create "
        f"synthetic paths or answers. Inspect: {audit_path}"
    )


def resolve_existing_file(path: str) -> str:
    """Resolve a dataset file robustly across case-sensitive file systems."""
    if os.path.exists(path):
        return path

    parent = os.path.dirname(path) or "."
    basename = os.path.basename(path).lower()
    if os.path.isdir(parent):
        matches = [
            os.path.join(parent, name)
            for name in os.listdir(parent)
            if name.lower() == basename
        ]
        if len(matches) == 1:
            return matches[0]

    raise FileNotFoundError(path)


def path_keys_to_list(path_keys: Union[str, List[str], Tuple[str, ...]]) -> List[str]:
    """Normalize the mixed single-string/list Path-keys representation."""
    if isinstance(path_keys, str):
        return [path_keys]
    if isinstance(path_keys, tuple):
        return list(path_keys)
    if isinstance(path_keys, list):
        return list(path_keys)
    raise TypeError(f"Unsupported Path-keys value: {type(path_keys)!r}")


def compact_path_keys(path_keys: List[str]) -> Union[str, List[str]]:
    """Store one relation chain as a string; retain a list only for >1 chains."""
    unique = list(dict.fromkeys(path_keys))
    if not unique:
        raise ValueError("A question family must contain at least one Path-key.")
    if len(unique) == 1:
        return unique[0]
    return unique


def choose_canonical_question(
    questions: List[str],
    family_id: str,
    random_seed: int,
) -> str:
    """Choose one canonical question randomly but reproducibly within a family.

    ``questions`` is kept in deterministic released order. A family-local RNG is
    derived from the global seed and stable family ID so canonical selection is
    reproducible and does not depend on dataframe iteration order elsewhere.
    """
    if not questions:
        raise ValueError("Cannot select a canonical question from an empty family.")

    seed_payload = f"{random_seed}\x1f{family_id}"
    local_seed = int(
        hashlib.sha256(seed_payload.encode("utf-8")).hexdigest()[:16],
        16,
    )
    rng = random.Random(local_seed)
    return questions[rng.randrange(len(questions))]


def build_question_path_ambiguity_audit(
    released_qa_df: pd.DataFrame,
) -> pd.DataFrame:
    """Audit same Question + Path-Key + Hops with differing sources/answers.

    Rows are reported when the same surface question, relation chain, and hop
    count occur with either more than one source/topic entity or more than one
    released answer set. Source/topic differences always remain separate. Answer
    differences for the *same* question/topic/path/hop are intentionally merged
    later into one multi-answer evaluation instance.
    """
    work = released_qa_df.reset_index(drop=True).copy()
    work["Released-Row-Number"] = range(len(work))
    audit_rows = []

    for (question, path_key, hops), group in work.groupby(
        ["Question", "Path-Key", "Hops"],
        sort=False,
        dropna=False,
    ):
        sources = list(dict.fromkeys(group["Source-Entity"].tolist()))

        answer_sets = []
        for answers in group["Released-Answer-Entities"]:
            normalized = sorted(set(answers))
            if normalized not in answer_sets:
                answer_sets.append(normalized)

        source_ambiguous = len(sources) > 1
        answer_ambiguous = len(answer_sets) > 1
        if not source_ambiguous and not answer_ambiguous:
            continue

        ambiguity_types = []
        if source_ambiguous:
            ambiguity_types.append("source")
        if answer_ambiguous:
            ambiguity_types.append("answer-set")

        audit_rows.append(
            {                "Question": question,
                "Path-Key": path_key,
                "Hops": int(hops),
                "Ambiguity-Types": "+".join(ambiguity_types),
                "Num-Source-Entities": len(sources),
                "Source-Entities": json.dumps(sources, ensure_ascii=False),
                "Num-Answer-Sets": len(answer_sets),
                "Answer-Sets": json.dumps(answer_sets, ensure_ascii=False),
                "Released-Row-Numbers": json.dumps(
                    group["Released-Row-Number"].astype(int).tolist()
                ),
            }
        )

    return pd.DataFrame(
        audit_rows,
        columns=[
            "Question",
            "Path-Key",
            "Hops",
            "Ambiguity-Types",
            "Num-Source-Entities",
            "Source-Entities",
            "Num-Answer-Sets",
            "Answer-Sets",
            "Released-Row-Numbers",
        ],
    )


def assign_split_label(
    index: int,
    total_length: int,
    split_ratio: float,
) -> str:
    train_end = int(total_length * split_ratio)
    dev_end = int(
        total_length
        * (split_ratio + (1.0 - split_ratio) / 2.0)
    )

    if index < train_end:
        return "train"
    elif index < dev_end:
        return "dev"
    else:
        return "test"


def split_by_hop(
    qa_df: pd.DataFrame,
    split_ratio: float,
    random_seed: int,
) -> pd.DataFrame:
    """Split family-level QA instances independently within each hop count."""
    split_frames = []

    for hops in sorted(qa_df["Hops"].unique()):
        subset = (
            qa_df[qa_df["Hops"] == hops]
            .sample(frac=1, random_state=random_seed)
            .reset_index(drop=True)
        )

        total_length = len(subset)
        subset["SplitLabel"] = [
            assign_split_label(
                i,
                total_length,
                split_ratio,
            )
            for i in range(total_length)
        ]

        LOGGER.info(
            "%s-hop family split: %s train / %s dev / %s test",
            hops,
            int((subset["SplitLabel"] == "train").sum()),
            int((subset["SplitLabel"] == "dev").sum()),
            int((subset["SplitLabel"] == "test").sum()),
        )

        split_frames.append(subset)

    return pd.concat(split_frames, ignore_index=True)


class UnionFind:
    """Small union-find helper for transitive question-family grouping."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)

        if left_root == right_root:
            return

        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root

        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def answer_set_key(answers: List[str]) -> Tuple[str, ...]:
    """Canonical hash/grouping key for an unordered valid-answer set."""
    return tuple(sorted(set(answers)))


def make_question_family_id(
    dataset: str,
    source_entity: str,
    answers: List[str],
    hops: int,
    questions: List[str],
    path_keys: List[str],
) -> str:
    """Create a deterministic identifier from the complete family contents."""
    payload = "\x1f".join(
        [
            dataset,
            str(hops),
            source_entity,
            "\x1e".join(sorted(set(answers))),
            "\x1e".join(sorted(set(questions))),
            "\x1e".join(sorted(set(path_keys))),
        ]
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"{dataset.lower()}_qf_{hops}h_{digest}"


def build_question_families(
    released_qa_df: pd.DataFrame,
    dataset: str,
    random_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Group released rows into structural question families for split control.

    The family graph uses transitive closure over two kinds of links:

    1. Structural paraphrase link (both PQ and PQL):
       Source-Entity + Path-Key + Hops match. This links distinct surface
       realizations of the same released structural query even when answer-specific
       release rows expose different subsets/orderings of the valid answers. Their
       released answer sets are unioned at the family level.

    2. Exact-question compatibility link:
       * PQ:  Question + Source-Entity + Path-Key + Hops match. This merges the
         rare answer-specific duplicate even when designated/released answer
         annotations differ.
       * PQL: Question + Source-Entity + Hops match. This additionally merges
         alternate valid released relation chains for the same textual question
         and topic entity, so all such chains remain in one split/family.

    Because an exact question can now legitimately contribute multiple released
    answer sets and, for PQL, multiple Path-Keys, the family answer set is the
    ordered union of all member released answers and ``Path-keys`` is the ordered
    union of all member relation chains.

    ``Family-Size`` counts unique textual realizations. ``Num-Relation-Chains``
    counts unique retained relation chains. Families are only split after this
    aggregation, preventing alternate valid paths/answers for one QA problem from
    crossing train/dev/test partitions.
    """
    if dataset not in {"PQ", "PQL"}:
        raise ValueError(f"Unsupported PathQuestion dataset: {dataset}")

    work = released_qa_df.reset_index(drop=True).copy()
    work["Released-Row-Number"] = range(len(work))
    work["_Answer-Set-Key"] = work["Released-Answer-Entities"].apply(
        answer_set_key
    )

    union_find = UnionFind(len(work))
    structural_first: Dict[Tuple, int] = {}
    textual_first: Dict[Tuple, int] = {}

    for idx, row in work.iterrows():
        structural_key = (
            row["Source-Entity"],
            row["Path-Key"],
            int(row["Hops"]),
        )

        if dataset == "PQL":
            # One textual PQL QA problem may have multiple valid released paths
            # and answer-specific rows. Keep all of them in one family.
            textual_key = (
                row["Question"],
                row["Source-Entity"],
                int(row["Hops"]),
            )
        else:
            # For PQ, merge the rare answer-specific duplicate only when the
            # exact structural query also matches.
            textual_key = (
                row["Question"],
                row["Source-Entity"],
                row["Path-Key"],
                int(row["Hops"]),
            )

        if structural_key in structural_first:
            union_find.union(idx, structural_first[structural_key])
        else:
            structural_first[structural_key] = idx

        if textual_key in textual_first:
            union_find.union(idx, textual_first[textual_key])
        else:
            textual_first[textual_key] = idx

    components = defaultdict(list)
    for idx in range(len(work)):
        components[union_find.find(idx)].append(idx)

    ordered_components = sorted(
        components.values(),
        key=lambda indices: min(indices),
    )

    family_rows = []
    membership_rows = []

    for component_indices in ordered_components:
        group = work.iloc[component_indices].sort_values(
            "Released-Row-Number"
        )

        sources = group["Source-Entity"].drop_duplicates().tolist()
        hops_values = group["Hops"].drop_duplicates().tolist()

        if len(sources) != 1 or len(hops_values) != 1:
            raise ValueError(
                "Question-family transitive closure crossed an invariant "
                "boundary. Each family must have one source entity and hop count."
            )

        source_entity = sources[0]
        hops = int(hops_values[0])

        answers: List[str] = []
        for released_answers in group["Released-Answer-Entities"]:
            for answer in released_answers:
                if answer not in answers:
                    answers.append(answer)

        if not answers:
            raise ValueError("A question family must contain at least one released answer.")

        questions = list(dict.fromkeys(group["Question"].tolist()))
        path_key_list = list(dict.fromkeys(group["Path-Key"].tolist()))

        family_id = make_question_family_id(
            dataset=dataset,
            source_entity=source_entity,
            answers=answers,
            hops=hops,
            questions=questions,
            path_keys=path_key_list,
        )
        canonical_question = choose_canonical_question(
            questions=questions,
            family_id=family_id,
            random_seed=random_seed,
        )

        family_rows.append(
            {
                "Question": canonical_question,
                "Question-Paraphrased": list(questions),
                "Question-Family-ID": family_id,
                "Source": source_entity,
                "Source-Entity": source_entity,
                "Answer": list(answers),
                "Answer-Entity": list(answers),
                "Topic-In-Released-Answers": source_entity in set(answers),
                "Path-keys": compact_path_keys(path_key_list),
                "Hops": hops,
                "Family-Size": len(questions),
                "Num-Relation-Chains": len(path_key_list),
                "Merged-Rows": len(group),
            }
        )

        for _, member in group.iterrows():
            membership_rows.append(
                {
                    "Released-Row-Number": int(member["Released-Row-Number"]),
                    "Question-Family-ID": family_id,
                    "Is-Canonical-Question": (
                        member["Question"] == canonical_question
                    ),
                    "Question": member["Question"],
                    "Source-Entity": member["Source-Entity"],
                    "Released-Answer-Entities": list(
                        member["Released-Answer-Entities"]
                    ),
                    "Released-Designated-Answer": member[
                        "Designated-Answer-Entity"
                    ],
                    "Annotated-Path": [
                        list(triplet) for triplet in member["Paths"]
                    ],
                    "Annotated-Entity-Sequence": list(
                        member["Annotated-Entity-Sequence"]
                    ),
                    "Path-Key": member["Path-Key"],
                    "Hops": int(member["Hops"]),
                    "Annotated-Path-Simple": bool(
                        member["Annotated-Path-Simple"]
                    ),
                    "Annotated-Path-Backtracks": bool(
                        member["Annotated-Path-Backtracks"]
                    ),
                    "Topic-Is-Designated-Answer": bool(
                        member["Topic-Is-Designated-Answer"]
                    ),
                    "Topic-In-Released-Answers": bool(
                        member["Topic-In-Released-Answers"]
                    ),
                }
            )

    family_df = pd.DataFrame(family_rows)
    membership_df = pd.DataFrame(membership_rows)

    if family_df["Question-Family-ID"].duplicated().any():
        raise AssertionError("Question-Family-ID collision detected.")

    paraphrase_size_mismatches = family_df[
        family_df["Question-Paraphrased"].apply(len)
        != family_df["Family-Size"]
    ]
    if len(paraphrase_size_mismatches) > 0:
        raise AssertionError(
            "Question-Paraphrased must contain every unique textual variant, "
            "including the canonical question."
        )

    missing_canonical = family_df[
        ~family_df.apply(
            lambda row: row["Question"] in row["Question-Paraphrased"],
            axis=1,
        )
    ]
    if len(missing_canonical) > 0:
        raise AssertionError(
            "Canonical Question is missing from Question-Paraphrased."
        )

    return family_df, membership_df

def build_evaluation_instances(
    family_df: pd.DataFrame,
    membership_df: pd.DataFrame,
    dataset: str,
) -> pd.DataFrame:
    """Materialize QA evaluation instances while preserving all valid references.

    Structural families remain the unit of split assignment, but evaluation rows
    preserve the released textual QA problems.

    * PQ groups rows only when exact Question + Source-Entity + Path-Key + Hops
      match. Released answer sets and designated answers are unioned. This merges
      the rare answer-specific duplicate into one multi-answer instance.
    * PQL groups rows when exact Question + Source-Entity + Hops match. All unique
      released Path-Keys and answer sets in that group are unioned. Therefore a
      surface question with multiple released valid relation chains becomes one
      multi-path, multi-answer instance rather than several competing instances.

    Released designated endpoints and annotated entity paths are additionally
    pooled over every released surface row sharing Source-Entity + Path-Key + Hops
    with one of the instance's retained Path-keys.  This prevents path evaluation
    from penalizing a model that follows the same released relation chain to a
    different endpoint that the release itself designates on another surface row.
    Exact source-row provenance is retained separately.
    """
    if dataset not in {"PQ", "PQL"}:
        raise ValueError(f"Unsupported PathQuestion dataset: {dataset}")

    family_lookup = family_df.set_index("Question-Family-ID", drop=False)
    work = membership_df.sort_values("Released-Row-Number").reset_index(drop=True)

    required_columns = {
        "Released-Row-Number",
        "Question-Family-ID",
        "Question",
        "Source-Entity",
        "Released-Answer-Entities",
        "Released-Designated-Answer",
        "Annotated-Path",
        "Path-Key",
        "Hops",
        "Annotated-Path-Simple",
        "Annotated-Path-Backtracks",
        "SplitLabel",
    }
    missing_columns = required_columns - set(work.columns)
    if missing_columns:
        raise KeyError(
            "Evaluation-instance materialization is missing membership fields: "
            f"{sorted(missing_columns)}"
        )

    if dataset == "PQ":
        group_columns = [
            "Question",
            "Source-Entity",
            "Path-Key",
            "Hops",
        ]
    else:
        # Deliberately omit Path-Key: alternate released paths for one exact
        # textual question/topic/hop problem are valid references to the same QA.
        group_columns = [
            "Question",
            "Source-Entity",
            "Hops",
        ]

    grouped_members = list(
        work.groupby(
            group_columns,
            sort=False,
            dropna=False,
        )
    )

    rows = []
    for _, members in grouped_members:
        family_ids = list(dict.fromkeys(members["Question-Family-ID"].tolist()))
        questions = list(dict.fromkeys(members["Question"].tolist()))
        sources = list(dict.fromkeys(members["Source-Entity"].tolist()))
        path_keys = list(dict.fromkeys(members["Path-Key"].tolist()))
        hops_values = list(dict.fromkeys(members["Hops"].astype(int).tolist()))
        split_labels = list(dict.fromkeys(members["SplitLabel"].tolist()))

        if (
            len(family_ids) != 1
            or len(questions) != 1
            or len(sources) != 1
            or len(hops_values) != 1
            or len(split_labels) != 1
        ):
            raise ValueError(
                "Evaluation-instance grouping crossed an invariant boundary. "
                "Each instance must have one family, question, source, hop count, "
                "and split."
            )

        if dataset == "PQ" and len(path_keys) != 1:
            raise AssertionError(
                "PQ grouping must retain exactly one Path-Key per evaluation instance."
            )

        family_id = family_ids[0]
        if family_id not in family_lookup.index:
            raise KeyError(f"Missing family row for {family_id}")
        family = family_lookup.loc[family_id]

        # The benchmark's parenthesized answer set and designated endpoints are
        # structural supervision, not surface-form-specific supervision.  A
        # released row may designate only one endpoint even though another
        # surface realization of the same Source + Path-Key + Hops structure
        # designates a different valid endpoint.  For evaluation, collect all
        # released supervision attached to the retained Path-key(s), while still
        # preserving the exact source rows separately for provenance.
        reference_members = work[
            (work["Source-Entity"] == sources[0])
            & (work["Hops"].astype(int) == hops_values[0])
            & (work["Path-Key"].isin(path_keys))
        ].sort_values("Released-Row-Number")

        reference_family_ids = set(reference_members["Question-Family-ID"].tolist())
        if reference_family_ids != {family_id}:
            raise ValueError(
                "Structural reference pooling crossed a family boundary. "
                f"Instance family={family_id}; reference families={sorted(reference_family_ids)}"
            )

        instance_answers: List[str] = []
        for released_answers in reference_members["Released-Answer-Entities"]:
            for answer in released_answers:
                if answer not in instance_answers:
                    instance_answers.append(answer)

        if not instance_answers:
            raise ValueError(
                "Evaluation instance has no released answers. "
                f"Question={questions[0]!r}"
            )

        family_answers = list(family["Answer"])
        if not set(instance_answers).issubset(set(family_answers)):
            raise ValueError(
                "Evaluation-instance released answers are not contained in its "
                f"structural family for {family_id}."
            )

        family_path_keys = set(path_keys_to_list(family["Path-keys"]))
        if not set(path_keys).issubset(family_path_keys):
            raise ValueError(
                "Evaluation-instance Path-keys are not contained in its structural "
                f"family for {family_id}."
            )

        row_designated_answers = list(
            dict.fromkeys(members["Released-Designated-Answer"].tolist())
        )
        designated_answers = list(
            dict.fromkeys(reference_members["Released-Designated-Answer"].tolist())
        )
        missing_designated = set(designated_answers) - set(instance_answers)
        if missing_designated:
            raise ValueError(
                "Released designated answers are absent from the merged valid-answer "
                f"set. Family={family_id}; missing={sorted(missing_designated)}"
            )

        released_row_numbers = [
            int(value) for value in members["Released-Row-Number"].tolist()
        ]

        # Exact source-row provenance for this textual QA instance.
        released_row_path_answer_pairs = [
            {
                "Released-Row-Number": int(member["Released-Row-Number"]),
                "Path-Key": member["Path-Key"],
                "Designated-Answer": member["Released-Designated-Answer"],
                "Annotated-Path": [
                    list(triplet) for triplet in member["Annotated-Path"]
                ],
                "Annotated-Entity-Sequence": list(member["Annotated-Entity-Sequence"]),
            }
            for _, member in members.iterrows()
        ]

        # Evaluation-reference provenance pooled over every released row with the
        # same source/hop and one of this instance's retained Path-keys.  Duplicate
        # surface templates collapse to one structural reference path.
        reference_map = {}
        for _, member in reference_members.iterrows():
            annotated_path = tuple(
                tuple(triplet) for triplet in member["Annotated-Path"]
            )
            key = (
                member["Path-Key"],
                member["Released-Designated-Answer"],
                annotated_path,
            )
            if key not in reference_map:
                reference_map[key] = {
                    "Released-Row-Numbers": [],
                    "Path-Key": member["Path-Key"],
                    "Designated-Answer": member["Released-Designated-Answer"],
                    "Annotated-Path": [list(triplet) for triplet in annotated_path],
                    "Annotated-Entity-Sequence": list(
                        member["Annotated-Entity-Sequence"]
                    ),
                }
            reference_map[key]["Released-Row-Numbers"].append(
                int(member["Released-Row-Number"])
            )
        released_path_answer_pairs = list(reference_map.values())

        # v16 export semantics group released reasoning paths by unique relation
        # chain so Multi-Paths-Key never contains duplicates.  The aligned shape is:
        #
        #   Multi-Paths-Key[i] -> Multi-Paths[i] -> List[Path]
        #
        # where Path = List[[head, relation, tail]]. Reference-ID mirrors the same
        # key/path nesting and stores the released rows supporting each exact path.
        grouped_references: Dict[str, List[dict]] = {}
        for item in released_path_answer_pairs:
            grouped_references.setdefault(str(item["Path-Key"]), []).append(item)

        multi_paths_key_label = list(grouped_references.keys())
        if len(multi_paths_key_label) != len(set(multi_paths_key_label)):
            raise AssertionError("Duplicate Multi-Paths-Key values survived grouping.")

        multi_paths_label: List[List[List[List[str]]]] = []
        reference_ids: List[List[List[int]]] = []
        for path_key in multi_paths_key_label:
            path_group: List[List[List[str]]] = []
            reference_group: List[List[int]] = []
            for item in grouped_references[path_key]:
                path_group.append(
                    [list(triplet) for triplet in item["Annotated-Path"]]
                )
                reference_group.append(
                    [int(row_number) for row_number in item["Released-Row-Numbers"]]
                )
            multi_paths_label.append(path_group)
            reference_ids.append(reference_group)

        # Every annotated reference path must terminate at a released answer, and
        # collectively the annotated paths must cover the complete released answer
        # set. This does NOT require each distinct relation chain to cover every
        # answer; different valid chains may support different answers.
        annotated_terminals = {
            path[-1][2]
            for path_group in multi_paths_label
            for path in path_group
            if path
        }
        answer_set = set(instance_answers)
        invalid_terminals = annotated_terminals - answer_set
        missing_annotated_answers = answer_set - annotated_terminals
        if invalid_terminals or missing_annotated_answers:
            raise ValueError(
                "Released annotated Multi-Paths do not exactly cover the merged "
                "released answer set. "
                f"Family={family_id}; Question={questions[0]!r}; "
                f"invalid_terminals={sorted(invalid_terminals)!r}; "
                f"missing_answers={sorted(missing_annotated_answers)!r}"
            )

        path_count = sum(len(path_group) for path_group in multi_paths_label)

        rows.append(
            {
                "Question": questions[0],
                "Question-Paraphrased": list(family["Question-Paraphrased"]),
                "Question-Family-ID": family_id,
                "Question-Variant-Index": -1,
                "Family-Weight": 0.0,
                "Is-Canonical-Question": bool(
                    questions[0] == family["Question"]
                ),
                "Released-Row-Count": len(members),
                "Released-Row-Numbers": released_row_numbers,
                "Released-Row-Designated-Answers": row_designated_answers,
                "Released-Designated-Answers": designated_answers,
                "Released-Row-Path-Answer-Pairs": released_row_path_answer_pairs,
                "Released-Path-Answer-Pairs": released_path_answer_pairs,
                "Released-Reference-Paths": [
                    [list(triplet) for triplet in item["Annotated-Path"]]
                    for item in released_path_answer_pairs
                ],
                "Multi-Paths": multi_paths_label,
                "Multi-Paths-Label": multi_paths_label,
                "Multi-Paths-Key": multi_paths_key_label,
                "Path-Count": int(path_count),
                "Reference-ID": reference_ids,
                "Source": family["Source"],
                "Source-Entity": family["Source-Entity"],
                "Released-Answer": list(instance_answers),
                "Released-Answer-Entity": list(instance_answers),
                "Topic-In-Released-Answers": bool(
                    family["Source-Entity"] in set(instance_answers)
                ),
                "Variant-Has-Repeated-Entity-Path": bool(
                    (~members["Annotated-Path-Simple"]).any()
                ),
                "Variant-Has-Immediate-Backtrack": bool(
                    members["Annotated-Path-Backtracks"].any()
                ),
                "Answer": list(instance_answers),
                "Answer-Entity": list(instance_answers),
                "Path-keys": compact_path_keys(path_keys),
                "Hops": hops_values[0],
                "SplitLabel": split_labels[0],
                "Family-Size": int(family["Family-Size"]),
                "Question-Family-Size": int(family["Family-Size"]),
                "Family-Instance-Count": 0,
                "Num-Relation-Chains": len(path_keys),
                "Family-Num-Relation-Chains": int(
                    family["Num-Relation-Chains"]
                ),
            }
        )

    instance_df = pd.DataFrame(rows)
    if len(instance_df) == 0:
        raise ValueError("No evaluation instances were materialized.")

    instance_df["Question-Variant-Index"] = (
        instance_df.groupby("Question-Family-ID", sort=False).cumcount()
    )
    family_instance_counts = instance_df.groupby(
        "Question-Family-ID",
        sort=False,
    )["Question"].transform("size")
    instance_df["Family-Instance-Count"] = family_instance_counts.astype(int)
    instance_df["Family-Weight"] = 1.0 / family_instance_counts.astype(float)

    split_counts = (
        instance_df.groupby("Question-Family-ID")["SplitLabel"].nunique()
    )
    if (split_counts > 1).any():
        raise AssertionError(
            "At least one structural family crosses train/dev/test after "
            "evaluation-instance materialization."
        )

    released_rows = len(work)
    evaluation_instances = len(instance_df)
    rows_collapsed = released_rows - evaluation_instances
    hop_counts = (
        instance_df["Hops"].value_counts().sort_index().to_dict()
    )

    LOGGER.info(
        "Merged evaluation-instance accounting (%s): %s released rows -> %s "
        "evaluation instances (%s rows collapsed).",
        dataset,
        released_rows,
        evaluation_instances,
        rows_collapsed,
    )
    LOGGER.info(
        "Evaluation-instance counts by hop: %s",
        hop_counts,
    )
    LOGGER.info(
        "Family-Weight is 1 / Family-Instance-Count, so weighting final QA "
        "instances by this field yields equal contribution from every structural "
        "family (family-macro aggregation)."
    )

    # Exact-path repetition diagnostic shared by both datasets.
    exact_path_columns = [
        "Question",
        "Source-Entity",
        "Path-Key",
        "Hops",
    ]
    exact_path_groups = list(
        work.groupby(
            exact_path_columns,
            sort=False,
            dropna=False,
        )
    )
    repeated_exact_path_groups = [
        members for _, members in exact_path_groups if len(members) > 1
    ]

    if dataset == "PQ":
        LOGGER.info(
            "PQ exact-QA grouping: grouping key = Question + Source-Entity + "
            "Path-Key + Hops; released answer sets and designated answers are "
            "unioned. Merged %s repeated group(s), absorbing %s released row(s).",
            len(repeated_exact_path_groups),
            rows_collapsed,
        )
    else:
        # Diagnose exact questions that contribute multiple released relation chains.
        question_topic_hop_columns = [
            "Question",
            "Source-Entity",
            "Hops",
        ]
        multi_chain_examples = []
        multi_chain_instance_count = 0
        multi_chain_path_count = 0

        for group_key, members in work.groupby(
            question_topic_hop_columns,
            sort=False,
            dropna=False,
        ):
            unique_paths = list(dict.fromkeys(members["Path-Key"].tolist()))
            if len(unique_paths) <= 1:
                continue

            multi_chain_instance_count += 1
            multi_chain_path_count += len(unique_paths)

            merged_answers: List[str] = []
            for released_answers in members["Released-Answer-Entities"]:
                for answer in released_answers:
                    if answer not in merged_answers:
                        merged_answers.append(answer)

            if len(multi_chain_examples) < 10:
                question, source_entity, hops = group_key
                multi_chain_examples.append(
                    {
                        "Question": question,
                        "Source-Entity": source_entity,
                        "Hops": int(hops),
                        "Path-keys": unique_paths,
                        "Merged-Released-Answers": merged_answers,
                        "Released-Row-Numbers": [
                            int(value)
                            for value in members["Released-Row-Number"].tolist()
                        ],
                    }
                )

        LOGGER.info(
            "PQL merged-QA grouping: grouping key = Question + Source-Entity + "
            "Hops. Answer-specific rows and alternate released relation chains "
            "are merged by unioning Released-Answer and Path-keys."
        )
        LOGGER.info(
            "PQL exact question/topic/hop instances with >1 released relation "
            "chain: %s; %s distinct Path-keys participate in those instances.",
            multi_chain_instance_count,
            multi_chain_path_count,
        )
        if multi_chain_examples:
            LOGGER.info(
                "PQL merged multi-chain QA examples: %s",
                multi_chain_examples,
            )

    multi_chain_output_instances = int(
        (instance_df["Num-Relation-Chains"] > 1).sum()
    )
    LOGGER.info(
        "Final evaluation instances with multiple valid Path-keys: %s/%s.",
        multi_chain_output_instances,
        len(instance_df),
    )

    return instance_df

def validate_family_relation_chains(
    qa_df: pd.DataFrame,
    adjacency: Dict[Tuple[str, str], Set[str]],
) -> None:
    """Validate multi-chain family semantics against the released KG.

    A listed relation chain is considered valid for a family when it reaches at
    least one released valid answer.  Across all listed chains, the union of
    reached valid answers must cover the family's complete released answer set.

    This is intentionally weaker than requiring every chain to reach every
    answer: different valid chains may support different members of a multi-
    answer set.
    """
    invalid_chains = []
    incomplete_families = []

    for idx, row in qa_df.iterrows():
        answer_set = set(row["Answer-Entity"])
        covered_answers = set()

        for path_key in path_keys_to_list(row["Path-keys"]):
            reachable = follow_relation_chain(
                row["Source-Entity"],
                path_key,
                adjacency,
            )
            valid_reached = answer_set.intersection(reachable)

            if not valid_reached:
                invalid_chains.append(
                    (
                        idx,
                        row["Question-Family-ID"],
                        row["Question"],
                        path_key,
                    )
                )
            else:
                covered_answers.update(valid_reached)

        missing_answers = answer_set - covered_answers
        if missing_answers:
            incomplete_families.append(
                (
                    idx,
                    row["Question-Family-ID"],
                    row["Question"],
                    sorted(missing_answers),
                )
            )

    if invalid_chains:
        LOGGER.error(
            "Some family Path-keys do not reach any released valid answer. "
            "Examples: %s",
            invalid_chains[:10],
        )
        raise ValueError(
            f"{len(invalid_chains)} family relation chains do not reach any "
            "released valid answer."
        )

    if incomplete_families:
        LOGGER.error(
            "Some family answer sets are not covered by the union of Path-keys. "
            "Examples: %s",
            incomplete_families[:10],
        )
        raise ValueError(
            f"{len(incomplete_families)} question families contain released "
            "answers not covered by any retained relation chain."
        )

    LOGGER.info(
        "✅ All family Path-keys reach released valid answers, and the union "
        "of relation-chain endpoints covers each family answer set."
    )


def validate_evaluation_instance_relation_chains(
    qa_df: pd.DataFrame,
    adjacency: Dict[Tuple[str, str], Set[str]],
    regime: str,
) -> None:
    """Hard-validate the final merged QA semantics in one active graph.

    For every final evaluation instance:

    * every retained Path-key must reach at least one released valid answer; and
    * the union of endpoints over all retained Path-keys must cover the complete
      merged released answer set.

    The second condition is the key v13 invariant for multi-path QA instances.
    It guarantees that merging alternate released relation chains cannot create
    an answer that is unsupported by all of the instance's retained paths.
    """
    invalid_paths = []
    incomplete_instances = []

    for idx, row in qa_df.iterrows():
        released_answers = set(row["Released-Answer-Entity"])
        covered_released = set()

        for path_key in path_keys_to_list(row["Path-keys"]):
            reachable = follow_relation_chain(
                row["Source-Entity"],
                path_key,
                adjacency,
            )
            valid_reached = released_answers.intersection(reachable)
            if not valid_reached:
                invalid_paths.append(
                    (
                        idx,
                        row["Question-Family-ID"],
                        row["Question"],
                        path_key,
                    )
                )
            covered_released.update(valid_reached)

        missing = released_answers - covered_released
        if missing:
            incomplete_instances.append(
                (
                    idx,
                    row["Question-Family-ID"],
                    row["Question"],
                    sorted(missing),
                    path_keys_to_list(row["Path-keys"]),
                )
            )

    if invalid_paths:
        LOGGER.error(
            "%s graph: some final evaluation-instance Path-keys do not reach "
            "any released answer. Examples: %s",
            regime,
            invalid_paths[:10],
        )
        raise ValueError(
            f"{len(invalid_paths)} final evaluation-instance relation chains "
            f"do not reach any released answer in the {regime} graph."
        )

    if incomplete_instances:
        LOGGER.error(
            "%s graph: some final merged released answer sets are not covered "
            "by the union of retained Path-keys. Examples: %s",
            regime,
            incomplete_instances[:10],
        )
        raise ValueError(
            f"{len(incomplete_instances)} final evaluation instances contain "
            f"released answers not reachable through any retained Path-key in "
            f"the {regime} graph."
        )

    LOGGER.info(
        "✅ All final merged evaluation instances are relation-chain valid in "
        "the %s graph: every Path-key reaches at least one released answer and "
        "the union of Path-key endpoints covers the complete released answer set.",
        regime,
    )


def expand_answers_from_graph(    qa_df: pd.DataFrame,
    adjacency: Dict[Tuple[str, str], Set[str]],
    regime: str,
) -> pd.DataFrame:
    """Audit graph-relative endpoints and materialize graph-consistent paths.

    ``Answer`` remains the merged released benchmark target. ``Graph-Answer``
    contains every endpoint reachable under the retained relation chain(s). v16
    additionally enumerates every unrestricted graph realization of each unique
    Multi-Paths-Key and stores them in Graph-Multi-Paths / Label using the same
    key-grouped nesting as Multi-Paths.
    """
    result = qa_df.copy(deep=True)

    graph_answer_lists: List[List[str]] = []
    graph_multi_paths_lists: List[List[List[List[List[str]]]]] = []
    graph_path_counts: List[int] = []
    added_answer_lists: List[List[str]] = []
    simple_answer_lists: List[List[str]] = []
    cycle_only_lists: List[List[str]] = []
    released_only_cyclic_lists: List[List[str]] = []
    missing_released_lists: List[List[str]] = []
    topic_in_graph_answers: List[bool] = []

    path_cache: Dict[Tuple[str, str], List[List[List[str]]]] = {}

    for _, row in result.iterrows():
        released_answers = list(dict.fromkeys(row["Released-Answer"]))
        released_set = set(released_answers)
        unrestricted_union: Set[str] = set()
        simple_union: Set[str] = set()
        graph_path_groups: List[List[List[List[str]]]] = []

        unique_path_keys = list(row["Multi-Paths-Key"])
        if len(unique_path_keys) != len(set(unique_path_keys)):
            raise ValueError(
                f"{regime} graph received duplicate Multi-Paths-Key values for "
                f"question {row['Question']!r}: {unique_path_keys!r}"
            )

        for path_key in unique_path_keys:
            cache_key = (str(row["Source-Entity"]), str(path_key))
            if cache_key not in path_cache:
                path_cache[cache_key] = enumerate_relation_chain_paths(
                    row["Source-Entity"],
                    path_key,
                    adjacency,
                )
            graph_paths = [
                [[head, relation, tail] for head, relation, tail in path]
                for path in path_cache[cache_key]
            ]
            if not graph_paths:
                raise ValueError(
                    f"{regime} graph has no realization for retained Path-key "
                    f"{path_key!r} from source {row['Source-Entity']!r}."
                )
            graph_path_groups.append(graph_paths)
            unrestricted_union.update(path[-1][2] for path in graph_paths)
            simple_union.update(
                follow_relation_chain_simple(
                    row["Source-Entity"],
                    path_key,
                    adjacency,
                )
            )

        missing_released = [
            answer for answer in released_answers
            if answer not in unrestricted_union
        ]
        if missing_released:
            raise ValueError(
                f"{regime} graph does not support all released answers for final "
                f"evaluation instance {row['Question']!r}. "
                f"Source={row['Source-Entity']!r}; "
                f"Path-keys={unique_path_keys!r}; "
                f"missing={missing_released!r}"
            )

        added_answers = sorted(unrestricted_union - released_set)
        graph_answers = list(released_answers) + added_answers

        simple_released = [
            answer for answer in released_answers if answer in simple_union
        ]
        simple_added = sorted(simple_union - released_set)
        simple_answers = simple_released + simple_added

        cycle_only = sorted(unrestricted_union - simple_union)
        released_only_cyclic = [
            answer for answer in released_answers
            if answer in unrestricted_union and answer not in simple_union
        ]

        # Every released annotated path must exist among the corresponding graph
        # realizations in every active graph. Compare human-readable path tuples so
        # this check is independent of later EID/RID encoding.
        for key_index, annotated_group in enumerate(row["Multi-Paths-Label"]):
            graph_group_signatures = {
                tuple(tuple(triple) for triple in path)
                for path in graph_path_groups[key_index]
            }
            for annotated_path in annotated_group:
                signature = tuple(tuple(triple) for triple in annotated_path)
                if signature not in graph_group_signatures:
                    raise ValueError(
                        f"Released annotated path is absent from the {regime} graph "
                        f"for question {row['Question']!r}, "
                        f"Path-key={unique_path_keys[key_index]!r}: {annotated_path!r}"
                    )

        graph_terminal_set = {
            path[-1][2]
            for path_group in graph_path_groups
            for path in path_group
        }
        if graph_terminal_set != unrestricted_union:
            raise AssertionError(
                f"Graph-Multi-Paths terminal coverage mismatch in {regime} graph."
            )

        graph_answer_lists.append(graph_answers)
        graph_multi_paths_lists.append(graph_path_groups)
        graph_path_counts.append(
            sum(len(path_group) for path_group in graph_path_groups)
        )
        added_answer_lists.append(added_answers)
        simple_answer_lists.append(simple_answers)
        cycle_only_lists.append(cycle_only)
        released_only_cyclic_lists.append(released_only_cyclic)
        missing_released_lists.append([])
        topic_in_graph_answers.append(
            row["Source-Entity"] in unrestricted_union
        )

    released_alignment_errors = result[
        ~result.apply(
            lambda row: (
                list(row["Answer"]) == list(row["Released-Answer"])
                and list(row["Answer-Entity"])
                == list(row["Released-Answer-Entity"])
            ),
            axis=1,
        )
    ]
    if len(released_alignment_errors) > 0:
        raise ValueError(
            f"{len(released_alignment_errors)} evaluation instances do not keep "
            "Answer aligned with Released-Answer before graph auditing."
        )

    result["Graph-Answer"] = [list(answers) for answers in graph_answer_lists]
    result["Graph-Answer-Entity"] = [
        list(answers) for answers in graph_answer_lists
    ]
    result["Graph-Multi-Paths"] = graph_multi_paths_lists
    result["Graph-Multi-Paths-Label"] = graph_multi_paths_lists
    result["Graph-Path-Count"] = graph_path_counts
    result["Added-Graph-Answers"] = added_answer_lists
    result["Simple-Path-Answers"] = simple_answer_lists
    result["Cycle-Only-Answers"] = cycle_only_lists
    result["Released-Answers-Only-Cyclic"] = released_only_cyclic_lists
    result["Released-Answers-Not-Reachable"] = missing_released_lists
    result["Topic-In-Graph-Answers"] = topic_in_graph_answers

    family_added_answers: Dict[str, Set[str]] = defaultdict(set)
    for family_id, added_answers in zip(
        result["Question-Family-ID"],
        added_answer_lists,
    ):
        family_added_answers[str(family_id)].update(added_answers)

    family_count = result["Question-Family-ID"].nunique()
    families_changed = sum(
        bool(answers) for answers in family_added_answers.values()
    )
    family_total_added = sum(
        len(answers) for answers in family_added_answers.values()
    )
    family_max_added = max(
        (len(answers) for answers in family_added_answers.values()),
        default=0,
    )

    changed_count = sum(bool(answers) for answers in added_answer_lists)
    total_added = sum(len(answers) for answers in added_answer_lists)
    max_added = max((len(answers) for answers in added_answer_lists), default=0)
    cycle_affected = sum(bool(answers) for answers in cycle_only_lists)
    released_cycle_dependent = sum(
        bool(answers) for answers in released_only_cyclic_lists
    )
    topic_answer_count = sum(topic_in_graph_answers)

    LOGGER.info(
        "%s graph answer semantics: primary Answer/Answer-Entity remain the "
        "merged released benchmark targets; Graph-Answer/Graph-Answer-Entity "
        "store unrestricted relation-chain endpoints for structural auditing.",
        regime,
    )
    LOGGER.info(
        "%s graph path materialization: %s unique unrestricted graph paths "
        "across %s evaluation instances.",
        regime,
        sum(graph_path_counts),
        len(result),
    )
    LOGGER.info(
        "%s graph family-level structural answer expansion: %s/%s families "
        "gained at least one additional relation-chain-consistent terminal entity.",
        regime,
        families_changed,
        family_count,
    )
    LOGGER.info(
        "%s graph family-level structural answer expansion: %s added "
        "family-answer pairs (each added endpoint counted once per family); max "
        "added to one family=%s.",
        regime,
        family_total_added,
        family_max_added,
    )
    LOGGER.info(
        "%s graph evaluation-instance-weighted answer expansion: %s/%s instances "
        "gained at least one additional relation-chain-consistent terminal entity.",
        regime,
        changed_count,
        len(result),
    )
    LOGGER.info(
        "%s graph evaluation-instance-weighted answer expansion: %s total added "
        "answers after counting repeated instances separately; max added to one "
        "instance=%s.",
        regime,
        total_added,
        max_added,
    )
    LOGGER.info(
        "%s graph cycle audit: %s/%s evaluation instances have at least one "
        "endpoint reachable only when entity revisits are allowed; %s evaluation "
        "instances have at least one released answer that is cycle-dependent.",
        regime,
        cycle_affected,
        len(result),
        released_cycle_dependent,
    )
    LOGGER.info(
        "%s graph topic/answer identity: topic entity is in Graph-Answer for "
        "%s/%s evaluation instances.",
        regime,
        topic_answer_count,
        len(result),
    )

    examples = []
    for row_idx in range(len(result)):
        added = added_answer_lists[row_idx]
        cycle_only = cycle_only_lists[row_idx]
        released_only_cyclic = released_only_cyclic_lists[row_idx]
        topic_is_answer = topic_in_graph_answers[row_idx]
        if not (added or cycle_only or released_only_cyclic or topic_is_answer):
            continue
        row = result.iloc[row_idx]
        examples.append(
            {
                "Question-Family-ID": row["Question-Family-ID"],
                "Question": row["Question"],
                "Source-Entity": row["Source-Entity"],
                "Released-Answer": row["Released-Answer"],
                "Graph-Answer": graph_answer_lists[row_idx],
                "Graph-Path-Count": graph_path_counts[row_idx],
                "Added-Graph-Answers": added,
                "Cycle-Only-Answers": cycle_only,
                "Released-Answers-Only-Cyclic": released_only_cyclic,
                "Topic-In-Graph-Answers": topic_is_answer,
            }
        )
        if len(examples) >= 10:
            break

    if examples:
        LOGGER.info(
            "%s graph answer/cycle-audit examples: %s",
            regime,
            examples,
        )

    return result


def encode_path_key(
    path_key: str,
    relation2id: Dict[str, str],
) -> str:
    """Map one relation-chain key to encoded relation IDs."""
    return "->".join(
        relation2id[relation]
        for relation in path_key.split("->")
    )


def encode_path_keys(
    path_keys: Union[str, List[str], Tuple[str, ...]],
    relation2id: Dict[str, str],
) -> Union[str, List[str]]:
    """Encode Path-keys while preserving the single-string/list convention."""
    encoded = [
        encode_path_key(path_key, relation2id)
        for path_key in path_keys_to_list(path_keys)
    ]
    return compact_path_keys(encoded)


def encode_multi_path(
    path: List[List[str]],
    entity2id: Dict[str, str],
    relation2id: Dict[str, str],
) -> List[List[str]]:
    """Encode one entity-level reasoning path."""
    return [
        [entity2id[head], relation2id[relation], entity2id[tail]]
        for head, relation, tail in path
    ]


def encode_multi_paths(
    path_groups: List[List[List[List[str]]]],
    entity2id: Dict[str, str],
    relation2id: Dict[str, str],
) -> List[List[List[List[str]]]]:
    """Encode key-grouped paths while preserving the v16 nested shape."""
    return [
        [
            encode_multi_path(path, entity2id, relation2id)
            for path in path_group
        ]
        for path_group in path_groups
    ]


def encode_multi_path_keys(
    path_keys: List[str],
    relation2id: Dict[str, str],
) -> List[str]:
    """Encode unique path keys; v16 always returns List[str]."""
    if len(path_keys) != len(set(path_keys)):
        raise ValueError(f"Duplicate Multi-Paths-Key values: {path_keys!r}")
    return [encode_path_key(path_key, relation2id) for path_key in path_keys]


def validate_multi_path_export_fields(qa_df: pd.DataFrame) -> None:
    """Validate v16 annotated/graph path grouping, coverage, and counts."""
    for row_index, row in qa_df.iterrows():
        paths = row["Multi-Paths"]
        labels = row["Multi-Paths-Label"]
        keys = row["Multi-Paths-Key"]
        references = row["Reference-ID"]
        graph_paths = row["Graph-Multi-Paths"]
        graph_labels = row["Graph-Multi-Paths-Label"]

        if not all(
            isinstance(value, list)
            for value in (paths, labels, keys, references, graph_paths, graph_labels)
        ):
            raise TypeError(
                f"v16 Multi-* fields must all use list containers (row {row_index})."
            )
        if not keys or not paths:
            raise ValueError(f"Evaluation instance {row_index} has no Multi-Paths.")
        if len(keys) != len(set(keys)):
            raise ValueError(
                f"Duplicate Multi-Paths-Key values at row {row_index}: {keys!r}"
            )
        if not (
            len(paths)
            == len(labels)
            == len(keys)
            == len(references)
            == len(graph_paths)
            == len(graph_labels)
        ):
            raise ValueError(
                "Multi-Paths, Multi-Paths-Label, Multi-Paths-Key, Reference-ID, "
                "Graph-Multi-Paths, and Graph-Multi-Paths-Label must align by "
                f"unique relation-chain group (row {row_index})."
            )

        annotated_count = 0
        graph_count = 0
        annotated_terminals = set()
        graph_terminals = set()

        for key_index, path_key in enumerate(keys):
            if not isinstance(path_key, str):
                raise TypeError(
                    f"Every Multi-Paths-Key entry must be a string (row {row_index})."
                )
            path_group = paths[key_index]
            label_group = labels[key_index]
            ref_group = references[key_index]
            graph_group = graph_paths[key_index]
            graph_label_group = graph_labels[key_index]

            if not all(
                isinstance(value, list)
                for value in (path_group, label_group, ref_group, graph_group, graph_label_group)
            ):
                raise TypeError(
                    f"Each relation-chain group must be list-valued (row {row_index}, key {key_index})."
                )
            if not path_group or not graph_group:
                raise ValueError(
                    f"Empty path group at row {row_index}, key {key_index}."
                )
            if len(path_group) != len(label_group) or len(path_group) != len(ref_group):
                raise ValueError(
                    f"Annotated path/label/reference groups do not align at row {row_index}, key {key_index}."
                )
            if len(graph_group) != len(graph_label_group):
                raise ValueError(
                    f"Graph path/label groups do not align at row {row_index}, key {key_index}."
                )

            graph_label_signatures = {
                tuple(tuple(triple) for triple in path)
                for path in graph_label_group
            }

            for path_index, (path, label_path, reference_ids) in enumerate(
                zip(path_group, label_group, ref_group)
            ):
                annotated_count += 1
                if len(path) != int(row["Hops"]) or len(label_path) != int(row["Hops"]):
                    raise ValueError(
                        f"Annotated path hop mismatch at row {row_index}, key {key_index}, path {path_index}."
                    )
                if not isinstance(reference_ids, list) or not reference_ids or not all(
                    isinstance(value, int) for value in reference_ids
                ):
                    raise TypeError(
                        "Reference-ID must be List[List[List[int]]] aligned to "
                        f"key/path groups (row {row_index}, key {key_index}, path {path_index})."
                    )
                for triple in path + label_path:
                    if not isinstance(triple, list) or len(triple) != 3:
                        raise TypeError(
                            "Each reasoning-path transition must be [head, relation, tail]. "
                            f"Row={row_index}, key={key_index}, path={path_index}."
                        )
                derived_key = "->".join(triple[1] for triple in path)
                if derived_key != path_key:
                    raise ValueError(
                        "Multi-Paths-Key does not match the aligned encoded path at "
                        f"row {row_index}, key {key_index}: {derived_key!r} != {path_key!r}"
                    )
                annotated_terminals.add(path[-1][2])
                if tuple(tuple(triple) for triple in label_path) not in graph_label_signatures:
                    raise ValueError(
                        f"Annotated path is absent from Graph-Multi-Paths-Label at row {row_index}, key {key_index}."
                    )

            for path_index, path in enumerate(graph_group):
                graph_count += 1
                if len(path) != int(row["Hops"]):
                    raise ValueError(
                        f"Graph path hop mismatch at row {row_index}, key {key_index}, path {path_index}."
                    )
                for triple in path:
                    if not isinstance(triple, list) or len(triple) != 3:
                        raise TypeError(
                            "Each graph-path transition must be [head, relation, tail]. "
                            f"Row={row_index}, key={key_index}, path={path_index}."
                        )
                derived_key = "->".join(triple[1] for triple in path)
                if derived_key != path_key:
                    raise ValueError(
                        "Multi-Paths-Key does not match the aligned Graph-Multi-Paths "
                        f"path at row {row_index}, key {key_index}: {derived_key!r} != {path_key!r}"
                    )
                graph_terminals.add(path[-1][2])

        if annotated_count != int(row["Path-Count"]):
            raise ValueError(
                f"Path-Count mismatch at row {row_index}: {row['Path-Count']} != {annotated_count}"
            )
        if graph_count != int(row["Graph-Path-Count"]):
            raise ValueError(
                f"Graph-Path-Count mismatch at row {row_index}: {row['Graph-Path-Count']} != {graph_count}"
            )
        if annotated_terminals != set(row["Answer-Entity"]):
            raise ValueError(
                "Annotated Multi-Paths terminal set does not equal Answer-Entity at "
                f"row {row_index}: terminals={sorted(annotated_terminals)!r}, "
                f"answers={sorted(set(row['Answer-Entity']))!r}"
            )
        if graph_terminals != set(row["Graph-Answer-Entity"]):
            raise ValueError(
                "Graph-Multi-Paths terminal set does not equal Graph-Answer-Entity at "
                f"row {row_index}: terminals={sorted(graph_terminals)!r}, "
                f"answers={sorted(set(row['Graph-Answer-Entity']))!r}"
            )


def encode_graph(
    kg_graph: pd.DataFrame,
    entity2id: Dict[str, str],
    relation2id: Dict[str, str],
) -> pd.DataFrame:
    """Encode one graph using a precomputed shared mapping."""
    encoded = kg_graph.copy()
    encoded["head"] = encoded["head"].map(entity2id)
    encoded["tail"] = encoded["tail"].map(entity2id)
    encoded["relation"] = encoded["relation"].map(relation2id)

    if encoded[["head", "relation", "tail"]].isna().any().any():
        raise ValueError("Graph encoding produced missing entity/relation IDs.")

    return encoded


def save_qa_outputs(
    qa_output: pd.DataFrame,
    qa_metadata_output: pd.DataFrame,
    question_path: str,
    dataset: str,
) -> Tuple[str, str]:
    """Save compact v16 QA files and their companion audit/metadata files."""
    os.makedirs(question_path, exist_ok=True)
    mixed_output = os.path.join(
        question_path,
        f"{dataset.lower()}_qa_nhop.csv",
    )
    mixed_metadata_output = os.path.join(
        question_path,
        f"{dataset.lower()}_qa_nhop_metadata.csv",
    )
    qa_output.to_csv(mixed_output, index=False)
    qa_metadata_output.to_csv(mixed_metadata_output, index=False)

    for hops in range(2, 4):
        qa_output[qa_output["Hops"] == hops].to_csv(
            os.path.join(
                question_path,
                f"{dataset.lower()}_qa_{hops}hop.csv",
            ),
            index=False,
        )
        qa_metadata_output[qa_metadata_output["Hops"] == hops].to_csv(
            os.path.join(
                question_path,
                f"{dataset.lower()}_qa_{hops}hop_metadata.csv",
            ),
            index=False,
        )

    return mixed_output, mixed_metadata_output



if __name__ == "__main__":

    args = parse_args()

    requested_graph_regime = args.graph_regime
    if requested_graph_regime == "full":
        requested_graph_regime = "source"

    if args.use_large:
        dataset = "PQL"
        prefixes = [f"{dataset}2-", f"{dataset}3-"]
    else:
        dataset = "PQ"
        prefixes = ["2H-", "3H-"]

    base_triplet_path = os.path.join(args.triplet_path, dataset)
    base_question_path = os.path.join(args.question_path, dataset)
    os.makedirs(base_triplet_path, exist_ok=True)
    os.makedirs(base_question_path, exist_ok=True)
    log_path = setup_logging(base_question_path, dataset)

    LOGGER.info("Dataset: %s", dataset)
    LOGGER.info("Source path: %s", args.source_path)
    LOGGER.info("Split ratio: %.4f", args.split_ratio)
    LOGGER.info("Random seed: %s", args.random_seed)
    LOGGER.info("Encode IDs: %s", args.encode_ids)
    LOGGER.info("Requested graph regime: %s", args.graph_regime)

    # ================================================================
    # 1. Load the released filtered KG and optional source/full KG
    # ================================================================

    filtered_kg_paths = [
        resolve_existing_file(
            os.path.join(
                args.source_path,
                prefix + "KB.txt",
            )
        )
        for prefix in prefixes
    ]

    filtered_kg_graph = (
        load_triplets(filtered_kg_paths, sep="\t")
        .drop_duplicates()
        .reset_index(drop=True)
    )
    filtered_triplet_set = set(
        map(
            tuple,
            filtered_kg_graph[["head", "relation", "tail"]].to_numpy(),
        )
    )
    filtered_adjacency = build_relation_adjacency(filtered_kg_graph)

    LOGGER.info(
        "Loaded released mixed filtered KG with %s unique triplets.",
        len(filtered_kg_graph),
    )
    LOGGER.info(
        "Filtered KG entities: %s",
        len(set(filtered_kg_graph["head"]).union(set(filtered_kg_graph["tail"]))),
    )
    LOGGER.info(
        "Filtered KG relations: %s",
        filtered_kg_graph["relation"].nunique(),
    )

    graph_frames: Dict[str, pd.DataFrame] = {"filtered": filtered_kg_graph}
    graph_triplet_sets: Dict[str, Set[Tuple[str, str, str]]] = {
        "filtered": filtered_triplet_set
    }
    graph_adjacencies = {"filtered": filtered_adjacency}
    source_kg_file: Optional[str] = None
    source_entity_collisions: Dict[str, List[str]] = {}
    source_relation_collisions: Dict[str, List[str]] = {}
    source_normalization_stats: Dict[str, int] = {
        "base_entity_collision_groups": 0,
        "resolved_entity_collision_groups": 0,
        "unresolved_entity_collision_groups": 0,
        "unresolved_relation_collision_groups": 0,
    }

    if requested_graph_regime in {"source", "both"}:
        if args.source_kg_file is not None:
            source_kg_file = resolve_existing_file(args.source_kg_file)
        else:
            default_source_filename = (
                "FB2M_TRI_NEW.txt" if args.use_large else "Freebase13.txt"
            )
            source_kg_file = resolve_existing_file(
                os.path.join(args.source_path, default_source_filename)
            )

        LOGGER.info("Loading source/full KG from: %s", source_kg_file)
        source_kg_raw = load_triplets([source_kg_file], sep="\t")
        LOGGER.info("Loaded %s source/full KG triple rows.", len(source_kg_raw))

        raw_rows = len(source_kg_raw)

        raw_unique = len(
            source_kg_raw[
                ["head", "relation", "tail"]
            ].drop_duplicates()
        )

        LOGGER.info(
            "Source/full KG raw rows: %s; raw unique triples: %s; "
            "duplicate raw rows: %s.",
            raw_rows,
            raw_unique,
            raw_rows - raw_unique,
        )

        (
            source_kg_graph,
            source_entity_collisions,
            source_relation_collisions,
            source_normalization_stats,
        ) = normalize_source_kg(
            source_kg_raw,
            use_large=args.use_large,
        )

        normalized_unique = len(source_kg_graph)

        LOGGER.info(
            "Normalization-induced triple collapses: %s.",
            raw_unique - normalized_unique,
        )
        del source_kg_raw

        source_triplet_set = set(
            map(
                tuple,
                source_kg_graph[["head", "relation", "tail"]].to_numpy(),
            )
        )
        source_adjacency = build_relation_adjacency(source_kg_graph)

        graph_frames["source"] = source_kg_graph
        graph_triplet_sets["source"] = source_triplet_set
        graph_adjacencies["source"] = source_adjacency

        LOGGER.info(
            "Normalized/deduplicated source/full KG: %s unique triplets.",
            len(source_kg_graph),
        )
        LOGGER.info(
            "Source/full KG entities: %s",
            len(set(source_kg_graph["head"]).union(set(source_kg_graph["tail"]))),
        )
        LOGGER.info(
            "Source/full KG relations: %s",
            source_kg_graph["relation"].nunique(),
        )

        missing_filtered_triplets = filtered_triplet_set - source_triplet_set
        LOGGER.info(
            "Filtered triples absent from normalized source/full KG: %s/%s",
            len(missing_filtered_triplets),
            len(filtered_triplet_set),
        )
        if missing_filtered_triplets:
            LOGGER.warning(
                "Examples of filtered/source coverage gaps: %s",
                list(missing_filtered_triplets)[:10],
            )
            if not args.allow_source_coverage_gaps:
                raise ValueError(
                    "The normalized source/full KG does not contain every "
                    "released filtered triple. Inspect PQL identifier "
                    "normalization or rerun with --allow-source-coverage-gaps "
                    "after verifying the missing triples."
                )

    if requested_graph_regime == "filtered":
        active_regimes = ["filtered"]
    elif requested_graph_regime == "source":
        active_regimes = ["source"]
    else:
        active_regimes = ["filtered", "source"]

    # ================================================================
    # 2. Load and parse released PQ/PQL rows
    # ================================================================

    hop_frames = []

    for hops in range(2, 4):
        qa_path = resolve_existing_file(
            os.path.join(
                args.source_path,
                f"{dataset}-{hops}H.txt",
            )
        )
        qa_df = load_pandas(
            qa_path,
            sep="\t",
            header=None,
            names=["Question", "Answer-Raw", "Path-Raw"],
        )

        LOGGER.info(
            "Loaded %s released %s-hop rows from %s.",
            len(qa_df),
            hops,
            qa_path,
        )

        parsed = qa_df["Path-Raw"].apply(
            lambda value: parse_annotated_path(
                value,
                expected_hops=hops,
            )
        )
        qa_df["Paths"] = parsed.apply(lambda x: x[0])
        qa_df["End"] = parsed.apply(lambda x: x[1])
        qa_df["Hops"] = hops
        qa_df["Source-Entity"] = qa_df["Paths"].apply(
            lambda path: path[0][0]
        )

        parsed_answers = qa_df["Answer-Raw"].apply(
            parse_released_answer_field
        )
        qa_df["Designated-Answer-Entity"] = parsed_answers.apply(
            lambda x: x[0]
        )
        qa_df["Released-Answer-Entities"] = parsed_answers.apply(
            lambda x: x[1]
        )
        qa_df["Annotated-Answer-Entity"] = qa_df["Paths"].apply(
            lambda path: path[-1][2]
        )
        qa_df["Path-Key"] = qa_df["Paths"].apply(make_path_key)
        qa_df["Annotated-Entity-Sequence"] = qa_df["Paths"].apply(
            annotated_path_entity_sequence
        )
        qa_df["Annotated-Path-Simple"] = qa_df["Paths"].apply(
            annotated_path_is_simple
        )
        qa_df["Annotated-Path-Backtracks"] = qa_df["Paths"].apply(
            annotated_path_has_backtrack
        )
        qa_df["Topic-Is-Designated-Answer"] = (
            qa_df["Source-Entity"] == qa_df["Designated-Answer-Entity"]
        )
        qa_df["Topic-In-Released-Answers"] = qa_df.apply(
            lambda row: row["Source-Entity"] in set(row["Released-Answer-Entities"]),
            axis=1,
        )

        LOGGER.info(
            "%s-hop annotated paths with repeated entities: %s/%s; "
            "immediate backtracks: %s/%s.",
            hops,
            int((~qa_df["Annotated-Path-Simple"]).sum()),
            len(qa_df),
            int(qa_df["Annotated-Path-Backtracks"].sum()),
            len(qa_df),
        )
        LOGGER.info(
            "%s-hop topic/answer identity: topic equals designated answer in "
            "%s/%s rows; topic appears anywhere in the released answer set in "
            "%s/%s rows.",
            hops,
            int(qa_df["Topic-Is-Designated-Answer"].sum()),
            len(qa_df),
            int(qa_df["Topic-In-Released-Answers"].sum()),
            len(qa_df),
        )

        endpoint_mismatches = qa_df[
            (qa_df["Annotated-Answer-Entity"] != qa_df["End"])
            | (
                qa_df["Annotated-Answer-Entity"]
                != qa_df["Designated-Answer-Entity"]
            )
        ]
        if len(endpoint_mismatches) > 0:
            LOGGER.error(
                "Endpoint mismatch examples:\n%s",
                endpoint_mismatches[
                    [
                        "Question",
                        "Answer-Raw",
                        "Annotated-Answer-Entity",
                        "Designated-Answer-Entity",
                        "End",
                        "Path-Raw",
                    ]
                ].head(10).to_string(index=False),
            )
            raise ValueError(
                f"{len(endpoint_mismatches)} released {hops}-hop rows "
                "disagree on the designated path endpoint."
            )
        LOGGER.info(
            "✅ All %s-hop designated answers match the annotated path endpoint.",
            hops,
        )

        path_triplets = {
            tuple(triplet)
            for path in qa_df["Paths"]
            for triplet in path
        }

        # The released paths must always exist in the released filtered graph.
        missing_filtered_path_triplets = path_triplets - filtered_triplet_set
        if missing_filtered_path_triplets:
            LOGGER.error(
                "Missing released annotated path triples from filtered KG: %s",
                list(missing_filtered_path_triplets)[:10],
            )
            raise ValueError(
                f"{len(missing_filtered_path_triplets)} annotated {hops}-hop "
                "path triples are missing from the released filtered KG."
            )
        LOGGER.info(
            "✅ All %s-hop annotated path triples are present in the filtered KG.",
            hops,
        )

        # If source/full is active, separately audit exact annotated path coverage.
        if "source" in active_regimes:
            missing_source_path_triplets = (
                path_triplets - graph_triplet_sets["source"]
            )
            LOGGER.info(
                "%s-hop annotated path triples absent from normalized source/full KG: %s/%s",
                hops,
                len(missing_source_path_triplets),
                len(path_triplets),
            )
            if missing_source_path_triplets:
                LOGGER.warning(
                    "Examples of missing source/full annotated path triples: %s",
                    list(missing_source_path_triplets)[:10],
                )
                if not args.allow_source_coverage_gaps:
                    raise ValueError(
                        f"{len(missing_source_path_triplets)} annotated {hops}-hop "
                        "path triples are absent from the normalized source/full KG."
                    )

        # Each released row contributes one relation chain. Its designated answer
        # must be reachable under the legacy unrestricted relation-chain semantics.
        # Simple-path reachability is audited separately rather than forced: a
        # released answer that requires a cycle/backtrack is precisely a benchmark
        # artifact we want to measure.
        for regime in active_regimes:
            unreachable_designated = []
            not_simple_reachable = []
            adjacency = graph_adjacencies[regime]
            for row_idx, row in qa_df.iterrows():
                reachable = follow_relation_chain(
                    row["Source-Entity"],
                    row["Path-Key"],
                    adjacency,
                )
                simple_reachable = follow_relation_chain_simple(
                    row["Source-Entity"],
                    row["Path-Key"],
                    adjacency,
                )
                if row["Designated-Answer-Entity"] not in reachable:
                    unreachable_designated.append(
                        (
                            row_idx,
                            row["Question"],
                            row["Source-Entity"],
                            row["Path-Key"],
                            row["Designated-Answer-Entity"],
                        )
                    )
                if row["Designated-Answer-Entity"] not in simple_reachable:
                    not_simple_reachable.append(
                        (
                            row_idx,
                            row["Question"],
                            row["Source-Entity"],
                            row["Path-Key"],
                            row["Designated-Answer-Entity"],
                        )
                    )

            if unreachable_designated:
                LOGGER.error(
                    "Some designated answers are unreachable through their row "
                    "Path-Key in %s graph. Examples: %s",
                    regime,
                    unreachable_designated[:10],
                )
                raise ValueError(
                    f"{len(unreachable_designated)} released {hops}-hop rows "
                    f"have a designated answer unreachable in the {regime} graph."
                )
            LOGGER.info(
                "✅ All %s-hop designated answers are reachable through their "
                "released relation chain in the %s graph (unrestricted).",
                hops,
                regime,
            )
            if not_simple_reachable:
                LOGGER.warning(
                    "%s-hop designated answers not reachable through any "
                    "entity-simple realization in the %s graph: %s/%s. "
                    "Examples: %s",
                    hops,
                    regime,
                    len(not_simple_reachable),
                    len(qa_df),
                    not_simple_reachable[:10],
                )
            else:
                LOGGER.info(
                    "✅ All %s-hop designated answers also have an entity-simple "
                    "realization in the %s graph.",
                    hops,
                    regime,
                )

        hop_frames.append(qa_df)

    released_qa_df = pd.concat(hop_frames, ignore_index=True)
    released_qa_df["Released-Row-Number"] = range(len(released_qa_df))
    LOGGER.info("Total released rows: %s", len(released_qa_df))

    annotated_path_audit_path = os.path.join(
        base_question_path,
        f"{dataset.lower()}_annotated_path_validity.csv",
    )
    annotated_path_audit = released_qa_df[
        [
            "Released-Row-Number",
            "Question",
            "Source-Entity",
            "Designated-Answer-Entity",
            "Path-Key",            "Hops",
            "Annotated-Entity-Sequence",
            "Annotated-Path-Simple",
            "Annotated-Path-Backtracks",
            "Topic-Is-Designated-Answer",
            "Topic-In-Released-Answers",
            "Path-Raw",
        ]
    ].copy()
    annotated_path_audit.to_csv(annotated_path_audit_path, index=False)
    LOGGER.info(
        "Annotated-path audit: %s/%s released rows revisit an entity; %s/%s "
        "contain an immediate backtrack.",
        int((~annotated_path_audit["Annotated-Path-Simple"]).sum()),
        len(annotated_path_audit),
        int(annotated_path_audit["Annotated-Path-Backtracks"].sum()),
        len(annotated_path_audit),
    )
    LOGGER.info(
        "Topic/answer identity audit: topic equals the designated answer in "
        "%s/%s released rows; topic appears anywhere in the released answer "
        "set in %s/%s rows.",
        int(annotated_path_audit["Topic-Is-Designated-Answer"].sum()),
        len(annotated_path_audit),
        int(annotated_path_audit["Topic-In-Released-Answers"].sum()),
        len(annotated_path_audit),
    )
    LOGGER.info(
        "✅ Saved annotated-path validity audit to: %s",
        annotated_path_audit_path,
    )

    # ================================================================
    # 3. Cross-source diagnostic and structural question families
    # ================================================================

    ambiguity_audit = build_question_path_ambiguity_audit(
        released_qa_df
    )
    ambiguity_audit_path = os.path.join(
        base_question_path,
        f"{dataset.lower()}_same_question_path_ambiguity.csv",
    )
    ambiguity_audit.to_csv(ambiguity_audit_path, index=False)

    source_ambiguity_count = int(
        ambiguity_audit["Ambiguity-Types"].str.contains("source", na=False).sum()
    ) if len(ambiguity_audit) else 0
    answer_ambiguity_count = int(
        ambiguity_audit["Ambiguity-Types"].str.contains("answer-set", na=False).sum()
    ) if len(ambiguity_audit) else 0

    LOGGER.info(
        "Same Question + Path-Key + Hops ambiguity groups: %s total "
        "(%s with different Source-Entity; %s with different answer sets).",
        len(ambiguity_audit),
        source_ambiguity_count,
        answer_ambiguity_count,
    )
    if len(ambiguity_audit) > 0:
        LOGGER.warning(
            "Source/topic ambiguity remains separate. For the same source/topic, "
            "answer-set differences are merged when the exact QA grouping policy "
            "permits it (PQ: same Question+Path-Key+Hops; PQL: same "
            "Question+Hops, including alternate valid Path-keys). Examples:\n%s",
            ambiguity_audit.head(10).to_string(index=False),
        )
    LOGGER.info(
        "✅ Saved same-question/path ambiguity audit to: %s",
        ambiguity_audit_path,
    )

    qa_df, membership_df = build_question_families(
        released_qa_df,
        dataset=dataset,
        random_seed=args.random_seed,
    )

    LOGGER.info(
        "Grouped %s released rows into %s structural question families "
        "(%s released annotations linked within families).",
        len(released_qa_df),
        len(qa_df),
        len(released_qa_df) - len(qa_df),
    )
    LOGGER.info(
        "Family counts by hop: %s",
        qa_df["Hops"].value_counts().sort_index().to_dict(),
    )
    LOGGER.info(
        "Family-size distribution (# unique textual questions; canonical included): %s",
        qa_df["Family-Size"].value_counts().sort_index().to_dict(),
    )
    LOGGER.info(
        "Question-Paraphrased contains the canonical question plus every other "
        "unique textual variant in released order."
    )
    LOGGER.info(
        "Relation-chain-count distribution: %s",
        qa_df["Num-Relation-Chains"].value_counts().sort_index().to_dict(),
    )
    LOGGER.info(
        "Families with >1 textual realization: %s/%s",
        int((qa_df["Family-Size"] > 1).sum()),
        len(qa_df),
    )
    LOGGER.info(
        "Families with multiple relation chains: %s/%s",
        int((qa_df["Num-Relation-Chains"] > 1).sum()),
        len(qa_df),
    )
    LOGGER.info(
        "Maximum textual variants in one family: %s",
        int(qa_df["Family-Size"].max()),
    )
    LOGGER.info(
        "Maximum relation chains in one family: %s",
        int(qa_df["Num-Relation-Chains"].max()),
    )

    answer_counts = qa_df["Answer-Entity"].apply(len)
    LOGGER.info(
        "Families with >1 valid answer: %s/%s",
        int((answer_counts > 1).sum()),
        len(qa_df),
    )
    LOGGER.info(
        "Average valid answers per family: %.4f",
        answer_counts.mean(),
    )
    LOGGER.info(
        "Maximum valid answers for one family: %s",
        int(answer_counts.max()),
    )
    LOGGER.info(
        "Families where the topic entity is itself a released valid answer: "
        "%s/%s",
        int(qa_df["Topic-In-Released-Answers"].sum()),
        len(qa_df),
    )

    # ================================================================
    # 4. Audit source-normalization collisions against actual QA traversals
    # ================================================================

    normalization_collision_audit = pd.DataFrame(
        columns=[
            "Type",
            "Canonical",
            "Raw-Values",
            "Used-In-QA-Traversal",
            "Num-Affected-Families",
            "Affected-Family-IDs",
            "Example-Questions",
            "Example-Sources",
        ]
    )
    normalization_collision_audit_path: Optional[str] = None

    if "source" in active_regimes:
        normalization_collision_audit_path = os.path.join(
            base_question_path,
            f"{dataset.lower()}_source_normalization_collisions.csv",
        )
        normalization_collision_audit = build_normalization_collision_audit(
            qa_df=qa_df,
            adjacency=graph_adjacencies["source"],
            entity_collisions=source_entity_collisions,
            relation_collisions=source_relation_collisions,
        )
        enforce_safe_normalization_collisions(
            normalization_collision_audit,
            normalization_collision_audit_path,
        )

    # ================================================================
    # 5. Validate family semantics in every active graph condition
    # ================================================================

    for regime in active_regimes:
        LOGGER.info(
            "Validating family Path-keys against %s graph...",
            regime,
        )
        validate_family_relation_chains(
            qa_df,
            adjacency=graph_adjacencies[regime],
        )

    # ================================================================
    # 6. Deterministic family-level train/dev/test split
    # ================================================================

    qa_df = split_by_hop(
        qa_df,
        split_ratio=args.split_ratio,
        random_seed=args.random_seed,
    )

    if qa_df["Question-Family-ID"].duplicated().any():
        raise AssertionError(
            "A structural family appears more than once after family-level split."
        )

    family_to_split = dict(
        zip(
            qa_df["Question-Family-ID"],
            qa_df["SplitLabel"],
        )
    )
    membership_df["SplitLabel"] = membership_df[
        "Question-Family-ID"
    ].map(family_to_split)

    # Families control split assignment. Evaluation rows then merge compatible
    # released annotations so all valid answers/paths for one textual QA problem
    # are exposed together: PQ merges exact Question+Source+Path-Key+Hops groups;
    # PQL additionally merges alternate Path-keys for exact Question+Source+Hops.
    evaluation_instance_df = build_evaluation_instances(
        family_df=qa_df,
        membership_df=membership_df,
        dataset=dataset,
    )
    LOGGER.info(
        "Materialized %s evaluation instances across %s structural families.",
        len(evaluation_instance_df),
        len(qa_df),
    )

    # Preserve an explicit release-space relation-chain -> designated-answer
    # provenance table for every final merged QA instance. This is especially
    # important for the PQL instances with multiple valid Path-keys.
    path_answer_provenance_rows = []
    for evaluation_instance_number, row in evaluation_instance_df.iterrows():
        for pair in row["Released-Row-Path-Answer-Pairs"]:
            path_answer_provenance_rows.append(
                {
                    "Evaluation-Instance-Number": int(evaluation_instance_number),
                    "Question-Family-ID": row["Question-Family-ID"],
                    "Question": row["Question"],
                    "Source-Entity": row["Source-Entity"],
                    "Released-Row-Number": int(pair["Released-Row-Number"]),
                    "Path-Key": pair["Path-Key"],
                    "Designated-Answer": pair["Designated-Answer"],
                    "Annotated-Path": pair["Annotated-Path"],
                    "Annotated-Entity-Sequence": pair["Annotated-Entity-Sequence"],
                    "Hops": int(row["Hops"]),
                    "SplitLabel": row["SplitLabel"],
                }
            )
    path_answer_provenance_df = pd.DataFrame(path_answer_provenance_rows)
    path_answer_provenance_path = os.path.join(
        base_question_path,
        f"{dataset.lower()}_released_path_answer_provenance.csv",
    )
    path_answer_provenance_df.to_csv(
        path_answer_provenance_path,
        index=False,
    )

    evaluation_reference_rows = []
    for evaluation_instance_number, row in evaluation_instance_df.iterrows():
        for reference_index, pair in enumerate(row["Released-Path-Answer-Pairs"]):
            evaluation_reference_rows.append(
                {
                    "Evaluation-Instance-Number": int(evaluation_instance_number),
                    "Question-Family-ID": row["Question-Family-ID"],
                    "Question": row["Question"],
                    "Source-Entity": row["Source-Entity"],
                    "Reference-Index": int(reference_index),
                    "Released-Row-Numbers": list(pair["Released-Row-Numbers"]),
                    "Path-Key": pair["Path-Key"],
                    "Designated-Answer": pair["Designated-Answer"],
                    "Annotated-Path": [
                        list(triplet) for triplet in pair["Annotated-Path"]
                    ],
                    "Annotated-Entity-Sequence": list(pair["Annotated-Entity-Sequence"]),
                    "Hops": int(row["Hops"]),
                    "SplitLabel": row["SplitLabel"],
                }
            )
    evaluation_reference_df = pd.DataFrame(evaluation_reference_rows)
    evaluation_reference_path = os.path.join(
        base_question_path,
        f"{dataset.lower()}_evaluation_reference_paths.csv",
    )
    evaluation_reference_df.to_csv(evaluation_reference_path, index=False)
    # Hard-validate the final merged QA unit in every active graph. The released
    # targets are fixed across regimes; graph expansion is audited separately.
    for regime in active_regimes:
        validate_evaluation_instance_relation_chains(
            evaluation_instance_df,
            adjacency=graph_adjacencies[regime],
            regime=regime,
        )

    # Construct graph-specific endpoint audits under the released unrestricted
    # walk semantics. Answer/Answer-Entity remain released benchmark targets;
    # Graph-Answer/Graph-Answer-Entity hold expanded endpoints. Entity-simple
    # traversal remains a cycle-dependence diagnostic only.
    qa_by_regime: Dict[str, pd.DataFrame] = {}
    for regime in active_regimes:
        qa_by_regime[regime] = expand_answers_from_graph(
            evaluation_instance_df,
            adjacency=graph_adjacencies[regime],
            regime=regime,
        )

        alignment_errors = qa_by_regime[regime][
            ~qa_by_regime[regime].apply(
                lambda row: (
                    list(row["Answer"]) == list(row["Answer-Entity"])
                    and list(row["Answer"]) == list(row["Released-Answer"])
                    and list(row["Released-Answer"])
                    == list(row["Released-Answer-Entity"])
                    and list(row["Graph-Answer"])
                    == list(row["Graph-Answer-Entity"])
                ),
                axis=1,
            )
        ]
        if len(alignment_errors) > 0:
            raise ValueError(
                f"{len(alignment_errors)} {regime} QA instances have misaligned "
                "released or graph-answer fields before ID encoding."
            )

    filtered_to_source_stats = None

    if requested_graph_regime == "both":
        filtered_qa = qa_by_regime["filtered"].reset_index(drop=True)
        source_qa = qa_by_regime["source"].reset_index(drop=True)

        identity_columns = [
            "Question-Family-ID",
            "Question",
            "Source-Entity",
            "Hops",
        ]
        if not filtered_qa[identity_columns].equals(
            source_qa[identity_columns]
        ):
            raise AssertionError(
                "Filtered/source QA rows are not aligned for direct "
                "expansion auditing."
            )

        new_answers_by_instance: List[Set[str]] = []
        family_new_answers: Dict[str, Set[str]] = defaultdict(set)

        for (_, filtered_row), (_, source_row) in zip(
            filtered_qa.iterrows(),
            source_qa.iterrows(),
        ):
            filtered_answers = set(filtered_row["Graph-Answer"])
            source_answers = set(source_row["Graph-Answer"])

            lost_answers = filtered_answers - source_answers
            if lost_answers:
                raise AssertionError(
                    "Source graph lost filtered graph answers for "
                    f"{filtered_row['Question']!r}: "
                    f"{sorted(lost_answers)!r}"
                )

            added_answers = source_answers - filtered_answers
            new_answers_by_instance.append(added_answers)

            family_new_answers[
                str(filtered_row["Question-Family-ID"])
            ].update(added_answers)

        path_deltas = (
            source_qa["Graph-Path-Count"].astype(int)
            - filtered_qa["Graph-Path-Count"].astype(int)
        )

        if (path_deltas < 0).any():
            raise AssertionError(
                "Source graph has fewer relation-chain-consistent paths "
                "than the filtered graph for at least one QA instance."
            )

        path_growth_mask = path_deltas > 0

        filtered_to_source_stats = {
            "evaluation_instances_with_added_answers": int(
                sum(bool(x) for x in new_answers_by_instance)
            ),
            "evaluation_instance_weighted_added_answers": int(
                sum(len(x) for x in new_answers_by_instance)
            ),
            "families_with_added_answers": int(
                sum(bool(x) for x in family_new_answers.values())
            ),
            "family_level_added_answer_pairs": int(
                sum(len(x) for x in family_new_answers.values())
            ),
            "evaluation_instances_with_added_paths": int(
                path_growth_mask.sum()
            ),
            "evaluation_instance_weighted_added_paths": int(
                path_deltas.sum()
            ),
            "families_with_added_paths": int(
                filtered_qa.loc[
                    path_growth_mask,
                    "Question-Family-ID",
                ].nunique()
            ),
            "max_added_paths_per_instance": int(path_deltas.max()),
        }

        LOGGER.info(
            "Filtered->source direct answer expansion: "
            "%s/%s evaluation instances and %s/%s families gained answers; "
            "%s instance-weighted added answers; "
            "%s added family-answer pairs.",
            filtered_to_source_stats[
                "evaluation_instances_with_added_answers"
            ],
            len(filtered_qa),
            filtered_to_source_stats["families_with_added_answers"],
            filtered_qa["Question-Family-ID"].nunique(),
            filtered_to_source_stats[
                "evaluation_instance_weighted_added_answers"
            ],
            filtered_to_source_stats[
                "family_level_added_answer_pairs"
            ],
        )

        LOGGER.info(
            "Filtered->source direct path expansion: "
            "%s/%s evaluation instances and %s/%s families gained paths; "
            "%s instance-weighted added paths; "
            "max added to one instance=%s.",
            filtered_to_source_stats[
                "evaluation_instances_with_added_paths"
            ],
            len(filtered_qa),
            filtered_to_source_stats["families_with_added_paths"],
            filtered_qa["Question-Family-ID"].nunique(),
            filtered_to_source_stats[
                "evaluation_instance_weighted_added_paths"
            ],
            filtered_to_source_stats["max_added_paths_per_instance"],
        )

    # ================================================================
    # 7. Build one encoding universe for all active graph conditions
    # ================================================================

    if requested_graph_regime == "both":
        encoding_graph = pd.concat(
            [graph_frames["filtered"], graph_frames["source"]],
            ignore_index=True,
        ).drop_duplicates()
        LOGGER.info(
            "Building one shared encoding from the union of filtered + "
            "normalized source/full graphs."
        )
    else:
        encoding_graph = graph_frames[active_regimes[0]]

    entities = set(encoding_graph["head"]).union(set(encoding_graph["tail"]))
    relations = set(encoding_graph["relation"])

    LOGGER.info("Encoding-universe entities: %s", len(entities))
    LOGGER.info("Encoding-universe relations: %s", len(relations))

    qa_encoded_by_regime: Dict[str, pd.DataFrame] = {
        regime: frame.copy(deep=True)
        for regime, frame in qa_by_regime.items()
    }
    encoded_graphs: Dict[str, pd.DataFrame] = {}
    entity_df = None
    relation_df = None

    if args.encode_ids:
        entity2id = {
            entity: f"e{idx}"
            for idx, entity in enumerate(sorted(entities))
        }
        relation2id = {
            relation: f"r{idx}"
            for idx, relation in enumerate(sorted(relations))
        }

        for regime in active_regimes:
            qa_encoded = qa_encoded_by_regime[regime]

            missing_sources = set(qa_encoded["Source-Entity"]) - set(entity2id)
            missing_answers = {
                answer
                for column in (
                    "Answer-Entity",
                    "Released-Answer-Entity",
                    "Graph-Answer-Entity",
                )
                for answers in qa_encoded[column]
                for answer in answers
                if answer not in entity2id
            }
            missing_relations = {
                relation
                for path_keys in qa_encoded["Path-keys"]
                for path_key in path_keys_to_list(path_keys)
                for relation in path_key.split("->")
                if relation not in relation2id
            }
            missing_path_entities = {
                entity
                for column in ("Multi-Paths-Label", "Graph-Multi-Paths-Label")
                for path_groups in qa_encoded[column]
                for path_group in path_groups
                for path in path_group
                for head, _, tail in path
                for entity in (head, tail)
                if entity not in entity2id
            }
            missing_path_relations = {
                relation
                for column in ("Multi-Paths-Label", "Graph-Multi-Paths-Label")
                for path_groups in qa_encoded[column]
                for path_group in path_groups
                for path in path_group
                for _, relation, _ in path
                if relation not in relation2id
            }
            if (
                missing_sources
                or missing_answers
                or missing_relations
                or missing_path_entities
                or missing_path_relations
            ):
                raise ValueError(
                    f"{regime} QA identifiers are absent from the shared encoding "
                    "vocabulary. "
                    f"Missing sources={list(missing_sources)[:10]}, "
                    f"answers={list(missing_answers)[:10]}, "
                    f"relations={list(missing_relations)[:10]}, "
                    f"path_entities={list(missing_path_entities)[:10]}, "
                    f"path_relations={list(missing_path_relations)[:10]}"
                )

            qa_encoded["Source-Entity"] = qa_encoded["Source-Entity"].map(
                entity2id
            )
            qa_encoded["Answer-Entity"] = qa_encoded["Answer-Entity"].apply(
                lambda answers: [entity2id[answer] for answer in answers]
            )
            qa_encoded["Released-Answer-Entity"] = qa_encoded[
                "Released-Answer-Entity"
            ].apply(
                lambda answers: [entity2id[answer] for answer in answers]
            )
            qa_encoded["Graph-Answer-Entity"] = qa_encoded[
                "Graph-Answer-Entity"
            ].apply(
                lambda answers: [entity2id[answer] for answer in answers]
            )
            qa_encoded["Path-keys"] = qa_encoded["Path-keys"].apply(
                lambda path_keys: encode_path_keys(path_keys, relation2id)
            )
            qa_encoded["Multi-Paths"] = qa_encoded["Multi-Paths-Label"].apply(
                lambda path_groups: encode_multi_paths(
                    path_groups, entity2id, relation2id
                )
            )
            qa_encoded["Graph-Multi-Paths"] = qa_encoded[
                "Graph-Multi-Paths-Label"
            ].apply(
                lambda path_groups: encode_multi_paths(
                    path_groups, entity2id, relation2id
                )
            )
            qa_encoded["Multi-Paths-Key"] = qa_encoded["Multi-Paths-Key"].apply(
                lambda path_keys: encode_multi_path_keys(path_keys, relation2id)
            )

            encoded_alignment_errors = qa_encoded[
                ~qa_encoded.apply(
                    lambda row: (
                        row["Answer-Entity"]
                        == [entity2id[label] for label in row["Answer"]]
                        and row["Released-Answer-Entity"]
                        == [
                            entity2id[label]
                            for label in row["Released-Answer"]
                        ]
                        and row["Graph-Answer-Entity"]
                        == [
                            entity2id[label]
                            for label in row["Graph-Answer"]
                        ]
                        and row["Answer-Entity"]
                        == row["Released-Answer-Entity"]
                    ),
                    axis=1,
                )
            ]
            if len(encoded_alignment_errors) > 0:
                raise ValueError(
                    f"{len(encoded_alignment_errors)} {regime} QA instances have "
                    "misaligned label and encoded answer fields."
                )

            qa_encoded_by_regime[regime] = qa_encoded

        for regime in active_regimes:
            encoded_graphs[regime] = encode_graph(
                graph_frames[regime],
                entity2id,
                relation2id,
            )

        entity_df = pd.DataFrame(
            list(entity2id.items()),
            columns=["Title", "EID"],
        )
        relation_df = pd.DataFrame(
            list(relation2id.items()),
            columns=["Title", "RID"],
        )

        LOGGER.info(
            "✅ Entities and relations have been encoded once and reused for "
            "all active graph regimes."
        )
    else:
        for regime in active_regimes:
            encoded_graphs[regime] = graph_frames[regime]

    # ================================================================
    # 8. Save family audit plus merged multi-answer/multi-path QA outputs
    # ================================================================

    family_output = os.path.join(
        base_question_path,
        f"{dataset.lower()}_question_families.csv",
    )
    qa_df.to_csv(family_output, index=False)

    membership_output = os.path.join(
        base_question_path,
        f"{dataset.lower()}_question_family_membership.csv",
    )
    membership_df.to_csv(membership_output, index=False)

    # v16 keeps qa_nhop intentionally compact.  All diagnostic/provenance fields
    # not listed here are exported to a companion *_qa_*hop_metadata.csv file.
    output_columns = [
        "Question-Number",
        "Question",
        "Question-Paraphrased",
        "Question-Family-ID",
        "Question-Family-Size",
        "Source",
        "Source-Entity",
        "Answer",
        "Answer-Entity",
        "Graph-Answer",
        "Graph-Answer-Entity",
        "Multi-Paths",
        "Multi-Paths-Label",
        "Multi-Paths-Key",
        "Path-Count",
        "Graph-Multi-Paths",
        "Graph-Multi-Paths-Label",
        "Graph-Path-Count",
        "Reference-ID",
        "Hops",
        "SplitLabel",
    ]


    for regime in active_regimes:
        qa_encoded = qa_encoded_by_regime[regime].copy()
        qa_encoded.insert(0, "Question-Number", range(len(qa_encoded)))
        validate_multi_path_export_fields(qa_encoded)
        qa_output = qa_encoded[output_columns].copy()

        # The companion metadata file contains every field omitted from qa_nhop,
        # plus stable join columns copied from the compact QA table.
        metadata_join_columns = [
            "Question-Number",
            "Question-Family-ID",
            "Reference-ID",
            "Hops",
            "SplitLabel",
        ]
        additional_columns = [
            column for column in qa_encoded.columns
            if column not in output_columns
        ]
        qa_metadata_columns = list(dict.fromkeys(
            metadata_join_columns + additional_columns
        ))
        qa_metadata_output = qa_encoded[qa_metadata_columns].copy()

        LOGGER.info(
            "Final %s QA columns (v16 compact schema): %s",
            regime,
            qa_output.columns.tolist(),
        )
        LOGGER.info(
            "Final %s companion QA metadata columns: %s",
            regime,
            qa_metadata_output.columns.tolist(),
        )

        if requested_graph_regime == "both":
            triplet_output_dir = os.path.join(base_triplet_path, regime)
            question_output_dir = os.path.join(base_question_path, regime)
        else:
            # Preserve v4's default directory layout for one-regime runs.
            triplet_output_dir = base_triplet_path
            question_output_dir = base_question_path

        os.makedirs(triplet_output_dir, exist_ok=True)
        os.makedirs(question_output_dir, exist_ok=True)

        graph_output = os.path.join(triplet_output_dir, "triplets.txt")
        save_triplets(encoded_graphs[regime], graph_output)
        mixed_output, mixed_metadata_output = save_qa_outputs(
            qa_output,
            qa_metadata_output,
            question_output_dir,
            dataset,
        )

        cycle_answer_audit_output = os.path.join(
            question_output_dir,
            f"{dataset.lower()}_cycle_answer_audit.csv",
        )
        qa_by_regime[regime][
            [
                "Question-Family-ID",
                "Question",
                "Released-Row-Count",
                "Released-Row-Numbers",
                "Released-Designated-Answers",
                "Released-Path-Answer-Pairs",
                "Source-Entity",
                "Released-Answer",
                "Topic-In-Released-Answers",
                "Answer",
                "Graph-Answer",
                "Topic-In-Graph-Answers",
                "Added-Graph-Answers",
                "Simple-Path-Answers",
                "Cycle-Only-Answers",
                "Released-Answers-Only-Cyclic",
                "Released-Answers-Not-Reachable",
                "Variant-Has-Repeated-Entity-Path",
                "Variant-Has-Immediate-Backtrack",
                "Path-keys",
                "Hops",
                "SplitLabel",
            ]
        ].to_csv(cycle_answer_audit_output, index=False)

        if args.encode_ids:
            # In --graph-regime both, intentionally write the same shared mapping
            # to both regime directories so node/relation IDs are identical.
            entity_df.to_csv(
                os.path.join(triplet_output_dir, "node_data.csv"),
                index=False,
            )
            relation_df.to_csv(
                os.path.join(triplet_output_dir, "relation_data.csv"),
                index=False,
            )

        metadata = {
            "dataset": dataset,
            "graph_regime": regime,
            "requested_graph_regime": args.graph_regime,
            "split_ratio": args.split_ratio,
            "random_seed": args.random_seed,
            "encode_ids": args.encode_ids,
            "shared_encoding_across_regimes": (
                bool(args.encode_ids) and requested_graph_regime == "both"
            ),
            "source_kg_file": source_kg_file,
            "active_unique_triples": len(graph_frames[regime]),
            "encoding_entities": len(entities),
            "encoding_relations": len(relations),
            "released_rows": len(released_qa_df),
            "question_families": len(qa_df),
            "evaluation_instances": len(qa_output),
            "qa_output": mixed_output,
            "qa_metadata_output": mixed_metadata_output,
            "qa_output_columns": output_columns,
            "multi_paths_semantics": (
                "List[PathGroup], aligned with unique Multi-Paths-Key values; "
                "PathGroup=List[Path], Path=List[[head, relation, tail]]"
            ),
            "graph_multi_paths_semantics": (
                "All unrestricted active-graph realizations grouped by the same "
                "unique Multi-Paths-Key order as annotated Multi-Paths"
            ),
            "multi_paths_key_semantics": (
                "Unique List[str]; position i identifies relation-chain group i in "
                "Multi-Paths and Graph-Multi-Paths"
            ),
            "reference_id_semantics": (
                "List[List[List[int]]]: key group -> annotated path -> original "
                "released row numbers supporting that exact path"
            ),
            "path_count_semantics": (
                "Total distinct released annotated paths across all key groups"
            ),
            "graph_path_count_semantics": (
                "Total distinct unrestricted graph-realized paths across all key groups"
            ),
            "evaluation_instances_with_multiple_relation_chains": int(
                (qa_by_regime[regime]["Num-Relation-Chains"] > 1).sum()
            ),
            # Backward-compatible alias retained from v10.
            "question_variants": len(qa_output),
            "released_rows_collapsed_into_evaluation_instances": (
                len(released_qa_df) - len(qa_output)
            ),
            "evaluation_instances_by_hop": {
                str(int(hops)): int(count)
                for hops, count in qa_output["Hops"].value_counts().sort_index().items()
            },
            "annotated_path_validity_audit": annotated_path_audit_path,
            "annotated_paths_with_repeated_entities": int(
                (~annotated_path_audit["Annotated-Path-Simple"]).sum()
            ),
            "annotated_paths_with_immediate_backtracks": int(
                annotated_path_audit["Annotated-Path-Backtracks"].sum()
            ),
            "released_rows_where_topic_is_designated_answer": int(
                annotated_path_audit["Topic-Is-Designated-Answer"].sum()
            ),
            "released_rows_where_topic_is_in_released_answer_set": int(
                annotated_path_audit["Topic-In-Released-Answers"].sum()
            ),
            "cycle_answer_audit": cycle_answer_audit_output,
            "released_path_answer_provenance": path_answer_provenance_path,
            "evaluation_reference_paths": evaluation_reference_path,
            "released_designated_answer_semantics": (
                "union_over_rows_sharing_source_hops_and_retained_path_keys"
            ),
            "released_row_designated_answer_semantics": (
                "exact_designated_endpoints_of_source_rows_for_this_textual_instance"
            ),
            "primary_answer_semantics": "merged_released_answers",
            "graph_answer_semantics": (
                "unrestricted_endpoints_of_retained_released_relation_chains"
            ),
            "final_merged_instance_reachability_is_fatal": True,
            "family_weight_semantics": "1_over_family_instance_count",
            "recommended_primary_aggregation": "family_macro",
            "filtered_to_source_direct_expansion": (
                filtered_to_source_stats
                if requested_graph_regime == "both"
                else None
            ),
            "same_question_path_ambiguity_groups": len(ambiguity_audit),
            "same_question_path_different_source_groups": source_ambiguity_count,
            "same_question_path_different_answer_set_groups": answer_ambiguity_count,
            "source_normalization_collision_audit": (
                normalization_collision_audit_path
                if regime == "source"
                else None
            ),
            "source_normalization_entity_collisions": (
                int((normalization_collision_audit["Type"] == "entity").sum())
                if regime == "source"
                else 0
            ),
            "source_normalization_base_entity_collision_groups": (
                source_normalization_stats[
                    "base_entity_collision_groups"
                ]
                if regime == "source"
                else 0
            ),
            "source_normalization_resolved_entity_collision_groups": (
                source_normalization_stats[
                    "resolved_entity_collision_groups"
                ]
                if regime == "source"
                else 0
            ),
            "source_normalization_unresolved_entity_collision_groups": (
                source_normalization_stats[
                    "unresolved_entity_collision_groups"
                ]
                if regime == "source"
                else 0
            ),
            "source_normalization_unresolved_relation_collision_groups": (
                source_normalization_stats[
                    "unresolved_relation_collision_groups"
                ]
                if regime == "source"
                else 0
            ),

            # Backward-compatible aliases; unresolved collisions only.
            "source_normalization_relation_collisions": (
                int((normalization_collision_audit["Type"] == "relation").sum())
                if regime == "source"
                else 0
            ),
            "source_normalization_collisions_used_in_qa": (
                int(normalization_collision_audit["Used-In-QA-Traversal"].sum())
                if regime == "source" and len(normalization_collision_audit)
                else 0
            ),
            "evaluation_instances_with_graph_added_answers": int(
                qa_by_regime[regime]["Added-Graph-Answers"].apply(bool).sum()
            ),
            # Backward-compatible alias retained from v10.
            "variants_with_graph_added_answers": int(
                qa_by_regime[regime]["Added-Graph-Answers"].apply(bool).sum()
            ),
            "evaluation_instance_weighted_total_added_answers": int(
                qa_by_regime[regime]["Added-Graph-Answers"].apply(len).sum()
            ),
            "evaluation_instance_max_added_answers": int(
                qa_by_regime[regime]["Added-Graph-Answers"].apply(len).max()
            ),
            "families_with_graph_added_answers": int(
                qa_by_regime[regime].loc[
                    qa_by_regime[regime]["Added-Graph-Answers"].apply(bool),
                    "Question-Family-ID",
                ].nunique()
            ),
            "family_level_total_added_answers": int(
                sum(
                    len(
                        set().union(
                            *(set(values) for values in group["Added-Graph-Answers"])
                        )
                    )
                    for _, group in qa_by_regime[regime].groupby(
                        "Question-Family-ID", sort=False
                    )
                )
            ),
            "family_level_max_added_answers": int(
                max(
                    (
                        len(
                            set().union(
                                *(set(values) for values in group["Added-Graph-Answers"])
                            )
                        )
                        for _, group in qa_by_regime[regime].groupby(
                            "Question-Family-ID", sort=False
                        )
                    ),
                    default=0,
                )
            ),
            "variants_with_cycle_only_answers": int(
                qa_by_regime[regime]["Cycle-Only-Answers"].apply(bool).sum()
            ),
            "variants_with_cycle_dependent_released_answers": int(
                qa_by_regime[regime][
                    "Released-Answers-Only-Cyclic"
                ].apply(bool).sum()
            ),
            "variants_with_unreachable_released_answers": int(
                qa_by_regime[regime][
                    "Released-Answers-Not-Reachable"
                ].apply(bool).sum()
            ),
            "variants_where_topic_is_released_answer": int(
                qa_by_regime[regime]["Topic-In-Released-Answers"].sum()
            ),
            "variants_where_topic_is_graph_answer": int(
                qa_by_regime[regime]["Topic-In-Graph-Answers"].sum()
            ),
        }
        with open(
            os.path.join(question_output_dir, "conversion_metadata.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)

        LOGGER.info(
            "✅ Saved %s graph to: %s",
            regime,
            graph_output,
        )
        LOGGER.info(
            "✅ Saved %s compact evaluation-instance QA to: %s",
            regime,
            mixed_output,
        )        LOGGER.info(
            "✅ Saved %s companion QA metadata to: %s",
            regime,
            mixed_metadata_output,
        )
        LOGGER.info(
            "✅ Saved %s cycle/answer audit to: %s",
            regime,
            cycle_answer_audit_output,
        )
        if args.encode_ids:
            LOGGER.info(
                "✅ Saved shared node/relation mappings in %s",
                triplet_output_dir,
            )

    LOGGER.info(
        "✅ Saved structural family table to: %s",
        family_output,
    )
    LOGGER.info(
        "✅ Saved released-row family membership audit to: %s",
        membership_output,
    )
    LOGGER.info(
        "✅ Saved released Path-key -> designated-answer provenance to: %s",
        path_answer_provenance_path,
    )
    if normalization_collision_audit_path is not None:
        LOGGER.info(
            "✅ Saved source-normalization collision audit to: %s",
            normalization_collision_audit_path,
        )
    LOGGER.info("✅ Conversion log saved to: %s", log_path)