"""Retrieval invariants: bounded joins, aligned samples, labels, and data isolation."""
import json
from copy import deepcopy

import pytest

from core.schema_retriever import BM25Index
from core.schema_selector import SchemaSelector
from evaluation.evaluate_schema_retrieval import evaluate
from evaluation.schema_retrieval import parse_schema_ddl, reference_schema
from scripts.import_bird import import_bird


def table(*names, fk=()):
    return {"columns": [{"name": n, "type": "TEXT", "pk": n == "id"} for n in names],
            "foreign_keys": list(fk), "samples": [[f"value_{n}" for n in names]]}


def test_bm25_evidence_and_index_reuse():
    schema = {"alpha": table("id", "alpha_name"), "beta": table("id", "rainfall")}
    selector = SchemaSelector(top_k_tables=1, top_k_columns=1)
    selected = selector.select("measurement", schema, "rainfall")
    assert list(selected) == ["beta"]
    assert {c['name'] for c in selected['beta']['columns']} == {"id", "rainfall"}
    assert not selector.last_metadata["index_cache_hit"]
    selector.select("measurement", schema, "rainfall")
    assert selector.last_metadata["index_cache_hit"]
    assert BM25Index(["alpha", "beta"]).scores("unknown") == [0, 0]


def test_projection_preserves_samples_and_does_not_mutate():
    schema = {"weather": table("id", "rainfall", "unrelated", "unused")}
    before = deepcopy(schema)
    selector = SchemaSelector(top_k_columns=2)
    selected = selector.select("rainfall", schema)
    columns = [c["name"] for c in selected["weather"]["columns"]]
    assert columns == ["id", "rainfall"]
    assert selected["weather"]["samples"] == [["value_id", "value_rainfall"]]
    assert schema == before


def test_shortest_bridge_is_bounded_and_keeps_join_columns():
    schema = {
        "alpha": table("id", "a_value"),
        "bridge": table("id", "a_id", "b_id", fk=[
            {"column": "a_id", "table": "ALPHA", "target": "ID"},
            {"column": "b_id", "table": "beta", "target": None}]),
        "beta": table("id", "b_value"),
        "unrelated": table("id", "a_id", fk=[{"column": "a_id", "table": "alpha", "target": "id"}]),
    }
    selector = SchemaSelector(top_k_tables=2, top_k_columns=1, max_bridge_tables=1)
    selector.retriever.rank = lambda q, s: ([(("alpha", None), 10), (("beta", None), 9)], {})
    selected = selector.select("connect", schema)
    assert set(selected) == {"alpha", "bridge", "beta"}
    assert {c["name"] for c in selected["bridge"]["columns"]} == {"id", "a_id", "b_id"}
    assert selected["bridge"]["foreign_keys"][0]["table"] == "alpha"
    assert selector.last_metadata["bridge_tables"] == ["bridge"]
    assert selector.last_metadata["mandatory_columns_over_budget"] == 2
    selector = SchemaSelector(top_k_tables=2, max_bridge_tables=0)
    selector.retriever.rank = lambda q, s: ([(("alpha", None), 10), (("beta", None), 9)], {})
    assert set(selector.select("connect", schema)) == {"alpha", "beta"}
    assert selector.last_metadata["disconnected_pairs"]


def test_no_match_fallback_and_bad_budgets():
    schema = {"weather": table("id", "rainfall")}
    selector = SchemaSelector(top_k_columns=1)
    assert selector.select("zzzzzz", schema) == schema
    assert selector.last_metadata["fallback_reason"] == "no_retrieval_match"
    with pytest.raises(ValueError):
        SchemaSelector(top_k_columns=0)


DDL = 'CREATE TABLE t (id INTEGER PRIMARY KEY, parent INTEGER, value TEXT, other TEXT);'


def test_reference_labels_handle_cte_union_star_and_correlated_subqueries():
    schema = parse_schema_ddl(DDL)
    refs = reference_schema('WITH x AS (SELECT id, value FROM t) SELECT value FROM x', schema)
    assert refs == ({"t"}, {("t", "id"), ("t", "value")})
    tables, columns = reference_schema('SELECT COUNT(*) FROM t', schema)
    assert tables == {"t"} and columns == set()
    assert len(reference_schema('SELECT * FROM t', schema)[1]) == 4
    assert reference_schema('SELECT id AS email FROM t UNION SELECT parent AS email FROM t ORDER BY email', schema)[1] == {("t", "id"), ("t", "parent")}
    assert reference_schema('SELECT a.id FROM t a WHERE EXISTS (SELECT 1 FROM t b WHERE b.parent=a.id)', schema)[1] == {("t", "id"), ("t", "parent")}
    with pytest.raises(Exception):
        reference_schema('SELECT missing FROM t', schema)


def test_evaluator_gold_is_not_retrieval_input(tmp_path, monkeypatch):
    questions = tmp_path / "questions.jsonl"
    questions.write_text(json.dumps({"question_id": 1, "db_id": "db", "question": "value", "evidence": "", "SQL": "SELECT id FROM t"}) + "\n")
    schemas = tmp_path / "schemas.json"
    schemas.write_text(json.dumps({"db": DDL}))
    queries = []
    original = SchemaSelector.select
    def observe(self, question, schema, evidence=""):
        queries.append((question, evidence))
        return original(self, question, schema, evidence)
    monkeypatch.setattr(SchemaSelector, "select", observe)
    report = evaluate(questions, schemas, tmp_path / "out")
    assert queries == [("value", "")]
    assert report["summary"]["questions"] == 1
    assert report["summary"]["label_errors"] == 0
    assert report["token_counter"] == "regex_proxy_not_model_tokens"
    assert (tmp_path / "out" / "details.jsonl").exists()


def test_importer_jsonl_routes_each_database_and_rejects_duplicate_ids(tmp_path):
    root = tmp_path / "dbs"
    for name in ("one", "two"):
        (root / name).mkdir(parents=True)
        (root / name / f"{name}.sqlite").touch()
    rows = [{"question_id": i, "db_id": name, "question": "x", "SQL": "SELECT 1"}
            for i, name in enumerate(("one", "two"))]
    source = tmp_path / "mini.jsonl"
    source.write_text("\n".join(json.dumps(r) for r in rows))
    result = import_bird(source, root, tmp_path / "converted.json")
    assert result[0]["db_path"] != result[1]["db_path"]
    assert result[0]["id"] == "bird_0"
    rows[1]["question_id"] = 0
    source.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="Duplicate"):
        import_bird(source, root, tmp_path / "converted.json")


def test_pipeline_prunes_initial_and_correction_prompts(config, db):
    from core.pipeline import Pipeline
    from core.types import Candidate
    class BrokenBackend:
        name = "broken_test"
        def __init__(self):
            self.prompts = []
        def start_request(self, seed):
            pass
        def generate(self, prompt, n, parameters, deadline=None):
            self.prompts.append(prompt)
            sql = "SELECT missing FROM students" if len(self.prompts) == 1 else "SELECT name FROM students"
            return [Candidate(0, raw_text=f"<answer>{sql}</answer>")]
    config['sampling']['mode'] = 'greedy'
    config['validation']['schema_selection'] = True
    config['retrieval'].update(top_k_tables=1, top_k_columns=2, max_bridge_tables=0)
    backend = BrokenBackend()
    result = Pipeline(config, backend).run(db, "students name")
    assert len(backend.prompts) == 2
    for prompt in backend.prompts:
        assert 'CREATE TABLE "students"' in prompt
        assert 'CREATE TABLE "subjects"' not in prompt
    assert result.public_dict()['schema_retrieval']['selected_tables'] == 1


def test_hybrid_dense_cache_and_fingerprint(tmp_path):
    np = pytest.importorskip('numpy')
    from core.schema_retriever import SchemaRetriever
    class Encoder:
        def __init__(self):
            self.batch_sizes = []
        def encode(self, texts, **kwargs):
            self.batch_sizes.append(len(texts))
            vectors = np.array([[1.0, 0.0] if 'rainfall' in text else [0.0, 1.0] for text in texts])
            return vectors
    schema = {"weather": table("id", "rainfall")}
    retriever = SchemaRetriever(mode='hybrid', cache_dir=str(tmp_path))
    encoder = Encoder()
    retriever._model = encoder
    ranking, first = retriever.rank('rainfall', schema)
    assert ranking[0][1] > 0
    retriever.rank('rainfall', schema)
    assert encoder.batch_sizes == [3, 1, 1]
    reloaded = SchemaRetriever(mode='hybrid', cache_dir=str(tmp_path))
    reloaded._model = Encoder()
    reloaded.rank('rainfall', schema)
    assert reloaded._model.batch_sizes == [1]
    schema['weather']['description'] = 'changed metadata'
    _, changed = retriever.rank('rainfall', schema)
    assert changed['index_fingerprint'] != first['index_fingerprint']


def test_export_preserves_order_and_failed_questions(tmp_path):
    from scripts.export_bird_predictions import export
    items = [{"id": f"bird_{i}", "db_id": "db", "gold_sql": "SELECT 1", "difficulty": "simple"} for i in (10, 5)]
    rows = [{"id": f"bird_{i}", "db_id": "db", "prediction": {"selected": {"sql": "SELECT 1"} if i == 10 else None}} for i in (5, 10)]
    dataset, predictions = tmp_path / 'dataset.json', tmp_path / 'predictions.jsonl'
    dataset.write_text(json.dumps(items))
    predictions.write_text('\n'.join(json.dumps(r) for r in rows))
    assert export(dataset, predictions, tmp_path / 'export') == 2
    output = json.loads((tmp_path / 'export/predict_mini_dev.json').read_text())
    assert len((tmp_path / "export/mini_dev_difficulty.jsonl").read_text().splitlines()) == 2
    assert output['0'] == 'SELECT 1\t----- bird -----\tdb'
    assert '__bird_missing_prediction__' in output['1']
    predictions.write_text(json.dumps(rows[0]))
    with pytest.raises(ValueError, match='counts differ'):
        export(dataset, predictions, tmp_path / 'export')


def test_evaluate_uses_per_question_database(config, tmp_path):
    import sqlite3
    from core.types import Candidate
    from evaluation.evaluate import evaluate as evaluate_sql
    class PerDatabaseBackend:
        name = 'database_fixture'
        def start_request(self, seed):
            pass
        def generate(self, prompt, n, parameters, deadline=None):
            sql = 'SELECT alpha FROM data' if 'alpha' in prompt else 'SELECT beta FROM data'
            return [Candidate(0, raw_text=f'<answer>{sql}</answer>')]
    records = []
    for index, column in enumerate(('alpha', 'beta')):
        database = tmp_path / f'{index}.sqlite'
        with sqlite3.connect(database) as connection:
            connection.execute(f'CREATE TABLE data ({column} TEXT)')
            connection.execute(f'INSERT INTO data VALUES (\'value_{index}\')')
        records.append({'id': str(index), 'db_id': str(index), 'db_path': database.name,
                        'question': f'list {column}', 'gold_sql': f'SELECT {column} FROM data',
                        'difficulty': 'simple'})
    dataset = tmp_path / 'dataset.json'
    dataset.write_text(json.dumps(records))
    config['sampling']['mode'] = 'greedy'
    summary = evaluate_sql(config, dataset, tmp_path / 'unused.sqlite', tmp_path / 'out', backend=PerDatabaseBackend())
    assert summary['execution_accuracy'] == 1.0
    assert summary['questions'] == 2


def test_export_does_not_rewrite_spaces_inside_literals():
    from scripts.export_bird_predictions import flatten_sql
    assert flatten_sql("SELECT\n 'a  b', \"my  column\"\nFROM t") == "SELECT 'a  b', \"my  column\" FROM t"
    assert flatten_sql("SELECT 'a''  b' FROM\n t") == "SELECT 'a''  b' FROM t"


def test_model_telemetry_counts_eos_and_context_overflow(config):
    torch = pytest.importorskip('torch')
    pytest.importorskip('transformers')
    import threading
    from types import SimpleNamespace
    from transformers import BatchEncoding
    from core.hf_model import HFModelBackend
    class Tokenizer:
        chat_template = None
        pad_token_id = 0
        def __call__(self, text, **kwargs):
            return BatchEncoding({'input_ids': torch.tensor([[10, 11, 12]])})
        def decode(self, tokens, **kwargs):
            return '<answer>SELECT 1</answer>'
    class Model:
        device = torch.device('cpu')
        config = SimpleNamespace(max_position_embeddings=1024)
        generation_config = SimpleNamespace(eos_token_id=2)
        def generate(self, **kwargs):
            return torch.tensor([[10, 11, 12, 20, 2, 0]] * kwargs['num_return_sequences'])
    backend = HFModelBackend.__new__(HFModelBackend)
    backend.torch, backend.lock = torch, threading.RLock()
    backend.model, backend.tokenizer = Model(), Tokenizer()
    backend.start_request(42)
    parameters = dict(config['generation'], batch_size=2, max_input_tokens=100, max_new_tokens=10)
    candidates = backend.generate('prompt', 2, parameters)
    assert len(candidates) == 2
    metrics = backend.request_metrics['calls'][0]
    assert metrics['input_tokens'] == 3
    assert metrics['output_tokens'] == 4  # EOS included; padding excluded.
    assert metrics['sequences'] == 2
    backend.generate('prompt', 1, dict(parameters, max_input_tokens=2))
    assert backend.request_metrics['calls'][-1]['context_overflow']


def test_named_constraints_preserve_primary_and_join_keys():
    schema = parse_schema_ddl('''
        CREATE TABLE parent(id INT, CONSTRAINT pk_parent PRIMARY KEY(id));
        CREATE TABLE child(id INT, pid INT, extra TEXT,
                           CONSTRAINT fk_child FOREIGN KEY(pid) REFERENCES parent(id));
    ''')
    assert schema['parent']['columns'][0]['pk'] == 1
    assert schema['child']['foreign_keys'] == [{'column': 'pid', 'table': 'parent', 'target': 'id'}]
    selected = SchemaSelector(top_k_tables=2, top_k_columns=1).select('parent child', schema)
    assert 'pid' in {c['name'] for c in selected['child']['columns']}


def test_natural_join_does_not_silently_inflate_reference_coverage():
    schema = parse_schema_ddl('CREATE TABLE a(id INT, shared TEXT, value TEXT); CREATE TABLE b(id INT, shared TEXT, other TEXT);')
    with pytest.raises(ValueError, match='NATURAL JOIN'):
        reference_schema('SELECT a.value FROM a NATURAL JOIN b', schema)
