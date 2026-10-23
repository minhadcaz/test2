from core.consistency_checks import ConsistencyChecker


def test_literal_must_exist_in_compared_column(db):
    checker = ConsistencyChecker(db)
    assert checker.check_sql("SELECT id FROM students WHERE gender = 'Female'")["ungrounded"]
    assert not checker.check_sql("SELECT id FROM students WHERE gender = 'Nữ'")["ungrounded"]
    assert not checker.check_sql("SELECT id FROM students s WHERE s.gender IN ('Nam', 'Nữ')")["ungrounded"]


def test_join_on_unrelated_keys_is_flagged(db):
    checker = ConsistencyChecker(db)
    wrong = "SELECT d.name FROM departments d JOIN students s ON d.id = s.class_id"
    right = ("SELECT d.name FROM departments d JOIN classes c ON d.id = c.department_id "
             "JOIN students s ON c.id = s.class_id")
    assert checker.check_sql(wrong)["bad_join"]
    assert not checker.check_sql(right)["bad_join"]


def test_degenerate_results(db):
    assert ConsistencyChecker.check_result([])["degenerate"]
    assert ConsistencyChecker.check_result([[None], [None]])["degenerate"]
    assert not ConsistencyChecker.check_result([[0]])["degenerate"]
