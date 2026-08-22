"""
Memory system tests. All real — SQLite is fast and needs no mocking.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


@pytest.fixture
def memory_db(tmp_path, monkeypatch):
    from backend.config import settings as settings_module
    monkeypatch.setattr(settings_module.settings, "MEMORY_DB_PATH", str(tmp_path / "memory.db"))
    import backend.memory.memory_store as mem
    return mem


class TestShortTermMemory:
    def test_add_and_get_history(self, memory_db):
        stm = memory_db.ShortTermMemory(max_turns=8)
        stm.add_turn("s1", "user", "hello")
        stm.add_turn("s1", "assistant", "hi there")
        history = stm.get_history("s1")
        assert len(history) == 2
        assert history[0]["role"] == "user"

    def test_trims_to_max_turns(self, memory_db):
        stm = memory_db.ShortTermMemory(max_turns=4)
        for i in range(10):
            stm.add_turn("s1", "user", f"message {i}")
        assert len(stm.get_history("s1")) == 4
        # Should keep the most recent, not the oldest
        assert stm.get_history("s1")[-1]["content"] == "message 9"

    def test_sessions_are_isolated(self, memory_db):
        stm = memory_db.ShortTermMemory()
        stm.add_turn("s1", "user", "from session 1")
        stm.add_turn("s2", "user", "from session 2")
        assert len(stm.get_history("s1")) == 1
        assert len(stm.get_history("s2")) == 1
        assert stm.get_history("s1")[0]["content"] != stm.get_history("s2")[0]["content"]

    def test_clear_session(self, memory_db):
        stm = memory_db.ShortTermMemory()
        stm.add_turn("s1", "user", "hello")
        stm.clear("s1")
        assert stm.get_history("s1") == []


class TestLongTermMemory:
    def test_add_and_list(self, memory_db):
        memory_db.add_long_term_memory("User's favorite color is blue.")
        memories = memory_db.list_long_term_memory()
        assert len(memories) == 1
        assert memories[0]["content"] == "User's favorite color is blue."

    def test_delete(self, memory_db):
        mem_id = memory_db.add_long_term_memory("Temporary fact.")
        deleted = memory_db.delete_long_term_memory(mem_id)
        assert deleted is True
        assert memory_db.list_long_term_memory() == []

    def test_delete_nonexistent_returns_false(self, memory_db):
        assert memory_db.delete_long_term_memory(99999) is False

    def test_search_finds_keyword_match(self, memory_db):
        memory_db.add_long_term_memory("User works at Acme Corp as an engineer.")
        memory_db.add_long_term_memory("User enjoys hiking on weekends.")
        results = memory_db.search_long_term_memory("Acme")
        assert len(results) == 1
        assert "Acme" in results[0]["content"]

    def test_search_no_match_returns_empty(self, memory_db):
        memory_db.add_long_term_memory("Some unrelated fact.")
        assert memory_db.search_long_term_memory("nonexistent_keyword_xyz") == []

    def test_not_everything_is_persisted_automatically(self, memory_db):
        # Nothing should exist unless explicitly added — this is the
        # "don't store everything automatically" requirement, verified
        # at the store level (the *decision* of what's memory-worthy
        # lives in the graph's memory_persist node, tested separately).
        assert memory_db.list_long_term_memory() == []


class TestPreferences:
    def test_add_and_list(self, memory_db):
        memory_db.add_preference("Always keep answers under 3 sentences.")
        prefs = memory_db.list_preferences()
        assert len(prefs) == 1

    def test_delete(self, memory_db):
        pref_id = memory_db.add_preference("Call me by my first name.")
        assert memory_db.delete_preference(pref_id) is True
        assert memory_db.list_preferences() == []

    def test_preferences_separate_from_long_term_memory(self, memory_db):
        memory_db.add_preference("Be concise.")
        memory_db.add_long_term_memory("User's birthday is in June.")
        assert len(memory_db.list_preferences()) == 1
        assert len(memory_db.list_long_term_memory()) == 1
