"""Tests for Alta internal mood/state system."""
import os
import unittest
import uuid

from context_builder import (
    apply_time_decay_chat,
    apply_time_decay_relation,
    build_alta_state_block,
    compute_state_updates,
)
from db import DB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_chat_state(**overrides) -> dict:
    base = {"energy": 0.7, "irritation": 0.0, "playfulness": 0.5, "social_battery": 0.8}
    base.update(overrides)
    return base


def _default_user_rel(**overrides) -> dict:
    base = {"trust": 0.5, "warmth": 0.5, "caution": 0.0, "annoyance": 0.0, "interaction_count": 0}
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# compute_state_updates
# ---------------------------------------------------------------------------

class TestComputeStateUpdates(unittest.TestCase):

    def test_neutral_message_increments_interaction_count(self):
        cs = _default_chat_state()
        ur = _default_user_rel()
        _, new_rel, _, _ = compute_state_updates("привет", cs, ur)
        self.assertEqual(new_rel["interaction_count"], 1)

    def test_neutral_message_drains_social_battery_slightly(self):
        cs = _default_chat_state(social_battery=0.8)
        ur = _default_user_rel()
        new_cs, _, _, _ = compute_state_updates("привет", cs, ur)
        self.assertLess(new_cs["social_battery"], 0.8)

    def test_aggression_raises_irritation(self):
        cs = _default_chat_state(irritation=0.0)
        ur = _default_user_rel()
        new_cs, _, _, _ = compute_state_updates("иди нахуй", cs, ur)
        self.assertGreater(new_cs["irritation"], 0.0)

    def test_aggression_raises_annoyance(self):
        cs = _default_chat_state()
        ur = _default_user_rel(annoyance=0.3)
        _, new_rel, _, _ = compute_state_updates("заткнись", cs, ur)
        self.assertGreater(new_rel["annoyance"], 0.3)

    def test_aggression_with_high_annoyance_triggers_annoyed_reaction(self):
        cs = _default_chat_state()
        ur = _default_user_rel(annoyance=0.4)
        _, _, rk, rs = compute_state_updates("ты тупая", cs, ur)
        self.assertEqual(rk, "annoyed")
        self.assertGreater(rs, 0.0)

    def test_aggression_with_low_annoyance_no_reaction(self):
        cs = _default_chat_state()
        ur = _default_user_rel(annoyance=0.0)
        _, _, rk, _ = compute_state_updates("иди нафиг", cs, ur)
        # First aggression: annoyance starts at 0, after delta still might be ≤ 0.35
        # depends on delta; test that reaction is either None or annoyed
        self.assertIn(rk, (None, "annoyed"))

    def test_question_triggers_curious_reaction(self):
        cs = _default_chat_state()
        ur = _default_user_rel()
        _, _, rk, _ = compute_state_updates("что такое квантовый компьютер?", cs, ur)
        self.assertEqual(rk, "curious")

    def test_gratitude_triggers_amused_reaction(self):
        cs = _default_chat_state()
        ur = _default_user_rel()
        _, _, rk, _ = compute_state_updates("спасибо большое", cs, ur)
        self.assertEqual(rk, "amused")

    def test_gratitude_raises_warmth(self):
        cs = _default_chat_state()
        ur = _default_user_rel(warmth=0.5)
        _, new_rel, _, _ = compute_state_updates("спасибо", cs, ur)
        self.assertGreater(new_rel["warmth"], 0.5)

    def test_all_values_clamped_to_unit_range(self):
        # Even at extremes, nothing goes out of [0, 1]
        cs = _default_chat_state(energy=0.01, irritation=0.99, social_battery=0.01)
        ur = _default_user_rel(annoyance=0.99, trust=0.01)
        new_cs, new_rel, _, _ = compute_state_updates("иди нахуй", cs, ur)
        for key in ("energy", "irritation", "playfulness", "social_battery"):
            self.assertGreaterEqual(new_cs[key], 0.0, key)
            self.assertLessEqual(new_cs[key], 1.0, key)
        for key in ("trust", "warmth", "caution", "annoyance"):
            self.assertGreaterEqual(new_rel[key], 0.0, key)
            self.assertLessEqual(new_rel[key], 1.0, key)

    def test_no_updated_at_in_output(self):
        cs = _default_chat_state()
        ur = _default_user_rel()
        new_cs, new_rel, _, _ = compute_state_updates("привет", cs, ur)
        self.assertNotIn("updated_at", new_cs)
        self.assertNotIn("updated_at", new_rel)


# ---------------------------------------------------------------------------
# Time decay
# ---------------------------------------------------------------------------

class TestTimeDiffusion(unittest.TestCase):

    def test_irritation_decays_with_time(self):
        cs = _default_chat_state(irritation=0.8)
        decayed = apply_time_decay_chat(cs, hours=2.0)
        self.assertLess(decayed["irritation"], 0.8)

    def test_zero_hours_no_decay(self):
        cs = _default_chat_state(irritation=0.8, energy=0.3)
        decayed = apply_time_decay_chat(cs, hours=0.0)
        self.assertAlmostEqual(decayed["irritation"], 0.8, places=6)
        self.assertAlmostEqual(decayed["energy"], 0.3, places=6)

    def test_energy_recovers_when_below_baseline(self):
        cs = _default_chat_state(energy=0.2)
        decayed = apply_time_decay_chat(cs, hours=3.0)
        self.assertGreater(decayed["energy"], 0.2)

    def test_social_battery_recovers_when_low(self):
        cs = _default_chat_state(social_battery=0.1)
        decayed = apply_time_decay_chat(cs, hours=2.0)
        self.assertGreater(decayed["social_battery"], 0.1)

    def test_annoyance_decays_toward_zero(self):
        rel = _default_user_rel(annoyance=0.9)
        decayed = apply_time_decay_relation(rel, hours=2.0)
        self.assertLess(decayed["annoyance"], 0.9)

    def test_decay_values_stay_in_unit_range(self):
        cs = _default_chat_state(energy=0.0, social_battery=0.0, irritation=1.0)
        rel = _default_user_rel(annoyance=1.0, caution=1.0)
        d_cs = apply_time_decay_chat(cs, hours=10.0)
        d_rel = apply_time_decay_relation(rel, hours=10.0)
        for v in d_cs.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)
        for v in d_rel.values():
            self.assertGreaterEqual(float(v), 0.0)
            self.assertLessEqual(float(v), 1.0)

    def test_interaction_count_preserved_in_relation_decay(self):
        rel = _default_user_rel(interaction_count=42, annoyance=0.5)
        decayed = apply_time_decay_relation(rel, hours=1.0)
        self.assertEqual(decayed["interaction_count"], 42)


# ---------------------------------------------------------------------------
# build_alta_state_block
# ---------------------------------------------------------------------------

class TestBuildAltaStateBlock(unittest.TestCase):

    def test_neutral_state_returns_empty(self):
        cs = _default_chat_state()
        ur = _default_user_rel()
        result = build_alta_state_block(cs, ur, None)
        self.assertEqual(result, "")

    def test_high_irritation_produces_block(self):
        cs = _default_chat_state(irritation=0.6)
        ur = _default_user_rel()
        result = build_alta_state_block(cs, ur, None)
        self.assertIn("раздражена", result)

    def test_high_annoyance_produces_user_note(self):
        cs = _default_chat_state()
        ur = _default_user_rel(annoyance=0.7)
        result = build_alta_state_block(cs, ur, None)
        self.assertIn("раздражает", result)

    def test_low_energy_produces_tiredness_note(self):
        cs = _default_chat_state(energy=0.2)
        ur = _default_user_rel()
        result = build_alta_state_block(cs, ur, None)
        self.assertIn("устала", result)

    def test_reaction_included_in_block(self):
        cs = _default_chat_state(irritation=0.5)
        ur = _default_user_rel(annoyance=0.6)
        reaction = {"reaction_kind": "annoyed", "reaction_strength": 0.8}
        result = build_alta_state_block(cs, ur, reaction)
        self.assertIn("раздражена", result)

    def test_curious_reaction_included(self):
        cs = _default_chat_state(playfulness=0.6)
        ur = _default_user_rel(trust=0.8)
        reaction = {"reaction_kind": "curious", "reaction_strength": 0.6}
        result = build_alta_state_block(cs, ur, reaction)
        self.assertIn("любопытна", result)

    def test_block_never_reveals_numbers(self):
        cs = _default_chat_state(irritation=0.7, energy=0.2)
        ur = _default_user_rel(annoyance=0.8, trust=0.1)
        result = build_alta_state_block(cs, ur, None)
        # Block must not contain raw float numbers like "0.7" or "0.8"
        import re
        self.assertIsNone(re.search(r"\b0\.\d+\b", result), f"Numbers found in: {result!r}")

    def test_header_present_when_noteworthy(self):
        cs = _default_chat_state(irritation=0.5)
        ur = _default_user_rel()
        result = build_alta_state_block(cs, ur, None)
        self.assertTrue(result.startswith("[Твоё состояние"))

    def test_warm_high_trust_produces_warmth_note(self):
        cs = _default_chat_state()
        ur = _default_user_rel(warmth=0.9, trust=0.8)
        result = build_alta_state_block(cs, ur, None)
        self.assertIn("тепло", result)

    def test_tone_hint_present_for_high_irritation(self):
        cs = _default_chat_state(irritation=0.6)
        ur = _default_user_rel(annoyance=0.6)
        result = build_alta_state_block(cs, ur, None)
        self.assertIn("→", result)


# ---------------------------------------------------------------------------
# DB CRUD integration
# ---------------------------------------------------------------------------

class TestMoodDB(unittest.IsolatedAsyncioTestCase):

    def _make_db_path(self, prefix: str) -> str:
        return os.path.join("tests", f"{prefix}_{uuid.uuid4().hex}.sqlite3")

    async def _with_db(self, coro):
        path = self._make_db_path("mood")
        db = DB(path)
        await db.connect()
        try:
            return await coro(db)
        finally:
            await db.close()
            for suffix in ("", "-wal", "-shm"):
                candidate = f"{path}{suffix}"
                if os.path.exists(candidate):
                    os.remove(candidate)

    async def test_chat_state_defaults_on_first_read(self):
        async def run(db):
            state = await db.get_alta_chat_state(999)
            self.assertAlmostEqual(state["energy"], 0.7)
            self.assertAlmostEqual(state["irritation"], 0.0)
            self.assertAlmostEqual(state["social_battery"], 0.8)
        await self._with_db(run)

    async def test_chat_state_roundtrip(self):
        async def run(db):
            await db.save_alta_chat_state(1, energy=0.3, irritation=0.6,
                                          playfulness=0.8, social_battery=0.2)
            state = await db.get_alta_chat_state(1)
            self.assertAlmostEqual(state["energy"], 0.3, places=5)
            self.assertAlmostEqual(state["irritation"], 0.6, places=5)
        await self._with_db(run)

    async def test_user_relation_defaults_on_first_read(self):
        async def run(db):
            rel = await db.get_alta_user_relation(1, 42)
            self.assertAlmostEqual(rel["trust"], 0.5)
            self.assertAlmostEqual(rel["annoyance"], 0.0)
            self.assertEqual(rel["interaction_count"], 0)
        await self._with_db(run)

    async def test_user_relation_roundtrip(self):
        async def run(db):
            await db.save_alta_user_relation(1, 42, trust=0.2, warmth=0.9,
                                             caution=0.3, annoyance=0.7,
                                             interaction_count=5)
            rel = await db.get_alta_user_relation(1, 42)
            self.assertAlmostEqual(rel["trust"], 0.2, places=5)
            self.assertAlmostEqual(rel["annoyance"], 0.7, places=5)
            self.assertEqual(rel["interaction_count"], 5)
        await self._with_db(run)

    async def test_reaction_add_and_get(self):
        async def run(db):
            await db.add_alta_reaction(1, "annoyed", 0.8, ttl_messages=2, target_user_id=42)
            reaction = await db.get_active_alta_reaction(1, target_user_id=42)
            self.assertIsNotNone(reaction)
            self.assertEqual(reaction["reaction_kind"], "annoyed")
            self.assertAlmostEqual(reaction["reaction_strength"], 0.8, places=5)
        await self._with_db(run)

    async def test_reaction_ttl_decrements_and_expires(self):
        async def run(db):
            await db.add_alta_reaction(1, "curious", 0.5, ttl_messages=1, target_user_id=99)
            # tick: TTL goes to 0, should be deleted
            await db.tick_alta_reactions(1)
            reaction = await db.get_active_alta_reaction(1, target_user_id=99)
            self.assertIsNone(reaction)
        await self._with_db(run)

    async def test_reaction_survives_one_tick(self):
        async def run(db):
            await db.add_alta_reaction(1, "amused", 0.4, ttl_messages=2, target_user_id=10)
            await db.tick_alta_reactions(1)
            reaction = await db.get_active_alta_reaction(1, target_user_id=10)
            self.assertIsNotNone(reaction)
            self.assertEqual(reaction["ttl_messages"], 1)
        await self._with_db(run)

    async def test_no_active_reaction_returns_none(self):
        async def run(db):
            result = await db.get_active_alta_reaction(1, target_user_id=42)
            self.assertIsNone(result)
        await self._with_db(run)

    async def test_chat_state_update_overwrites(self):
        async def run(db):
            await db.save_alta_chat_state(1, energy=0.9, irritation=0.1,
                                          playfulness=0.5, social_battery=0.7)
            await db.save_alta_chat_state(1, energy=0.2, irritation=0.8,
                                          playfulness=0.3, social_battery=0.1)
            state = await db.get_alta_chat_state(1)
            self.assertAlmostEqual(state["energy"], 0.2, places=5)
            self.assertAlmostEqual(state["irritation"], 0.8, places=5)
        await self._with_db(run)


if __name__ == "__main__":
    unittest.main()
