import json
import unittest

from campaign import (
    active_check, apply_ability_event, initial_identity, merge_facts, merge_scene, normalize_ability,
    pending_check, roll_result, rules_result,
)


class CampaignTest(unittest.TestCase):
    def test_legacy_check_survives_request_with_trimmed_history(self):
        reply = {"role": "assistant", "content": "Ловкость, DC 12. Команда: /roll 2d20kh1+1"}
        history = [{"role": "user", "content": "Иду по балке."}, reply]
        check = active_check({}, history)
        self.assertEqual(check["id"], active_check({}, history[-1:])["id"])
        restored = json.loads(json.dumps({"pending_check": check}))
        check = active_check(restored, [])
        roll = {"expr": "2d20kh1+1", "rolls": [15, 2], "check_id": check["id"]}
        result = roll_result(check, roll)
        self.assertEqual(result["total"], 16)
        self.assertTrue(result["success"])
        self.assertIsNone(active_check({"pending_check": None}, history))

    def test_dice_reject_mismatched_check_and_formula(self):
        check = pending_check("Атака копьём против КД 13. /roll 1d20+6")
        self.assertEqual(check["kind"], "attack")
        base = {"expr": "1d20+6", "rolls": [19], "check_id": check["id"]}
        for altered in ({"check_id": "another"}, {"expr": "1d20+4"}, {"rolls": [21]}):
            with self.subTest(altered=altered), self.assertRaises(ValueError):
                roll_result(check, {**base, **altered})
        self.assertEqual(roll_result(check, base)["total"], 25)

    def test_damage_calculation_and_capped_healing(self):
        check = pending_check("Болт попадает в тебя. Урон арбалета: /roll 1d8+2")
        result = roll_result(check, {"expr": check["expr"], "rolls": [1], "check_id": check["id"]})
        self.assertEqual(rules_result({"hp": 18, "max_hp": 18}, result)["hp_after"], 15)
        heal = pending_check("Второе дыхание. Брось лечение: /roll 1d10+3")
        result = roll_result(heal, {"expr": heal["expr"], "rolls": [4], "check_id": heal["id"]})
        self.assertEqual(rules_result({"hp": 15, "max_hp": 18}, result)["hp_after"], 18)

    def test_rest_and_spending(self):
        stance = {"name": "Защитная стойка", "uses_left": 1, "uses_max": 2, "recharge": "short_rest"}
        passive = {"name": "Сопротивление кислоте", "recharge": "passive"}
        spent, ok = apply_ability_event([stance, passive], "Защитная стойка")
        self.assertTrue(ok)
        self.assertEqual(spent[0]["uses_left"], 0)
        self.assertFalse(apply_ability_event(spent, "Защитная стойка")[1])
        restored, _ = apply_ability_event(spent, rest="short_rest")
        self.assertEqual(restored, [{**stance, "uses_left": 2}, passive])

    def test_legacy_unlimited_ability_has_no_charge_counter(self):
        legacy = {
            "name": "**Мистический заряд 1d10&#x20;**",
            "uses_max": -1,
            "uses_left": -1,
            "recharge": "short_rest",
        }
        normalized = {"name": "Мистический заряд 1d10", "recharge": "unlimited"}
        self.assertEqual(normalize_ability(legacy), normalized)
        self.assertEqual(apply_ability_event([legacy], "Мистический заряд 1d10"), ([normalized], True))
        self.assertEqual(apply_ability_event([normalized], rest="short_rest")[0], [normalized])
        passive = {**legacy, "name": "Сопротивление", "recharge": "passive"}
        self.assertEqual(normalize_ability(passive), {"name": "Сопротивление", "recharge": "passive"})

    def test_character_identity_keeps_chosen_fields_and_fills_skipped_fields(self):
        reply = "[[ЛИСТ]]\n👤 Лис — Эльф · Следопыт · Скиталец\n[[/ЛИСТ]]"
        selected = {"name": "Пустой", "race": "", "class": "Ловец ветра"}
        self.assertEqual(initial_identity(reply, selected), {
            "name": "Лис", "race": "Эльф", "class": "Ловец ветра", "background": "Скиталец",
        })
        self.assertEqual(initial_identity("Выберите расу.", selected), {})
        self.assertEqual(initial_identity(reply.removesuffix("[[/ЛИСТ]]"), selected), {})

    def test_scene_identity_and_transfer_of_a_clue(self):
        scene = merge_scene({}, {"location": "Мельница", "participants": ["Рагар", "Гонец"]})
        update = merge_scene(scene, {"situation": "Обнаружена сумка."})
        self.assertEqual(scene["id"], update["id"])
        arrived = merge_scene(update, {"location": "Чёрный брод", "participants": ["Рагар", "Начальник"]})
        self.assertNotEqual(scene["id"], arrived["id"])
        facts = merge_facts([], [{"id": "дощечка", "subject": "Дощечка", "detail": "Имя Вейн",
                                  "holder": "Рагар", "status": "confirmed"}])
        facts = merge_facts(facts, [{"id": "дощечка", "holder": "Начальник"}])
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["detail"], "Имя Вейн")
        self.assertEqual(facts[0]["holder"], "Начальник")


if __name__ == "__main__":
    unittest.main()
