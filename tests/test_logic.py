"""Static import of the pure game functions so tests run even without Flask/OpenAI."""
import ast
import copy
import json
import re
import unittest
from pathlib import Path


def load_logic():
    source = Path(__file__).resolve().parents[1] / 'app.py'
    nodes = ast.parse(source.read_text(encoding='utf-8')).body
    names = {'_identity', 'state_for_model', 'apply_state_delta', '_xp_after_turn'}
    selected = [n for n in nodes if isinstance(n, ast.FunctionDef) and n.name in names]
    code = compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec')
    ns = {'copy': copy, 'json': json, 're': re,
          'XP_EVENT_RE': re.compile(r'\+\s*(\d+)\s*XP\b', re.IGNORECASE),
          'STATE_LIST_FIELDS': ('inventory', 'notes', 'quests', 'conditions', 'relationships',
                                'faction_clocks', 'dc_modifier_reasons', 'proficiencies',
                                'skills', 'abilities', 'level_history'),
          'STATE_SET_FIELDS': {'hp', 'max_hp', 'gold', 'name', 'race', 'class', 'background',
                               'stats', 'position', 'cover', 'weapon_in_hand',
                               'dc_modifier', 'restrained'}}
    exec(code, ns)
    return ns


L = load_logic()


class LogicTests(unittest.TestCase):
    def test_delta_add_update_remove_without_data_loss(self):
        initial = {'hp': 10, 'inventory': ['ключ', 'зелье'],
                   'quests': [{'id': 'q1', 'name': 'Пещера', 'status': 'active', 'stages': [1, 2]}],
                   'notes': [{'id': 'n1', 'title': 'Письмо', 'content': 'Текст', 'unread': True}],
                   'conditions': ['отравлен']}
        delta = {'set': {'hp': 7, 'xp': 999999},
                 'remove': {'inventory': ['зелье'], 'conditions': ['отравлен']},
                 'update': {'quests': [{'id': 'q1', 'status': 'completed'}]},
                 'add': {'inventory': ['нож'], 'notes': [{'id': 'n2', 'title': 'Слух'}]}}
        actual = L['apply_state_delta'](initial, delta)
        self.assertEqual(actual['hp'], 7)
        self.assertNotIn('xp', actual)
        self.assertEqual(actual['inventory'], ['ключ', 'нож'])
        self.assertEqual(actual['conditions'], [])
        self.assertEqual(actual['quests'][0]['stages'], [1, 2])
        self.assertEqual(actual['quests'][0]['status'], 'completed')
        self.assertEqual(actual['notes'][0]['content'], 'Текст')
        self.assertEqual(initial['inventory'], ['ключ', 'зелье'])

    def test_removed_xp_hashes_from_prompt(self):
        st = {'name': 'Герой', 'xp_event_ids': ['hash'] * 500, 'xp_scored_through': 800,
              'level_history': [{'level': 2}],
              'notes': [{'id': 'n1', 'title': 'Карта', 'content': 'Длинный секрет'}],
              'quests': [{'id': 'q1', 'name': 'Поиск', 'status': 'active', 'rewards': {'xp': 100}}],
              'relationships': [], 'faction_clocks': []}
        trimmed = L['state_for_model'](st)
        as_text = json.dumps(trimmed, ensure_ascii=False)
        self.assertNotIn('xp_event_ids', as_text)
        self.assertNotIn('hash', as_text)
        self.assertNotIn('Длинный секрет', as_text)
        self.assertNotIn('rewards', as_text)
        self.assertIn('Карта', as_text)

    def test_xp_cursor_prevents_replay_and_preserves_old_history(self):
        msgs = [{'role': 'assistant', 'content': 'Победа +50 XP'},
                {'role': 'user', 'content': 'Иду дальше'},
                {'role': 'assistant', 'content': '+25 XP за ловкость'}]
        self.assertEqual(L['_xp_after_turn'](msgs, {'xp': 50}, {'xp': 50, 'xp_scored_through': 1}), (75, 3))
        self.assertEqual(L['_xp_after_turn'](msgs, {'xp': 75}, {'xp': 75, 'xp_scored_through': 3}), (75, 3))
        self.assertEqual(L['_xp_after_turn'](msgs, {'xp': 50, 'xp_event_ids': ['legacy']}, {'xp': 50, 'xp_event_ids': ['legacy']}), (75, 3))


if __name__ == '__main__':
    unittest.main()
