/* UI-only character and dice cards: no browser dependencies.
   Run: node tests/test_chat_cards.js */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '..', 'templates', 'index.html'), 'utf8');
const js = html.match(/<script>([\s\S]*?)<\/script>/)[1];
new vm.Script(js);
const gameplayStart = js.indexOf('/* Gameplay cards are local UI events');
const gameplayEnd = js.indexOf('/* API CALLS */', gameplayStart);
const renderStart = js.indexOf('function renderAll()');
const renderEnd = js.indexOf('function showIntro()', renderStart);
assert.ok(gameplayStart >= 0 && gameplayEnd > gameplayStart && renderStart >= 0 && renderEnd > renderStart);
const rendered = [];
let saveCount = 0;
const ctx = {
  messages: [], uiEvents: [],
  chatEl: {innerHTML: ''}, tts: {},
  esc: s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),
  characterIdentity: s => `${s.race || ''} ${s.class || ''} ${s.background || ''}`,
  addHtmlMsg: (role, html) => { rendered.push({role, html}); },
  addMsg: (role, text) => { rendered.push({role, text}); },
  persist: () => { saveCount++; },
};
vm.createContext(ctx);
vm.runInContext(js.slice(gameplayStart, gameplayEnd) + '\n' + js.slice(renderStart, renderEnd) + '\nthis.helpers={snapshotCharacter,characterSheetHtml,diceResultHtml,recordUiEvent,renderAll};', ctx);
const {snapshotCharacter, characterSheetHtml, diceResultHtml, recordUiEvent, renderAll} = ctx.helpers;
const sheet = snapshotCharacter({name:'Эльф <img src=x onerror=alert(1)>', race:'Эльф', class:'Маг',background:'Преступник',
  hp:15,max_hp:20,level:2,xp:75,xp_next:100,gold:7, stats:{str:8,dex:16,con:12,int:18,wis:10,cha:14}});
const sheetHtml = characterSheetHtml(sheet);
assert.ok(sheetHtml.includes('Лист героя') && sheetHtml.includes('HP 15/20'));
for (const label of ['СИЛ', 'ЛОВ', 'ТЕЛ', 'ИНТ', 'МДР', 'ХАР']) assert.ok(sheetHtml.includes(label), label);
assert.ok(sheetHtml.includes('>+4</div>') && sheetHtml.includes('>−1</div>') === false);
assert.ok(!sheetHtml.includes('<img'));
const roll = {expr:'1d20+3', rolls:[16], mod:3, total:19};
assert.ok(diceResultHtml(roll).includes('19'));
assert.ok(!diceResultHtml({...roll,expr:'<script>'}).includes('<script>'));
// Record a character sheet once at the end of the character-creation turn.
ctx.messages = [{role:'user',content:'Создаю персонажа'},{role:'assistant',content:'Первая сцена'}];
recordUiEvent({kind:'character_sheet', sheet});
assert.equal(ctx.uiEvents[0].at, 2);
// Then, immediately before a dice-prompt sent to the model.
recordUiEvent({kind:'dice',roll});
ctx.messages.push({role:'user',content:'[[БРОСОК]] 1d20+3 → [16]+3 = 19. Сравни с DC...'});
ctx.messages.push({role:'assistant',content:'Результат действия'});
assert.equal(ctx.uiEvents[1].at, 2);
assert.equal(saveCount, 2);
rendered.length = 0;
renderAll();
assert.deepEqual(rendered.map(x => x.role), ['user','master','character-sheet','dice','master']);
assert.ok(rendered[2].html.includes('Эльф &lt;img'));
assert.ok(rendered[3].html.includes('1d20+3') && rendered[3].html.includes('19'));
// Older campaigns: roll embedded in a user message is rendered as a dice result.
ctx.uiEvents = [];
rendered.length = 0;
renderAll();
assert.deepEqual(rendered.map(x => x.role), ['user','master','dice','master']);
// UI events never become model dialogue and do not alter dialogue indexes.
assert.equal(ctx.messages.length, 4);
assert.equal(ctx.messages[2].role, 'user');
console.log('Character sheet and dice cards: chronology, restore, XSS and legacy checks passed');
