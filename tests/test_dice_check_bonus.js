/* Test automatic modifier display and the exact server roll expression.
   Run: node tests/test_dice_check_bonus.js */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '..', 'templates', 'index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
new vm.Script(script);
const a = script.indexOf('function diceResultHtml(');
const b = script.indexOf('function showUiEvent(', a);
const c = script.indexOf('async function doRoll(');
const d = script.indexOf('/* SHEET */', c);
assert.ok(a >= 0 && b > a && c >= 0 && d > c);
const ctx = {
  busy: false, rolling: false, messages: [],
  state: {stats: {str:8, dex:14, con:10, int:12, wis:11, cha:9}},
  calls: [], cards: [], persisted: 0, sentMessages: [],
  esc: value => String(value).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch])),
  addHtmlMsg: () => ({remove(){}}),
  addMsg: (role, msg) => { throw new Error('Unexpected UI error: ' + role + msg); },
  recordUiEvent: e => ctx.cards.push(e),
  persist: () => {ctx.persisted += 1;},
  callAI: async () => {ctx.sentMessages.push(ctx.messages[ctx.messages.length - 1].content);},
  setTimeout: callback => callback(),
  apiFetch: async (url, options) => {
    assert.equal(url, '/api/roll');
    const {expr} = JSON.parse(options.body);
    ctx.calls.push(expr);
    const m = /^(?:1d20|d20)([+-]\d+)?$/.exec(expr);
    assert.ok(m, expr);
    const mod = Number(m[1] || 0);
    return {ok: true, json: async () => ({expr, rolls:[17], mod, total:17 + mod})};
  }
};
vm.createContext(ctx);
vm.runInContext(script.slice(a, b) + '\n' + script.slice(c, d) + '\n' +
                'this.helpers={pendingDeclaredCheck,diceResultHtml,doRoll};', ctx);
const {pendingDeclaredCheck, diceResultHtml, doRoll} = ctx.helpers;
function prompt(line) {
  ctx.messages = [{role:'assistant',content:line}];
  ctx.calls.length=0; ctx.cards.length=0; ctx.sentMessages.length=0;
}
(async () => {
  prompt('Ты стоишь перед дверью.\n\n**Бросок Ловкости: 1d20 + 2 — Сложность 12** (база 10, скользкие доски +2).');
  assert.equal(pendingDeclaredCheck().mod, 2);
  assert.equal(pendingDeclaredCheck().dc, 12);
  await doRoll('1d20');
  assert.equal(ctx.calls[0], '1d20+2');
  assert.equal(ctx.cards[0].roll.total, 19);
  assert.equal(ctx.cards[0].roll.label, 'Ловкости');
  assert.equal(ctx.cards[0].roll.dc, 12);
  const box = diceResultHtml(ctx.cards[0].roll);
  assert.ok(box.includes('17 + 2 = 19'), box);
  assert.ok(box.includes('Сложность 12'));
  assert.ok(ctx.sentMessages[0].includes('= 19'));
  assert.ok(ctx.sentMessages[0].includes('НЕ прибавляй модификаторы второй раз'));
  // Explicit manually entered modifier must be respected instead of +2 from the story.
  prompt('Бросок Ловкости: 1d20 + 2 — Сложность 12');
  await doRoll('1d20+4');
  assert.equal(ctx.calls[0], '1d20+4');
  assert.ok(diceResultHtml(ctx.cards[0].roll).includes('17 + 4 = 21'));
  // When GM specifies both ability and proficiency, include both exactly once.
  prompt('Бросок Ловкости: 1d20+3 (Ловкость) +2 (Акробатика) — Сложность 15');
  await doRoll('1d20');
  assert.equal(ctx.calls[0], '1d20+5');
  assert.ok(diceResultHtml(ctx.cards[0].roll).includes('17 + 5 = 22'));
  // When model omits the modifier, infer only the named ability from the sheet.
  prompt('Проверка Ловкости: 1d20 — Сложность 12');
  await doRoll('1d20');
  assert.equal(ctx.calls[0], '1d20+2');
  // A negative ability modifier should reduce total as declared.
  prompt('Бросок Силы: 1d20 — Сложность 12');
  await doRoll('1d20');
  assert.equal(ctx.calls[0], '1d20-1');
  assert.ok(diceResultHtml(ctx.cards[0].roll).includes('17 − 1 = 16'));
  // Normal die rolls do not borrow modifiers from earlier GM messages.
  prompt('Из сундука доносится стук.');
  await doRoll('1d20');
  assert.equal(ctx.calls[0], '1d20');
  prompt('Бросок Ловкости: 1d20 + 2 — Сложность 12');
  await doRoll('1d20+2');
  assert.equal(ctx.calls[0], '1d20+2'); // no +2 twice
  assert.ok(!diceResultHtml({...ctx.cards[0].roll, label:'<img src=x>'}).includes('<img'));
  console.log('D20 check modifier: auto +2, explicit modifiers, ability fallback, negative, no double count, XSS passed');
})().catch(err=>{console.error(err);process.exitCode=1;});
