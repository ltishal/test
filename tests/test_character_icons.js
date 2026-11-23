/* UI regression tests without browser dependencies: node tests/test_character_icons.js */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '..', 'templates', 'index.html'), 'utf8');
assert.equal((html.match(/<script>/g) || []).length, 1);
const js = html.match(/<script>([\s\S]*?)<\/script>/)[1];
new vm.Script(js);  // whole application must have valid JS syntax

for (const forbidden of [/pollinations/i, /dicebear/i, /generateAvatar/, /cachedAvatarFor/, /new Image\s*\(/, /regen-avatar/, /AVATAR_KEY/]) {
  assert.ok(!forbidden.test(html), `old image generation must be removed: ${forbidden}`);
}

const functions = js.slice(js.indexOf('const CHARACTER_CHOICES ='), js.indexOf('/* TTS */'));
const ctx = {esc: s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))};
vm.createContext(ctx);
vm.runInContext(functions + '\nthis.checkIcon = characterIcon; this.makeTags = characterIdentity;', ctx);
for (const [type, text, expected] of [
  ['race','Эльф','🧝'], ['race','человек','🧑'], ['race','полуэльф','🌿'],
  ['race','Полуорк','🛡️'], ['race','драконорожденный','🐉'],
  ['race','тифлинг','😈'], ['race','дроу','🌙'], ['race','Гном','🛠️'],
  ['class','Маг','🧙'], ['class','волшебник','🧙'], ['class','Воин','⚔️'],
  ['class','Паладин','🛡️'], ['class','Плут','🗡️'], ['class','Друид','🌳'],
  ['background','преступник','⛓️'], ['background','Солдат','🎖️'],
  ['background','Благородный','👑'], ['background','Моряк','⚓'],
]) {
  assert.equal(ctx.checkIcon(type, text), expected, `${type}/${text}`);
}
const tags = ctx.makeTags({race:'Эльф',class:'Маг',background:'Преступник'});
assert.ok(tags.includes('🧝') && tags.includes('🧙') && tags.includes('⛓️'));
assert.ok(!ctx.makeTags({race:'',class:'',background:''}));
assert.ok(!ctx.makeTags({race:'<img src=x onerror=alert(1)>',class:'',background:''}).includes('<img'));
assert.ok(html.includes('data-action="create"'), 'new character button available in menu');
assert.ok(html.includes('data-group="${key}"'), 'character creation options exist');
assert.ok(html.includes("startGame(description, {name, race, class: cls, background})"), 'selected values immediately applied to character state');
console.log('Character insignia and UI smoke checks passed');
