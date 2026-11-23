/* Transfer/export/import regression test without a browser. */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '..', 'templates', 'index.html'), 'utf8');
const js = html.match(/<script>([\s\S]*?)<\/script>/)?.[1];
assert.ok(js, 'application JS was not found');
new vm.Script(js);
assert.ok(html.includes('data-action="transfer"'));
assert.ok(js.includes("case '/transfer': return doTransfer();"));
assert.ok(js.includes('recovered_from_backup'));

const start = js.indexOf('function doTransfer()');
const end = js.indexOf('function doReset()', start);
assert.ok(start >= 0 && end > start, 'transfer function was not found');

const elements = {};
for (const id of ['transfer-area','transfer-info','transfer-file','transfer-close',
                  'transfer-copy','transfer-download','transfer-file-btn','transfer-apply']) {
  elements[id] = {value:'', textContent:'', className:'', onclick:null,
                  addEventListener(){}, focus(){}, select(){}, click(){}};
}
const calls = {persist:0, renderAll:0, renderStatus:0, messages:[]};
const ctx = {
  messages: [{role:'user',content:'старый диалог'}],
  state: {name:'Старый герой',level:1},
  memory: {}, uiEvents: [], busy:false, rolling:false,
  EMPTY_STATE: {name:'Пустой',race:'',class:'',background:'',position:'стоит',cover:'нет',weapon_in_hand:'',level:1,xp:0,xp_next:100,hp:10,max_hp:10,gold:0,dc_modifier:0,level_pending:0,stats:{str:10,dex:10,con:10,int:10,wis:10,cha:10},proficiencies:[],skills:[],abilities:[],inventory:[],notes:[],quests:[],conditions:[],relationships:[],faction_clocks:[],dc_modifier_reasons:[],level_history:[],xp_event_ids:[],restrained:false},
  modalContent: {innerHTML:''},
  modalEl: {classList:{add(){},remove(){calls.closed=true;}}},
  document: {getElementById(id){return elements[id];}},
  persist(){calls.persist++;},
  renderAll(){calls.renderAll++;},
  renderStatus(){calls.renderStatus++;},
  addMsg(role,text){calls.messages.push({role,text});},
  confirm(){return true;},
};
vm.createContext(ctx);
vm.runInContext(js.slice(start,end) + '\nthis.runTransfer = doTransfer;',ctx);
ctx.runTransfer();

const imported = {
  schema_version:4,
  messages:[{role:'user',content:'Новая партия'},{role:'assistant',content:'Начало'}],
  state:{name:'Новый герой',level:4,gold:'<img src=x onerror=alert(1)>',race:{bad:'value'}},
  memory:{summary:'Память',covered_count:0,prefix_hash:''},
  ui_events:[{at:2,kind:'dice',roll:{expr:'1d20',rolls:[12],mod:0,total:12}}],
};
elements['transfer-area'].value = JSON.stringify(imported);
elements['transfer-apply'].onclick();
assert.equal(ctx.state.name,'Новый герой');
assert.equal(ctx.state.level,4);
assert.equal(ctx.state.gold,0);
assert.equal(ctx.state.race,'');
assert.equal(ctx.messages.length,2);
assert.equal(ctx.memory.summary,'Память');
assert.equal(ctx.uiEvents.length,1);
assert.equal(calls.persist,1);
assert.equal(calls.renderAll,1);
assert.equal(calls.renderStatus,1);
assert.equal(calls.closed,true);
assert.ok(calls.messages[0].text.includes('импортирована'));

// Import must reject malformed dialogue without replacing the current game.
ctx.runTransfer();
elements['transfer-area'].value = JSON.stringify({messages:[{role:'system',content:'bad'}]});
elements['transfer-apply'].onclick();
assert.equal(ctx.state.name,'Новый герой');
assert.equal(calls.persist,1);
assert.ok(elements['transfer-info'].textContent.includes('неверного формата'));

console.log('Campaign transfer: export UI, import, state/memory/cards and invalid data passed');
