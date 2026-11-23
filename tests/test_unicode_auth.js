/* Verify that Russian usernames are never transmitted in fetch headers.
   Run: node tests/test_unicode_auth.js */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '..', 'templates', 'index.html'), 'utf8');
const js = html.match(/<script>([\s\S]*?)<\/script>/)?.[1];
assert.ok(js, 'application JS was not found');
new vm.Script(js);

// Browsers (and Node's Request class) reject these characters in header values.
assert.throws(() => new Request('https://example.test/api/chat', {
  headers: {'X-RPG-User': 'иван_123'}
}), /ByteString|non ISO-8859-1|outside the range|invalid|character/i);

const sent = [];
const ctx = {
  authUser: 'иван_123',
  authToken: 'aB1_-safeSessionToken0123456789',
  fetch: async (url, options) => {
    // Actually run the browser-compatible HTTP header validator.
    const req = new Request('https://example.test' + url, options);
    const headers = req.headers;
    assert.equal(headers.get('x-rpg-user'), null, 'username leaked in header');
    assert.equal(headers.get('authorization'), 'Bearer aB1_-safeSessionToken0123456789');
    sent.push({url, headers});
    if (url === '/api/auth/check') {
      return {ok: true, json: async () => ({ok: true, user: 'иван_123'})};
    }
    return {ok: true, status: 200, json: async () => ({ok: true})};
  },
};
vm.createContext(ctx);
const start = js.indexOf('function authHeaders()');
const stop = js.indexOf('async function showLogin()', start);
const apiStart = js.indexOf('async function apiFetch(');
const apiStop = js.indexOf('/* POSITION / COVER */', apiStart);
assert.ok(start >= 0 && stop > start && apiStart > 0 && apiStop > apiStart);
vm.runInContext(js.slice(start, stop) + '\n' + js.slice(apiStart, apiStop), ctx);

(async () => {
  assert.equal(await ctx.checkAuth(ctx.authToken), 'иван_123');
  const response = await ctx.apiFetch('/api/chat', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({messages: [{role: 'user', content: 'Привет'}]})
  });
  assert.equal(response.status, 200);
  assert.deepEqual(sent.map(x => x.url), ['/api/auth/check', '/api/chat']);
  assert.equal(sent[1].headers.get('content-type'), 'application/json');
  assert.equal(Object.keys(ctx.authHeaders()).length, 1);
  console.log('Cyrillic usernames: login check and chat fetch headers pass');
})().catch(err => { console.error(err); process.exitCode = 1; });
