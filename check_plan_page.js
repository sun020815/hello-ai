// 用 node 跑一遍 plan.html 的渲染逻辑：不开浏览器，验证 render() 不崩、输出对。
//
// 为什么不截图：截图只能证明「这一次长这样」，证明不了「数据变少了不会崩」。
// 这里喂的是 docs/plan_sample.json（一次真实请求的完整返回），
// 除了正常分支还会再造一份「有错误、有被剔除动作」的返回体走一遍红条分支。
//
// 桩只实现页面真正用到的那几个 DOM 能力，宁可桩少也不要桩出浏览器不存在的行为。
//
// 跑法：node check_plan_page.js    （CI 的 static job 里跑，runner 自带 node）
const fs = require('fs');

const html = fs.readFileSync('static/plan.html', 'utf8');
const js = html.match(/<script>([\s\S]*)<\/script>/)[1];
const data = JSON.parse(fs.readFileSync('docs/plan_sample.json', 'utf8'));

const nodes = new Map();
function mk(sel) {
  const o = {
    _sel: sel, innerHTML: '', textContent: '', value: '', disabled: false,
    dataset: {}, style: {}, onclick: null, onsubmit: null, title: '',
    classList: { _s: new Set(), add(c) { this._s.add(c); }, remove(c) { this._s.delete(c); }, contains(c) { return this._s.has(c); } },
  };
  return o;
}
const document = {
  querySelector(sel) {
    if (!nodes.has(sel)) nodes.set(sel, mk(sel));
    return nodes.get(sel);
  },
  querySelectorAll() { return []; },
};
const store = {};
const localStorage = { getItem: k => store[k] || '', setItem: (k, v) => { store[k] = v; } };
const fetchStub = () => Promise.resolve({ ok: true, json: () => ({}) });
const setIntervalStub = () => 0;
const clearIntervalStub = () => {};

const run = new Function(
  'document', 'localStorage', 'fetch', 'setInterval', 'clearInterval', 'console', 'Date',
  js + '\n;return {render};'
);
const api = run(document, localStorage, fetchStub, setIntervalStub, clearIntervalStub, console, Date);

let fail = 0;
const check = (name, cond, extra = '') => {
  console.log((cond ? '  ok   ' : '  FAIL ') + name + (extra ? '   ' + extra : ''));
  if (!cond) fail++;
};

let out = '';
try {
  api.render(data);
  out = nodes.get('#res').innerHTML;
} catch (e) {
  console.log('  FAIL render 抛异常：' + e.message);
  process.exit(1);
}

check('渲染出内容（非空）', out.length > 500, out.length + ' 字符');
check('每天一张卡片', (out.match(/class="day"/g) || []).length === data.plan.length,
  (out.match(/class="day"/g) || []).length + ' vs ' + data.plan.length);
const nEx = data.plan.reduce((a, x) => a + x.exercises.length, 0);
check('动作行数对得上', (out.match(/class="ex-n"/g) || []).length === nEx,
  (out.match(/class="ex-n"/g) || []).length + ' vs ' + nEx);
check('每个动作都带 ref 标签', (out.match(/class="ref"/g) || []).length === nEx);
check('trace 每一步都渲染', (out.match(/class="tr /g) || []).length === data.trace.length,
  (out.match(/class="tr /g) || []).length + ' vs ' + data.trace.length);
check('显示工具调用次数', out.includes(data.trace.length + '</b>次工具调用'));
check('显示 token 数', out.includes(String(data.tokens) + '</b>token'));
check('摘要出现', !data.summary || out.includes('class="summary"'));
check('没有 undefined/null 漏到页面上', !/>undefined<|>null</.test(out));
check('没有未转义的脚本注入', !out.includes('<script'));

// 再跑一次「有错」的分支：错误清单和被剔除动作要能显示
const bad = JSON.parse(JSON.stringify(data));
bad.needs_review = true;
bad.audit.errors = [{ type: 'unknown_exercise', detail: '1 个动作不在库里，已剔除' }];
bad.dropped = [{ day: 1, name: '杠铃俯身飞鸟' }];
try {
  api.render(bad);
  const out2 = nodes.get('#res').innerHTML;
  check('红条出现', out2.includes('banner bad'));
  check('错误文案出现在页面上', out2.includes('1 个动作不在库里'));
  check('被剔除的动作单独列出', out2.includes('杠铃俯身飞鸟'));
} catch (e) {
  check('错误分支渲染不崩', false, e.message);
}

console.log(fail ? `失败 ${fail} 项` : '渲染检查全部通过');
process.exit(fail ? 1 : 0);
