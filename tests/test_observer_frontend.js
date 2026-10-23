// Offline DOM fixture: no HTTP requests, browser profile, or application startup.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
class Element{
  constructor(name){this.name=name;this.children=[];this.attributes={};this.style={};this.textContent='';this.namespaceURI='http://www.w3.org/2000/svg'}
  replaceChildren(){this.children=[];this.textContent=''}
  append(child){this.children.push(child)}
  setAttribute(name,value){this.attributes[name]=value}
}
const elements=new Map(),element=id=>{if(!elements.has(id))elements.set(id,new Element(id));return elements.get(id)};
let abortTimeout;const requestedUrls=[];
const context=vm.createContext({document:{getElementById:element,querySelector:element,createElement:name=>new Element(name),createElementNS:(_,name)=>new Element(name)},Date,AbortController,
  setTimeout:fn=>{abortTimeout=fn;return 1},clearTimeout:()=>{},
  fetch:(url,options)=>{requestedUrls.push(url);return new Promise((resolve,reject)=>options.signal.addEventListener('abort',()=>reject(Error('aborted'))))}});
const html=fs.readFileSync(require('path').join(__dirname,'../observer/index.html'),'utf8');
const script=html.split('<script>')[1].split('</script>')[0].split("$('loginForm').addEventListener")[0];
vm.runInContext(script,context);
const data={completed_at:Date.now()/1000,mode:'shadow',budget_tmn:2_000_000,risk:{limits_known:true,floor:null,ceiling:null,gross_cap:.5,max_drawdown:.05,daily_loss:.02,position_cap:.25},
  presentation:{title:'موتور در حال اجراست',warnings:[],history:[]},assets:[],open_orders:[],journal:{exists:true,mode:'shadow',ledger:{cash_tmn:1_900_000,realized_pnl_tmn:-42,fees_tmn:2,fills:1},summary:{positions:{}},paper_trades:[{symbol:'BTCUSDT',side:'BUY',price:.12345678,fee:2,quantity:1,ts:100}],curve:[]}};
context.fixture=data;vm.runInContext('render(fixture)',context);
assert(element('modeBadge').textContent.includes('فرضی'));
assert(element('cashUnit').textContent.includes('فرضی'));
assert(element('tradesTitle').textContent.includes('شبیه‌سازی'));
assert(element('footer').textContent.includes('بدون سفارش واقعی'));
assert.equal(element('floor').textContent,'غیرفعال');
assert(element('dailyLoss').textContent.includes('۲'));
data.journal.orders=[{id:'synthetic-buy',symbol:'BTCTMN',side:'BUY',status:'DONE',requested:2,executed:1.996,price:100200.4008016,fee:0,fee_tmn:400,reason:'synthetic'},
  {id:'synthetic-pending',symbol:'BTCTMN',side:'BUY',status:'SENDING',requested:2,fee_tmn:null}];
vm.runInContext('render(fixture)',context);
const orderRows=element('orders').children[0].children;
assert.equal(orderRows[0].children[7].textContent,'کل کارمزد (تومان)');
assert.equal(orderRows[1].children[7].textContent,'۴۰۰','base-asset fee valuation must be visible despite quote fee being zero');
assert.equal(orderRows[2].children[7].textContent,'—','pending orders must not imply confirmed zero fees');
assert.equal(element('adaptiveRiskPanel').hidden,false);
assert(element('adaptiveStatus').textContent.includes('هنوز ارزیابی'));
data.journal.summary.adaptive_risk={policy:'paper_risk_reduction_v1',as_of:Date.now()/1000,multiplier:.4,
  reasons:['BTCTMN:volatility_elevated','news_coverage_limited'],effective_caps:{position_cap:.1,gross_cap:.2,risk_fraction:.0012,portfolio_stop_risk:.0032},
  news:{coverage:false,freshness_seconds:3600,source_count:2,conflict_share:.25,model_news_coverage:true},
  markets:{BTCTMN:{multiplier:.4,daily_volatility:.07,spread_bps:12,depth_tmn:1_000_000,reasons:['volatility_elevated']}}};
vm.runInContext('render(fixture)',context);
assert.equal(element('adaptiveMultiplier').textContent,'۰٫۴ برابر');
assert.equal(element('adaptiveGross').textContent,'۲۰٪');
assert(element('adaptiveNews').textContent.includes('ناکافی'));
assert(element('adaptiveNews').textContent.includes('۱ ساعت'));
assert(element('adaptiveNews').textContent.includes('سن انتشار آخرین خبر'));
assert(element('adaptiveNews').textContent.includes('شش ساعت پیش از ارزیابی'));
assert(element('adaptiveReasons').textContent.includes('علت‌های ثبت‌شده'));
assert(element('adaptiveReasons').textContent.includes('BTCTMN: نوسان بازار بالاست'));
assert(!element('adaptiveReasons').textContent.includes('volatility_elevated'));
assert.equal(vm.runInContext("riskReason('unrecognized_reason')",context),'unrecognized_reason');
assert.equal(element('adaptiveMarkets').children[0].children.length,2);
data.journal.summary.adaptive_risk.multiplier=0;vm.runInContext('render(fixture)',context);
assert(element('adaptiveStatus').textContent.includes('متوقف'));
data.journal.summary.adaptive_risk.multiplier=.4;
vm.runInContext('adaptiveRisk(null,false)',context);
assert.equal(element('adaptiveRiskPanel').hidden,true,'adaptive paper policy must not appear as live account risk');
vm.runInContext('render(fixture)',context);
assert(vm.runInContext("price({symbol:'BTCUSDT',price:.12345678})",context).endsWith('USDT'));
assert(vm.runInContext("price({symbol:'UNKNOWN',price:2})",context).endsWith('واحد نامشخص'));
vm.runInContext('last.completed_at-=100;freshness()',context);
assert(element('engine').textContent.startsWith('وضعیت فعلی تأیید نمی‌شود'));
assert(element('warning').textContent.includes('داده تازه نیست'));
vm.runInContext('chart([{ts:100,equity:10},{ts:101,equity:20},{ts:200,equity:15}])',context);
const points=element('chart').children[0].children[0].attributes.points.split(' ').map(x=>Number(x.split(',')[0]));
assert(Math.abs(points[1]-21.7)<.001,'chart must preserve irregular timestamp spacing');
(async()=>{
 const pending=vm.runInContext('refresh()',context);abortTimeout();await pending;
 assert.equal(requestedUrls.at(-1),'/api/snapshot','default observer endpoint must remain unchanged');
 assert.equal(vm.runInContext('busy',context),false,'timeout must release poll guard');
 assert(element('updated').textContent.includes('ناموفق'));
 const again=vm.runInContext('refresh()',context);abortTimeout();await again;
 assert.equal(vm.runInContext('busy',context),false,'polling must remain retryable');
 const paperContext=vm.createContext({WALLEX_API_PREFIX:'/paper',document:context.document,Date,AbortController,
   setTimeout:context.setTimeout,clearTimeout:()=>{},fetch:context.fetch});
 vm.runInContext(script,paperContext);
 const paperRefresh=vm.runInContext('refresh()',paperContext);abortTimeout();await paperRefresh;
 assert.equal(requestedUrls.at(-1),'/paper/api/snapshot');
 const paperLogin=vm.runInContext("request('/login',{method:'POST'})",paperContext);abortTimeout();await assert.rejects(paperLogin);
 assert.equal(requestedUrls.at(-1),'/paper/login');
 assert.throws(()=>vm.runInContext("render({mode:'read_only'})",paperContext),/non-simulation snapshot/);
 const invalidPrefix=vm.createContext({WALLEX_API_PREFIX:'//elsewhere.example'});
 assert.throws(()=>vm.runInContext(script,invalidPrefix),/invalid dashboard API prefix/);
 console.log('Observer frontend: simulation/adaptive risk/news, API prefixes, currencies, watchdog, timestamps and timeout passed.');
})().catch(error=>{console.error(error);process.exitCode=1});
