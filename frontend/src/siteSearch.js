import { matchesSearch } from './cabinetToolsState.js';

export const SEARCH_SECTIONS = [
  {title:'Главная',text:'Обзор кабинета и важные события для компании',keywords:'главный экран сводка поступления списания',view:'home',icon:'⌂'},
  {title:'Инсайты',text:'Новости, риски и возможности для вашего бизнеса',keywords:'события поиск по тегам профиль персонализация',view:'insights',icon:'✦'},
  {title:'Компания',text:'Сведения о бизнесе, группа и банковские продукты',keywords:'профиль отрасль ОКВЭД регион выручка команда численность ВЭД внешнеторговые связи ФТС страны',view:'company',icon:'▤'},
  {title:'Контрагенты',text:'Поставщики, покупатели и платёжные связи',keywords:'компании платим получаем',view:'counterparties',icon:'↔'},
  {title:'Платёжная аналитика',text:'Поступления, списания и структура платежей',keywords:'платежи средний платёж концентрация отрасли регионы сводка скачать CSV',view:'payments',icon:'↗'},
  {title:'Документооборот',text:'Аналитические отчёты и отраслевые обзоры в PDF',keywords:'документы импорт загрузка аналитика',view:'documents',icon:'▧'},
  {title:'Календарь',text:'События из новостей и отчётов по датам',view:'calendar',icon:'□'},
  {title:'Чат с банком',text:'Задайте помощнику вопрос о вашей компании',keywords:'ИИ AI агент',view:'chat',icon:'◌'},
];

export const findSearchItems = (items, query) => items.filter(item=>matchesSearch(`${item.title} ${item.text} ${item.keywords || ''}`,query));
export const companySearchNumber = id => String(id).replace(/^U_/,'');
const rubles = value => value == null ? 'Нет данных' : new Intl.NumberFormat('ru-RU',{style:'currency',currency:'RUB',maximumFractionDigits:0}).format(value);

export function companySearchItems(data) {
  const p=data?.profile;
  if (!p) return [];
  const o=p.overview || {};
  const item=(title,text,target)=>({title,text,view:'company',target});
  return [
    item('Основные сведения',[p.name,p.industry,p.region,o.industry_code && `ОКВЭД ${o.industry_code}`].filter(Boolean).join(' · '),'overview'),
    item('Выручка и масштаб бизнеса',`${rubles(o.revenue)} · ${p.scale || o.size_class || 'Нет данных'}`,'metrics'),
    item('Продукты Альфа-Банка',(p.bank_products || []).join(' · ') || 'Нет отметок об использовании','products'),
    item('Внешнеторговые связи',[...(o.trade?.countries || []),o.trade?.in_customs_database === true ? 'Есть отметка ФТС' : o.trade?.in_customs_database === false ? 'Нет отметки ФТС' : 'Статус ФТС неизвестен'].join(' · '),'trade'),
    item('Группа компаний',[p.group,o.group_size && `Юрлиц: ${o.group_size}`,o.group_revenue != null && `Совокупная выручка: ${rubles(o.group_revenue)}`].filter(Boolean).join(' · '),'group'),
    ...(o.facts || []).map(fact=>item(({profile:'Команда и развитие',geo:'География компании',ownership:'Собственность',bank:'Банковское обслуживание',trade:'Внешнеторговые связи'})[fact.category] || 'Группа компаний',fact.text,
      fact.category.startsWith('group') ? 'group' : ({profile:'team',geo:'overview',ownership:'overview',bank:'products',trade:'trade'})[fact.category] || 'overview')),
  ];
}

export function paymentSearchItems(data) {
  if (!data?.summary) return [];
  return ['in','out'].flatMap(direction=>{
    const title=direction==='in' ? 'Поступления' : 'Списания';
    const total=data.summary[direction];
    const flow=data.flows?.[direction];
    return [{title:`${title} компании`,text:`${rubles(total.amount)} · ${total.count} платежей · средний платёж ${rubles(total.average)}`,view:'payments',direction,query:''},
      ...['industries','regions'].flatMap(key=>(flow?.[key] || []).map(row=>({title:`${title} · ${row.label}`,text:`${rubles(row.amount)} · ${new Intl.NumberFormat('ru-RU',{style:'percent',maximumFractionDigits:1}).format(row.share)} потока`,view:'payments',direction,query:''})))];
  });
}

export function counterpartyPaymentItems(rows) {
  return rows.flatMap(row=>['in','out'].filter(direction=>row[direction]?.count > 0).map(direction=>({
    title:`${direction==='in' ? 'Поступления от' : 'Списания в пользу'} ${row.name}`,
    text:`${rubles(row[direction].amount)} · ${row[direction].count} платежей`,view:'payments',direction,query:companySearchNumber(row.id),
  })));
}
