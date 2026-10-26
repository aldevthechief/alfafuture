import React, { memo, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { Button } from '@alfalab/core-components/button';
import { Status } from '@alfalab/core-components/status';
import { Tag } from '@alfalab/core-components/tag';
import { Tabs, Tab } from '@alfalab/core-components/tabs';
import { Typography } from '@alfalab/core-components/typography';
import { Input } from '@alfalab/core-components/input';
import { CompactCard } from './insightCard.jsx';
import { ALLOW_INFERENCE } from './cabinetMode.js';
import { findInsights, insightTags, toggleInsightTag } from './cabinetToolsState.js';
import { ExportControl } from './exportControl.jsx';
import { AnalysisPlaceholder } from './liveInsights.jsx';

const valid = s => s && !isNaN(new Date(s));
const date = s => valid(s) ? new Date(s).toLocaleDateString('ru-RU', { day: 'numeric', month: 'long' }) : '';
const time = s => valid(s) ? new Date(s).toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' }) : '';
const Title = props => <Typography.Title font="system" {...props} />;
const PAGE_SIZE = 8;

const LIVE = { loading: 'Загружаем анализ…', none: 'Анализа ещё нет', starting:'Запускаем анализ…', running: 'Идёт анализ новостей и аналитики Банка…',
               error: 'Не удалось получить анализ' };
function LiveBar({ live, run }) {
  if (live.state === 'off') return null;
  const errors = live.feed?.errors || [];
  return <div className="ip-live">
    {live.state !== 'ready' && <Typography.Text view="primary-small" color="secondary">{live.feed?.running_job?.stage?.label || LIVE[live.state]}</Typography.Text>}
    {live.state === 'ready' && errors.length > 0 && <Status color="orange" size={20}>Часть источников не обработана</Status>}
    {live.error && <Typography.Text view="primary-small" color="negative">{live.error}</Typography.Text>}
    {ALLOW_INFERENCE && <Button size={32} view="secondary" loading={['running','starting'].includes(live.state)} disabled={['loading','running','starting'].includes(live.state)} onClick={run}>
      {live.feed ? 'Обновить анализ' : 'Запустить анализ'}</Button>
    }
  </div>;
}

export default memo(function Page({ company, companyId, roleId, feed, role, live, run, visible, hidden, answers, answer, profileState, onOpen, onRestore, searchRequest }) {
  const [tab, setTab] = useState('feed');
  const [filters, setFilters] = useState([]);
  const [pinned, setPinned] = useState(false);
  const searchEdge = useRef(null);
  const [query, setQuery] = useState(searchRequest?.query || '');
  const [limit, setLimit] = useState(PAGE_SIZE);
  useEffect(() => setLimit(PAGE_SIZE), [filters, query]);
  useEffect(() => { if (searchRequest) {setQuery(searchRequest.query || '');setFilters([]);setTab('feed');} }, [searchRequest]);
  useEffect(()=>{
    setPinned(false);
    if (tab!=='feed' || !searchEdge.current) return;
    const observer=new IntersectionObserver(([entry])=>setPinned(!entry.isIntersecting && entry.boundingClientRect.top<12),{rootMargin:'-12px 0px 0px 0px',threshold:0});
    observer.observe(searchEdge.current);
    return ()=>observer.disconnect();
  },[tab]);
  const channels = [...new Set(visible.flatMap(insightTags))].filter(tag=>!['Риски','Возможности'].includes(tag));
  const shown = findInsights(visible,query,filters);
  const questions = company ? company.profile.hypotheses.filter(h => !answers[h.id]).length : 0;
  return <div className="ip">
    <div className="export-page-heading"><h1 className="h-sys">Инсайты</h1>
      {companyId && roleId && <ExportControl kind="insights" params={{company_id:companyId,role:roleId}}
        description="Вся сохранённая лента выбранной компании для этой роли, включая скрытые карточки. Фильтры поиска не ограничивают выгрузку." />}</div>
    <p className="h-sub">Что в новостях важно для {company?.profile.name} · {role}{feed && ` · обновлено ${date(feed.computed_at)} в ${time(feed.computed_at)}`}</p>
    <LiveBar live={live} run={run} />
    <Tabs selectedId={tab} onChange={(_, { selectedId }) => setTab(selectedId)} size="s">
      <Tab id="feed" title="События для вас">
        {tab === 'feed' && <>
        <div ref={searchEdge} className="ip-search-edge" aria-hidden="true"/>
        <section className={`card ip-search${pinned ? ' ip-search-pinned' : ''}`} aria-label="Поиск и теги инсайтов">
          <div className="ip-search-field"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true"><circle cx="10" cy="10" r="6.5" stroke="currentColor" strokeWidth="2"/><path d="m15 15 6 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/></svg>
            <input type="search" aria-label="Поиск по словам и тегам в инсайтах" placeholder="Найти в инсайтах: слово, фраза или #тег" value={query} maxLength={160} onChange={event=>setQuery(event.target.value)}/>
          </div>
        <div className="ip-search-tags"><span>Теги</span><div className="ip-filters" role="group" aria-label="Теги инсайтов">
          {[['all', 'Все'], ['risks', 'Риски'], ['opps', 'Возможности'], ...channels.map(c => [c, c])].map(([id, t]) =>
            <Tag key={id} size={32} checked={id==='all' ? !filters.length : filters.includes(id)} aria-pressed={id==='all' ? !filters.length : filters.includes(id)}
              onClick={() => setFilters(previous=>toggleInsightTag(previous,id))}>{t.charAt(0).toUpperCase() + t.slice(1)}</Tag>)}
        </div></div>
        <div className="ip-search-status"><span role="status">{query.trim() || filters.length ? 'Найдено' : 'В ленте'} · {shown.length}{filters.length>0 && ` · тегов: ${filters.length}`}</span>
          {(query || filters.length>0) && <button type="button" onClick={()=>{setQuery('');setFilters([]);}}>Сбросить поиск и теги</button>}</div>
        </section>
        <AnalysisPlaceholder live={live} />
        {shown.slice(0, limit).map(i => <CompactCard key={`${i.job_id || 'demo'}:${i.id}`} insight={i} onOpen={onOpen} />)}
        {shown.length > limit && <Button size={40} view="secondary" onClick={() => setLimit(n => n + PAGE_SIZE)}>Показать ещё</Button>}
        {!shown.length && !['running','starting'].includes(live.state) && <p className="ip-empty">{query.trim() || filters.length ? 'Ничего не найдено. Попробуйте другие слова или измените выбранные теги.' : 'Сейчас нет событий, важных для вашей роли.'}</p>}
        {hidden.length > 0 && <Button view="text" size={32} onClick={onRestore}>Показать скрытые ({hidden.length})</Button>}
        </>}
      </Tab>
      <Tab id="profile" title="Профиль компании" rightAddons={questions > 0 && <Status color="blue" size={20}>{questions}</Status>}>
        {tab === 'profile' && <Profile profile={company?.profile} answers={answers} onAnswer={answer} state={profileState} />}
      </Tab>
    </Tabs>
    <p className="ip-disclaimer">Сформировано AI на основе открытых новостей и аналитики Альфа-Банка. Не является инвестиционной рекомендацией.</p>
  </div>;
});

// what the bank knows about the company + HITL: agent hypotheses the data could not settle become optional
// «Верно ли, что …?» questions; answers make the exposure profile more precise
const VERDICT = { yes: 'да', no: 'нет' };
function ProfileMotion({ children }) {
  const panel = useRef(null);
  const content = useRef(null);
  const previous = useRef(null);
  const animation = useRef(null);
  useLayoutEffect(() => {
    const height = content.current.getBoundingClientRect().height;
    const from = animation.current?.playState === 'running' ? panel.current.getBoundingClientRect().height : previous.current;
    previous.current = height;
    if (from === null || Math.abs(from - height) < 1) return;
    animation.current?.cancel();
    if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      animation.current = panel.current.animate([{ height:`${from}px` }, { height:`${height}px` }],
        { duration:180, easing:'cubic-bezier(.2,.8,.2,1)' });
    }
  });
  useEffect(() => () => animation.current?.cancel(), []);
  return <div className="profile-motion" ref={panel}><div className="profile-motion-content" ref={content}>{children}</div></div>;
}

function Profile({ profile: p, answers, onAnswer, state = {} }) {
  const [later, setLater] = useState(false);
  const [other, setOther] = useState(null); // text while «Другое» is open
  if (!p) return null;
  const rest = p.hypotheses.filter(h => !answers[h.id]);
  const h = rest[0];
  const answered = p.hypotheses.filter(x => answers[x.id] && answers[x.id].verdict !== 'skip');
  const save = async a => { if (await onAnswer(h.id, a)) setOther(null); };
  const disabled = state.loading || state.busy || state.writable === false;
  const rows = [['Отрасль', p.industry], ['Масштаб', p.scale], ['Группа', p.group],
    ['Продукты Банка', p.bank_products.length ? p.bank_products.join(', ') : 'Нет отметок об использовании']];

  return <ProfileMotion>
    {state.loading && <p role="status">Загружаем профиль и ваши уточнения…</p>}
    {state.error && <div className="profile-error" role="alert"><p>{state.error}</p>
      <Button view="secondary" size={32} disabled={state.loading || state.busy} onClick={state.onRetry}>Повторить загрузку</Button></div>}
    {h && !later && state.writable !== false && <div className="card hitl">
      <div className="hitl-head">
        <Typography.Text view="primary-small" color="secondary">
          Помогите точнее подбирать события · вопрос {p.hypotheses.length - rest.length + 1} из {p.hypotheses.length}</Typography.Text>
        <Button view="text" size={32} disabled={disabled} onClick={() => setLater(true)}>Не сейчас</Button>
      </div>
      <div className="hitl-copy" key={h.id}>
      <Title tag="h3" view="small" weight="bold">{h.question}</Title>
      <Typography.Text view="primary-small" color="secondary" tag="p">Почему спрашиваем: {h.for}. {h.impact}</Typography.Text>
      </div>
      <Typography.Text view="primary-small" color="secondary" tag="p">Сохраним уточнение для вашей компании и учтём при следующем анализе.</Typography.Text>
      <span className="hitl-save-status" role="status" aria-live="polite">{state.busy ? 'Сохраняем уточнение…' : '\u00a0'}</span>
      {other === null
        ? <div className="iw-btns">
          <Button size={40} view="secondary" disabled={disabled} onClick={() => save({ verdict: 'yes' })}>Да</Button>
          <Button size={40} view="secondary" disabled={disabled} onClick={() => save({ verdict: 'no' })}>Нет</Button>
          <Button size={40} view="secondary" disabled={disabled} onClick={() => setOther('')}>Другое</Button>
          <Button size={40} view="text" disabled={disabled} onClick={() => save({ verdict: 'skip' })}>Пропустить</Button>
        </div>
        : <div className="iw-btns">
          <Input className="hitl-input" size={40} autoFocus placeholder="Как на самом деле?" value={other} maxLength={500} disabled={disabled}
            onChange={e => setOther(e.target.value)} onKeyDown={e => e.key === 'Enter' && other.trim() && save({ verdict: 'other', text: other.trim() })} />
          <Button size={40} view="primary" disabled={disabled || !other.trim()} onClick={() => save({ verdict: 'other', text: other.trim() })}>Сохранить</Button>
          <Button size={40} view="text" disabled={disabled} onClick={() => setOther(null)}>Отмена</Button>
        </div>}
    </div>}

    <div className="card prof">
      <Title tag="h3" view="xsmall" weight="bold">Как мы видим вашу компанию</Title>
      <dl>{rows.map(([k, v]) => <React.Fragment key={k}><dt>{k}</dt><dd>{v || (state.loading ? 'Загружаем…' : 'Нет данных')}</dd></React.Fragment>)}</dl>
      {p.channels.length > 0 && <><Title tag="h3" view="xsmall" weight="bold">Возможные каналы влияния новостей</Title>
        <p className="profile-note">По профилю компании. Применимость к конкретному событию проверяем в анализе.</p>
        <ul>{p.channels.map(c => <li key={c.name}><b>{c.name}</b> — {c.why}</li>)}</ul></>}
      {answered.length > 0 && <>
        <Title tag="h3" view="xsmall" weight="bold">Ваши уточнения</Title>
        <ul className="answers">{answered.map(x => {
          const a = answers[x.id];
          return <li key={x.id}>{x.text}: <b>{VERDICT[a.verdict] || a.text}</b>
            <Button view="text" size={32} disabled={disabled} onClick={async () => { if (await onAnswer(x.id, undefined)) setLater(false); }}>Изменить</Button></li>;
        })}</ul>
      </>}
      {later && rest.length > 0 && state.writable !== false && <Button view="text" size={32} onClick={() => setLater(false)}>Ответить на вопросы о компании ({rest.length})</Button>}
      {!state.loading && !state.error && !p.hypotheses.length && <p className="profile-note">В текущих инсайтах нет допущений, которые нужно уточнить у вас.</p>}
    </div>
  </ProfileMotion>;
}

