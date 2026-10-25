import React, { memo, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { Button } from '@alfalab/core-components/button';
import { Status } from '@alfalab/core-components/status';
import { Tag } from '@alfalab/core-components/tag';
import { Tabs, Tab } from '@alfalab/core-components/tabs';
import { Typography } from '@alfalab/core-components/typography';
import { Input } from '@alfalab/core-components/input';
import { CompactCard } from './insightCard.jsx';
import { ALLOW_INFERENCE } from './cabinetMode.js';

const valid = s => s && !isNaN(new Date(s));
const date = s => valid(s) ? new Date(s).toLocaleDateString('ru-RU', { day: 'numeric', month: 'long' }) : '';
const time = s => valid(s) ? new Date(s).toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' }) : '';
const Title = props => <Typography.Title font="system" {...props} />;
const PAGE_SIZE = 8;

const LIVE = { loading: 'Загружаем анализ…', none: 'Анализа ещё нет', running: 'Идёт анализ новостей и аналитики Банка…',
               error: 'Не удалось получить анализ' };
function LiveBar({ live, run }) {
  if (live.state === 'off') return null;
  const errors = live.feed?.errors || [];
  return <div className="ip-live">
    {live.state !== 'ready' && <Typography.Text view="primary-small" color="secondary">{LIVE[live.state]}</Typography.Text>}
    {live.state === 'ready' && errors.length > 0 && <Status color="orange" size={20}>Часть источников не обработана</Status>}
    {live.error && <Typography.Text view="primary-small" color="negative">{live.error}</Typography.Text>}
    {ALLOW_INFERENCE && <Button size={32} view="secondary" loading={live.state === 'running'} disabled={live.state === 'loading'} onClick={run}>
      {live.feed ? 'Обновить анализ' : 'Запустить анализ'}</Button>
    }
  </div>;
}

export default memo(function Page({ company, feed, role, live, run, visible, hidden, answers, answer, profileState, onOpen, onRestore }) {
  const [tab, setTab] = useState('feed');
  const [filter, setFilter] = useState('all');
  const [limit, setLimit] = useState(PAGE_SIZE);
  useEffect(() => setLimit(PAGE_SIZE), [feed, filter]);
  const channels = [...new Set(visible.map(i => i.channel))];
  const shown = visible.filter(i => filter === 'all' || (filter === 'risks' ? (i.counts?.risks ?? i.risks?.length) : filter === 'opps' ? (i.counts?.opportunities ?? i.opportunities?.length) : i.channel === filter));
  const questions = company ? company.profile.hypotheses.filter(h => !answers[h.id]).length : 0;
  return <div className="ip">
    <h1 className="h-sys">Инсайты</h1>
    <p className="h-sub">Что в новостях важно для {company?.profile.name} · {role}{feed && ` · обновлено ${date(feed.computed_at)} в ${time(feed.computed_at)}`}</p>
    <LiveBar live={live} run={run} />
    <Tabs selectedId={tab} onChange={(_, { selectedId }) => setTab(selectedId)} size="s">
      <Tab id="feed" title="События для вас">
        {tab === 'feed' && <>
        <div className="ip-filters">
          {[['all', 'Все'], ['risks', 'Риски'], ['opps', 'Возможности'], ...channels.map(c => [c, c])].map(([id, t]) =>
            <Tag key={id} size={32} checked={filter === id} onClick={() => setFilter(id)}>{t}</Tag>)}
        </div>
        {shown.slice(0, limit).map(i => <CompactCard key={i.id} insight={i} onOpen={onOpen} />)}
        {shown.length > limit && <Button size={40} view="secondary" onClick={() => setLimit(n => n + PAGE_SIZE)}>Показать ещё</Button>}
        {!shown.length && <p className="ip-empty">Сейчас нет событий, важных для вашей роли.</p>}
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

