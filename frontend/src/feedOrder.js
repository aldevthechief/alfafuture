// Ordering never filters a card: new arrivals and unknown IDs retain their existing relative order.
export function applyTopOrder(feed,order) {
  if (!feed || !Array.isArray(order)) return feed;
  const positions=new Map();
  for (const id of order) if (typeof id==='string' && !positions.has(id)) positions.set(id,positions.size);
  return {...feed,insights:[...feed.insights].sort((a,b)=>(positions.get(a.id) ?? positions.size)-(positions.get(b.id) ?? positions.size))};
}
