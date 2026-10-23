"""Incrementally extract supported news facts. Run explicitly; never inside a search request."""
import argparse
import json

from common.news_fact_index import FactStore
from common.search import news_pool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=None, help='maximum new/changed articles attempted this run')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error('--limit must be nonnegative')
    report = FactStore().refresh(news_pool(), limit=args.limit)
    print(json.dumps(report, ensure_ascii=False))
    return 1 if report['failed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
