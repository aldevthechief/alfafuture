"""The product's clock. FEED_NOW=YYYY-MM-DD freezes it (the demo stream is frozen at its last hour; tests mock another day)."""
import os
from datetime import date


def today() -> date:
    return date.fromisoformat(os.environ['FEED_NOW'][:10]) if os.environ.get('FEED_NOW') else date.today()
