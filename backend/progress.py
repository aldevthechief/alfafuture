"""Kirill's ordered checked-card collector, attached to the common job stage/event context.

News rows arrive only after writer/checker/audit. Bank rows arrive from the existing final merge,
after its cross-source deduplication and quota, so tentative bank cards never disappear later.
"""
import copy
import threading


class RunProgress:
    def __init__(self, publish, migrate):
        self.publish, self.migrate = publish, migrate
        self.lock = threading.Lock()
        self.rows = {}

    def __call__(self, event, **data):
        with self.lock:
            if event == 'row':
                self.rows[tuple(data['key'])] = self.migrate(copy.deepcopy(data['row']))
            elif event == 'ranked':
                for key, row in data.get('rows', []):
                    self.rows[tuple(key)] = self.migrate(copy.deepcopy(row))
            elif event == 'merged':
                self.publish(copy.deepcopy(data['rows']))
                return
            else:
                return
            self.publish([self.rows[key] for key in sorted(self.rows)])
