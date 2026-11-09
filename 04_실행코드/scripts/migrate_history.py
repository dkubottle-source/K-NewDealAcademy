"""Back up and migrate an existing local history, without changing old values."""
from datetime import datetime, timezone
import sqlite3
from contextlib import closing
from src.monitoring import HISTORY_DB, history_connection


def run():
    if HISTORY_DB.exists():
        backup = HISTORY_DB.with_name('detection_history_before_migration_' +
                    datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f') + '.sqlite3')
        with closing(sqlite3.connect(HISTORY_DB)) as source, closing(sqlite3.connect(backup)) as target:
            source.backup(target)
        print('Backup:', backup)
    with history_connection() as connection:
        print('Preserved records:', connection.execute('SELECT COUNT(*) FROM detections').fetchone()[0])


if __name__ == '__main__':
    run()
