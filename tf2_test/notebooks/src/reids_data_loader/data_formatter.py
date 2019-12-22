import calendar
import itertools
from decimal import Decimal

month_dict = dict((k, v) for k, v in enumerate(calendar.month_abbr))


def format_amount(amount, currency, rounding='.01'):
    """Format amounts with appropriate decimals."""
    try:
        amount = float(amount)
    except Exception:
        return None
    # Currencies without decimal places (e.g., JP Yen).
    if currency in ['JPY', 'KRW', 'VND']:
        if amount == (amount // 1):
            return str(int(amount))
        else:
            return None
    return str(Decimal(amount).quantize(Decimal(rounding)))


def format_date(date,
                sep='',
                day_len=2,
                month_format='int',
                year_len=4,
                ordering='ymd'):
    """Format date in human-readable medium."""
    date = str(date)
    year = date[4 - year_len:4]
    month = date[4:6]
    if month_format == 'str':
        month = str(month_dict[int(month)])
    day = date[6:8]
    if day_len is None:
        day = str(int(day))
    if ordering == 'dmy':
        out_date = '{0}{3}{1}{3}{2}'.format(month, day, year, sep)
    elif ordering == 'mdy':
        out_date = '{0}{3}{1}{3}{2}'.format(day, month, year, sep)
    else:        
        out_date = '{0}{3}{1}{3}{2}'.format(year, month, day, sep)
    if sep == ' ' and month_format == 'str':
        # Format dates with string months to have comma.
        out_date = ', '.join(out_date.rsplit(' ', 1))
    return out_date


def get_date_params():
    """Get list of date parameter combinations."""
    param_dict = {
        'sep': ['', ' ', '-', '/'],
        'day_len': [None, 2],
        'month_format': ['int', 'str'],
        'year_len': [2, 4],
        'ordering': ['dmy', 'mdy', 'ymd']
    }
    param_list = [
        dict(itertools.zip_longest(param_dict, v))
        for v in itertools.product(*param_dict.values())
    ]
    return param_list
