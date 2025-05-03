from sunpy.net import Fido, attrs as a
from datetime import datetime, timedelta
import astropy.units as u

# 取某个事件的时间段
t_center = datetime(2010, 1, 15, 8, 41)
t_start = t_center - timedelta(minutes=10)
t_end = t_center + timedelta(minutes=10)

# 不加 a.Sample
result = Fido.search(
    a.Time(t_start, t_end),
    a.Instrument("AIA"),
    a.Wavelength(94 * u.angstrom)
)

print(result)