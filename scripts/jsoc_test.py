from sunpy.net import Fido, attrs as a
from sunpy.net.jsoc import JSOCClient
from datetime import datetime, timedelta
import os
import astropy.units as u

client = JSOCClient()

# 设置时间范围
start_time = datetime(2011, 1, 21, 18, 53, 0)
end_time = start_time + timedelta(minutes=20)

# 查询条件
query = client.search(
    a.Time(start_time, end_time),
    a.jsoc.Series('aia.lev1_euv_12s'),
    a.jsoc.Wavelength(94 * u.angstrom),
    a.jsoc.Segment('image'),
    a.jsoc.Notify('kyang30@student.gsu.edu')  # ⚠️ 改成你自己的邮箱
)

print(f"Found {len(query)} results.")

# 下载目录
download_dir = os.path.abspath("./jsoc_data")
os.makedirs(download_dir, exist_ok=True)

# 下载
if query:
    print("Requesting download...")
    resp = client.fetch(query, path=os.path.join(download_dir, '{file}'))
    print("Download complete.")
else:
    print("No data found.")

