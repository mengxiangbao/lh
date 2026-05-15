# Tushare 统一初始化方式

本项目所有 Tushare 数据接口统一从下面这个文件初始化：

```text
dragon_backtest/tushare_client.py
```

等价调用方式：

```python
import tushare as ts

pro = ts.pro_api(token)
pro._DataApi__http_url = "http://101.35.233.113:8020/"
```

项目代码中使用：

```python
from dragon_backtest.tushare_client import init_tushare, pro_bar

pro = init_tushare()
df = pro.index_basic(limit=5)

bar = pro_bar(ts_code="000001.SZ", limit=3)
```

Token 默认从环境变量读取：

```powershell
$env:TUSHARE_TOKEN="你的token"
$env:TUSHARE_HTTP_URL="http://101.35.233.113:8020/"
```

注意：

- 不建议把 token 明文写入 Python 源码。
- 如果接口提示 Token 不对，优先检查 `TUSHARE_HTTP_URL` 是否为 `http://101.35.233.113:8020/`。
- `fetch-tushare`、后续日线数据下载、`pro_bar` 调用都应复用 `dragon_backtest.tushare_client`。
