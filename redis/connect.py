
import redis

r = redis.Redis(
    host="127.0.0.1",
    port=6379,
    password="123456",
    db=0,
    decode_responses=True,
    protocol=2   # 关键！强制使用 RESP2 老协议，不再发送 HELLO
)

print(r.ping())
